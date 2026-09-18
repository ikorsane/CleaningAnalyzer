import io, json, zipfile, tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from analysis_backend import (
    rectify, lab_float, roi_mask, robust_lab, spatial_dirty_model,
    cleaning_fraction, originally_soiled_mask, propose_footprint,
    bottom_exclusion_mask, metrics, heatmap_overlay, save_report_workbook,
)

st.set_page_config(page_title="Cleaning Analyzer", page_icon="🧪", layout="wide")

CSS='''
<style>
.block-container{max-width:1450px;padding-top:2rem;padding-bottom:4rem}
h1,h2,h3{color:#17365D}.stButton>button,.stDownloadButton>button{border-radius:7px;font-weight:600}
[data-testid="stMetric"]{background:#f7f9fc;border:1px solid #e3e8ef;padding:12px;border-radius:8px}
.smallnote{color:#667085;font-size:.9rem}.step{font-weight:700;color:#4472C4;letter-spacing:.02em}
</style>'''
st.markdown(CSS, unsafe_allow_html=True)

if 'exp' not in st.session_state:
    st.session_state.exp={'n_products':4,'files':{},'results':[],'configured':{},'accepted':{}}

def decode(upload):
    b=np.frombuffer(upload.getvalue(),np.uint8)
    return cv2.imdecode(b,cv2.IMREAD_COLOR)

def pil_bgr(img): return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def display_fit(img,maxw=1050):
    h,w=img.shape[:2]; s=min(1.0,maxw/w); return cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_AREA),s

def canvas_objects_to_rects(objs,scale):
    out=[]
    for o in objs or []:
        if o.get('type')=='rect':
            x=o.get('left',0)/scale; y=o.get('top',0)/scale
            w=o.get('width',0)*o.get('scaleX',1)/scale; h=o.get('height',0)*o.get('scaleY',1)/scale
            out.append((int(x),int(y),int(w),int(h)))
    return out

def polygon_from_canvas(objs,shape,scale):
    H,W=shape[:2]; m=np.zeros((H,W),np.uint8)
    paths=[]
    for o in objs or []:
        if o.get('type')=='path' and o.get('path'):
            pts=[]
            for cmd in o['path']:
                if len(cmd)>=3 and cmd[0] in ('M','L'):
                    pts.append([cmd[1]/scale,cmd[2]/scale])
            if len(pts)>=3: paths.append(np.array(pts,np.int32))
        elif o.get('type')=='polygon' and o.get('points'):
            left=o.get('left',0); top=o.get('top',0); sx=o.get('scaleX',1); sy=o.get('scaleY',1)
            pts=np.array([[(left+p['x']*sx)/scale,(top+p['y']*sy)/scale] for p in o['points']],np.int32)
            if len(pts)>=3: paths.append(pts)
    if paths: cv2.fillPoly(m,paths,255)
    return m

def click_points_from_canvas(objs,scale):
    pts=[]
    for o in objs or []:
        if o.get('type') in ('circle','ellipse'):
            x=(o.get('left',0)+o.get('radius',5))/scale; y=(o.get('top',0)+o.get('radius',5))/scale
            pts.append((x,y))
    return pts

def mask_overlay(img,mask):
    out=img.copy(); cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out,cnts,-1,(0,0,255),3); return out

def prepare_plate(img, cfg):
    plate=rectify(img,np.float32(cfg['corners'])); lab=lab_float(plate)
    dirty_m=roi_mask(plate.shape,tuple(cfg['dirty'])); clean_m=roi_mask(plate.shape,tuple(cfg['clean'])); clean_lab=robust_lab(lab,clean_m)
    excluded=clean_m.copy()
    for r in cfg['guides']: excluded=cv2.bitwise_or(excluded,roi_mask(plate.shape,tuple(r)))
    excluded[dirty_m>0]=0
    dirty_model=spatial_dirty_model(lab,excluded); frac=cleaning_fraction(lab,dirty_model,clean_lab)
    soil=originally_soiled_mask(dirty_model,clean_lab); valid=bottom_exclusion_mask(plate.shape)
    return plate,dirty_model,frac,soil,valid

def build_outputs():
    exp=st.session_state.exp; rows=[]; files={}
    for ri,name in enumerate(exp['files'],1):
        img=exp['files'][name]; cfg=exp['configured'][name]; plate,dirty_model,frac,soil,valid=prepare_plate(img,cfg)
        prefix=f"Replicate_{ri}"
        files[f'{prefix}/rectified_plate.png']=cv2.imencode('.png',plate)[1].tobytes()
        dirty_vis=cv2.cvtColor(np.clip(dirty_model,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
        files[f'{prefix}/estimated_dirty_baseline.png']=cv2.imencode('.png',dirty_vis)[1].tobytes()
        files[f'{prefix}/estimated_original_soil_mask.png']=cv2.imencode('.png',soil)[1].tobytes()
        files[f'{prefix}/valid_plate_mask_bottom_excluded.png']=cv2.imencode('.png',valid)[1].tobytes()
        for i in range(exp['n_products']):
            key=(name,i); accepted=exp['accepted'][key]
            am=cv2.bitwise_and(cv2.bitwise_and(accepted,soil),valid)
            pname=f"Product {chr(65+i)}"; row=metrics(pname,frac,am); row['Replicate']=ri; rows.append(row)
            stem=pname.replace(' ','_')
            files[f'{prefix}/{stem}_contact_mask.png']=cv2.imencode('.png',accepted)[1].tobytes()
            files[f'{prefix}/{stem}_analysis_mask.png']=cv2.imencode('.png',am)[1].tobytes()
            files[f'{prefix}/{stem}_footprint.png']=cv2.imencode('.png',heatmap_overlay(plate,frac,am,pname))[1].tobytes()
    df=pd.DataFrame(rows); df=df[['Replicate','Product']+[c for c in df.columns if c not in ('Replicate','Product')]]
    with tempfile.TemporaryDirectory() as td:
        x=Path(td)/'results.xlsx'; save_report_workbook(df,x); xbytes=x.read_bytes()
    files['results.csv']=df.to_csv(index=False).encode(); files['results.xlsx']=xbytes
    z=io.BytesIO()
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zz:
        for p,b in files.items(): zz.writestr(p,b)
    return df,xbytes,z.getvalue()

st.title("Cleaning Analyzer")
st.caption("Image-based comparison of cleaning performance across products and independent replicate plates")

with st.sidebar:
    st.header("Experiment")
    n=st.number_input("Number of product tracks",1,12,value=st.session_state.exp['n_products'],step=1)
    st.session_state.exp['n_products']=int(n)
    st.info("The bottom 5% of the rectified plate is excluded from scoring because product can pool against the rack/stand.")
    if st.button("Reset experiment",use_container_width=True):
        st.session_state.exp={'n_products':int(n),'files':{},'results':[],'configured':{},'accepted':{}}; st.rerun()

st.markdown('<div class="step">STEP 1 · REPLICATES</div>',unsafe_allow_html=True)
uploads=st.file_uploader("Upload all replicate plate photographs",type=['jpg','jpeg','png','bmp','tif','tiff'],accept_multiple_files=True)
if uploads:
    for u in uploads:
        if u.name not in st.session_state.exp['files']: st.session_state.exp['files'][u.name]=decode(u)

names=list(st.session_state.exp['files'])
if not names:
    st.stop()
st.write(f"**{len(names)} replicate(s) loaded:** "+", ".join(names))

st.markdown('<div class="step">STEP 2 · PLATE SETUP</div>',unsafe_allow_html=True)
st.caption("Configure each replicate. The same products must appear in the same left-to-right order.")

for ri,name in enumerate(names,1):
    img=st.session_state.exp['files'][name]
    with st.expander(f"Replicate {ri} — {name}",expanded=name not in st.session_state.exp['configured']):
        st.markdown("**A. Plate corners** — choose *point* mode and place exactly four points: top-left, top-right, bottom-right, bottom-left.")
        shown,s=display_fit(img)
        c=st_canvas(fill_color='rgba(68,114,196,0.5)',stroke_width=2,stroke_color='#D62728',background_image=pil_bgr(shown),update_streamlit=True,height=shown.shape[0],width=shown.shape[1],drawing_mode='point',point_display_radius=6,key=f'corners_{name}')
        corners=click_points_from_canvas((c.json_data or {}).get('objects',[]),s)
        if len(corners)!=4:
            st.warning(f"Place exactly 4 corner points. Currently: {len(corners)}")
            continue
        plate=rectify(img,np.float32(corners)); pshow,ps=display_fit(plate)
        st.markdown(f"**B. Controls + {int(n)} product guides** — draw rectangles in this exact order: **dirty control, clean control, Product A, Product B…**")
        rc=st_canvas(fill_color='rgba(68,114,196,0.12)',stroke_width=3,stroke_color='#4472C4',background_image=pil_bgr(pshow),update_streamlit=True,height=pshow.shape[0],width=pshow.shape[1],drawing_mode='rect',key=f'rois_{name}')
        rects=canvas_objects_to_rects((rc.json_data or {}).get('objects',[]),ps)
        need=2+int(n)
        st.caption(f"Rectangles: {len(rects)} / {need}")
        if len(rects)==need:
            if st.button(f"Save setup for replicate {ri}",key=f'save_{name}'):
                st.session_state.exp['configured'][name]={'corners':corners,'dirty':rects[0],'clean':rects[1],'guides':rects[2:]}
                # invalidate previous accepted masks for this replicate
                for i in range(int(n)): st.session_state.exp['accepted'].pop((name,i),None)
                st.rerun()

if len(st.session_state.exp['configured'])<len(names):
    st.info("Finish and save the setup for every replicate to continue.")
    st.stop()

st.markdown('<div class="step">STEP 3 · FOOTPRINT VERIFICATION</div>',unsafe_allow_html=True)
st.caption("The red outline is the automatic contact-footprint proposal. Accept it when correct; otherwise draw a manual polygon around the actual wetted/contacted track.")

for ri,name in enumerate(names,1):
    plate,dirty_model,frac,soil,valid=prepare_plate(st.session_state.exp['files'][name],st.session_state.exp['configured'][name])
    st.subheader(f"Replicate {ri}")
    cols=st.columns(2)
    for i,r in enumerate(st.session_state.exp['configured'][name]['guides']):
        pname=f"Product {chr(65+i)}"; key=(name,i); proposal=propose_footprint(frac,tuple(r))
        with cols[i%2]:
            st.markdown(f"**{pname}**")
            if key not in st.session_state.exp['accepted']:
                st.image(pil_bgr(mask_overlay(plate,proposal)),caption="Automatic proposal",use_container_width=True)
                a,b=st.columns(2)
                if a.button("Accept proposal",key=f'acc_{ri}_{i}',use_container_width=True):
                    st.session_state.exp['accepted'][key]=proposal; st.rerun()
                if b.button("Correct manually",key=f'corr_{ri}_{i}',use_container_width=True):
                    st.session_state[f'manual_{ri}_{i}']=True
                if st.session_state.get(f'manual_{ri}_{i}',False):
                    st.caption("Use polygon mode: click around the actual footprint and double-click to close it.")
                    pshow,ps=display_fit(plate,maxw=650)
                    mc=st_canvas(fill_color='rgba(214,39,40,0.15)',stroke_width=3,stroke_color='#D62728',background_image=pil_bgr(pshow),update_streamlit=True,height=pshow.shape[0],width=pshow.shape[1],drawing_mode='polygon',key=f'poly_{ri}_{i}')
                    mm=polygon_from_canvas((mc.json_data or {}).get('objects',[]),plate.shape,ps)
                    if np.count_nonzero(mm)>0 and st.button("Use manual footprint",key=f'usepoly_{ri}_{i}'):
                        st.session_state.exp['accepted'][key]=mm; st.session_state[f'manual_{ri}_{i}']=False; st.rerun()
            else:
                st.image(pil_bgr(mask_overlay(plate,st.session_state.exp['accepted'][key])),caption="Accepted footprint",use_container_width=True)
                if st.button("Change footprint",key=f'change_{ri}_{i}'):
                    st.session_state.exp['accepted'].pop(key,None); st.rerun()

expected=len(names)*int(n)
if len(st.session_state.exp['accepted'])<expected:
    st.info(f"Verify all footprints to continue ({len(st.session_state.exp['accepted'])}/{expected}).")
    st.stop()

st.markdown('<div class="step">STEP 4 · RESULTS</div>',unsafe_allow_html=True)
df,xlsx,zipbytes=build_outputs()
summary=df.groupby('Product',sort=False).agg(n=('Replicate','count'),mean_cleaning=('Mean cleaning depth (%)','mean'),sd_cleaning=('Mean cleaning depth (%)','std'),coverage50=('Footprint >=50% cleaned (%)','mean'),coverage75=('Footprint >=75% cleaned (%)','mean'),footprint=('Footprint area (px)','mean')).reset_index()
summary['Rank']=summary['mean_cleaning'].rank(ascending=False,method='min').astype(int)

st.subheader("Cleaning performance across replicate plates")
st.dataframe(summary.rename(columns={'mean_cleaning':'Mean cleaning (%)','sd_cleaning':'SD (%)','coverage50':'Area ≥50% cleaned (%)','coverage75':'Area ≥75% cleaned (%)','footprint':'Mean footprint area (px)'}),hide_index=True,use_container_width=True,column_config={'Mean cleaning (%)':st.column_config.NumberColumn(format='%.1f'),'SD (%)':st.column_config.NumberColumn(format='%.1f'),'Area ≥50% cleaned (%)':st.column_config.NumberColumn(format='%.1f'),'Area ≥75% cleaned (%)':st.column_config.NumberColumn(format='%.1f'),'Mean footprint area (px)':st.column_config.NumberColumn(format='%.0f')})

chart=summary.set_index('Product')[['mean_cleaning']].rename(columns={'mean_cleaning':'Mean cleaning (%)'})
st.bar_chart(chart,y_label='Cleaning (%)',height=360)
coverage=summary.set_index('Product')[['coverage50','coverage75']].rename(columns={'coverage50':'≥50% cleaned','coverage75':'≥75% cleaned'})
st.bar_chart(coverage,y_label='Footprint meeting threshold (%)',height=360)

c1,c2=st.columns(2)
c1.download_button("Download Excel report",xlsx,file_name='Cleaning_Analyzer_results.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
c2.download_button("Download complete analysis ZIP",zipbytes,file_name='Cleaning_Analyzer_analysis.zip',mime='application/zip',use_container_width=True)
st.caption("Across-plate values are arithmetic means with each replicate weighted equally. SD is the sample standard deviation between replicate plates; with n=1, SD is not reported. Footprint area is a contact/wetting diagnostic, not cleaning efficacy by itself.")
