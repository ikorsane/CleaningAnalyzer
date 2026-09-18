import io, json, zipfile, tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
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

def _click_signature(v):
    if not v: return None
    return (v.get("x"), v.get("y"), v.get("unix_time"))

def collect_click(image, key, state_key, max_points=None):
    """Display image and accumulate click coordinates across Streamlit reruns."""
    pts=st.session_state.setdefault(state_key, [])
    v=streamlit_image_coordinates(image, key=key)
    sig=_click_signature(v)
    last_key=state_key+"_last"
    if sig and sig != st.session_state.get(last_key):
        st.session_state[last_key]=sig
        if max_points is None or len(pts)<max_points:
            pts.append((float(v["x"]), float(v["y"])))
    return pts

def draw_points(img, pts, labels=None):
    out=img.copy()
    for i,(x,y) in enumerate(pts):
        cv2.circle(out,(int(x),int(y)),8,(0,0,255),-1)
        txt=(labels[i] if labels and i<len(labels) else str(i+1))
        cv2.putText(out,txt,(int(x)+10,int(y)-8),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2,cv2.LINE_AA)
    return out

def rect_from_two_points(a,b):
    x1,y1=a; x2,y2=b
    x,y=min(x1,x2),min(y1,y2)
    return (int(x),int(y),max(1,int(abs(x2-x1))),max(1,int(abs(y2-y1))))

def draw_rects(img, rects, labels=None):
    out=img.copy()
    for i,(x,y,w,h) in enumerate(rects):
        cv2.rectangle(out,(x,y),(x+w,y+h),(196,114,68),3)
        if labels and i<len(labels):
            cv2.putText(out,labels[i],(x+5,max(22,y+24)),cv2.FONT_HERSHEY_SIMPLEX,.7,(196,114,68),2,cv2.LINE_AA)
    return out

def polygon_mask(shape, pts):
    m=np.zeros(shape[:2],np.uint8)
    if len(pts)>=3: cv2.fillPoly(m,[np.array(pts,np.int32)],255)
    return m

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
        
        for k in list(st.session_state.keys()):
            if k != 'exp': del st.session_state[k]
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
        st.markdown("**A. Plate corners** — click the four corners in this order: **top-left, top-right, bottom-right, bottom-left**.")
        shown,s=display_fit(img)
        corner_state=f"corner_pts_{name}"
        corner_pts=st.session_state.setdefault(corner_state,[])
        corner_vis=draw_points(shown,corner_pts,["TL","TR","BR","BL"])
        corners_disp=collect_click(pil_bgr(corner_vis),f"corners_{name}",corner_state,max_points=4)
        b1,b2=st.columns([1,3])
        if b1.button("Undo corner",key=f"undo_corner_{name}",disabled=not corners_disp):
            corners_disp.pop(); st.session_state.pop(corner_state+"_last",None); st.rerun()
        b2.caption(f"Corner points: {len(corners_disp)} / 4")
        if len(corners_disp)!=4:
            st.warning(f"Place exactly 4 corner points. Currently: {len(corners_disp)}")
            continue
        corners=[(x/s,y/s) for x,y in corners_disp]
        plate=rectify(img,np.float32(corners)); pshow,ps=display_fit(plate)

        need=2+int(n)
        labels=["Dirty control","Clean control"]+[f"Product {chr(65+i)}" for i in range(int(n))]
        st.markdown(f"**B. Controls + {int(n)} product guides** — define each area below. For every area, click **two opposite corners** of the rectangle.")
        rect_state=f"rects_{name}"
        rects=st.session_state.setdefault(rect_state,[])
        current=len(rects)
        if current<need:
            st.info(f"Now select: **{labels[current]}** ({current+1}/{need})")
            pair_state=f"rect_pair_{name}_{current}"
            pair=st.session_state.setdefault(pair_state,[])
            rect_vis=draw_rects(pshow,rects,labels)
            rect_vis=draw_points(rect_vis,pair,["1","2"])
            pair=collect_click(pil_bgr(rect_vis),f"rect_click_{name}_{current}",pair_state,max_points=2)
            if len(pair)==2:
                rr=rect_from_two_points((pair[0][0]/ps,pair[0][1]/ps),(pair[1][0]/ps,pair[1][1]/ps))
                rects.append(rr)
                st.session_state.pop(pair_state,None); st.session_state.pop(pair_state+"_last",None); st.rerun()
        else:
            st.image(pil_bgr(draw_rects(pshow,rects,labels)),caption="Selected controls and product guides",width="stretch")
        r1,r2=st.columns([1,3])
        if r1.button("Undo last area",key=f"undo_rect_{name}",disabled=not rects):
            rects.pop(); st.rerun()
        r2.caption(f"Areas: {len(rects)} / {need}")
        if len(rects)==need:
            if st.button(f"Save setup for replicate {ri}",key=f"save_{name}",type="primary"):
                st.session_state.exp['configured'][name]={'corners':corners,'dirty':rects[0],'clean':rects[1],'guides':rects[2:]}
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
                    st.caption("Click around the actual footprint. Use at least 3 points, then press **Use manual footprint**.")
                    pshow,ps=display_fit(plate,maxw=650)
                    poly_state=f"poly_pts_{ri}_{i}"
                    poly_pts=st.session_state.setdefault(poly_state,[])
                    pvis=draw_points(pshow,poly_pts)
                    if len(poly_pts)>=2:
                        cv2.polylines(pvis,[np.array(poly_pts,np.int32)],False,(0,0,255),3)
                    poly_pts=collect_click(pil_bgr(pvis),f'poly_{ri}_{i}',poly_state)
                    pc1,pc2=st.columns(2)
                    if pc1.button("Undo point",key=f'undopoly_{ri}_{i}',disabled=not poly_pts):
                        poly_pts.pop(); st.session_state.pop(poly_state+"_last",None); st.rerun()
                    if pc2.button("Clear points",key=f'clearpoly_{ri}_{i}',disabled=not poly_pts):
                        st.session_state[poly_state]=[]; st.session_state.pop(poly_state+"_last",None); st.rerun()
                    mm=polygon_mask(plate.shape,[(x/ps,y/ps) for x,y in poly_pts])
                    if len(poly_pts)>=3 and st.button("Use manual footprint",key=f'usepoly_{ri}_{i}',type="primary"):
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
