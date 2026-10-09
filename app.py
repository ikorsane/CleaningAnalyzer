import io, json, zipfile, tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates
from analysis_backend import (
    rectify, lab_float, roi_mask, robust_lab, spatial_dirty_model,
    cleaning_fraction, originally_soiled_mask, propose_footprint,
    bottom_exclusion_mask, metrics, heatmap_overlay, save_report_workbook,
    FOOTPRINT_ALGORITHM_VERSION, lane_guides,
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
# A new boundary algorithm invalidates previously cached automatic proposals.
# Retain uploads and ROI selections so nobody has to mark plate corners again.
if st.session_state.get('footprint_algorithm_version') != FOOTPRINT_ALGORITHM_VERSION:
    exp = st.session_state.exp
    for name, config in list(exp.get('configured', {}).items()):
        if 'tips' not in config:
            # Preserve already selected dirty/clean reference areas; only the
            # product tops need selecting under the new workflow.
            if 'dirty' in config and 'clean' in config:
                st.session_state[f'controls_rects_{name}'] = [config['dirty'], config['clean']]
            exp['configured'].pop(name, None)
    exp['accepted'] = {}
    st.session_state['footprints_confirmed'] = False
    st.session_state['footprint_algorithm_version'] = FOOTPRINT_ALGORITHM_VERSION


def decode(upload):
    b=np.frombuffer(upload.getvalue(),np.uint8)
    return cv2.imdecode(b,cv2.IMREAD_COLOR)

def pil_bgr(img): return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def display_fit(img,maxw=650):
    h,w=img.shape[:2]; s=min(1.0,maxw/w); return cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_AREA),s

def _click_signature(v):
    if not v: return None
    return (v.get("x"), v.get("y"), v.get("unix_time"))

def collect_click(image, key, state_key, max_points=None):
    """Display a clickable image and accumulate coordinates without forcing an extra rerun.

    The component value from the previous click is processed *before* the widget is
    rendered. This lets an annotated image be supplied on the same rerun while avoiding
    the explicit st.rerun() loop that could make the component disappear.
    """
    pts=st.session_state.setdefault(state_key, [])
    last_key=state_key+"_last"

    # Custom-component values are kept in session state under their widget key.
    # Process that value first so the current click is available immediately on rerun.
    prev=st.session_state.get(key)
    sig=_click_signature(prev) if isinstance(prev,dict) else None
    if sig and sig != st.session_state.get(last_key):
        st.session_state[last_key]=sig
        if max_points is None or len(pts)<max_points:
            pts.append((float(prev["x"]), float(prev["y"])))

    # Supplying an explicit width makes iframe sizing reliable on Streamlit Cloud.
    width=getattr(image,"width",None)
    v=streamlit_image_coordinates(image, key=key, width=width, cursor="crosshair")

    # Fallback for component versions where the value is not exposed in session_state
    # until after the component call. Do not force a rerun; the click itself already
    # triggers Streamlit's normal rerun cycle.
    sig=_click_signature(v)
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

def draw_rects(img, rects, labels=None, scale=1.0):
    """Draw stored full-resolution rectangles at the correct display scale."""
    out=img.copy()
    for i,(x,y,w,h) in enumerate(rects):
        dx=int(round(x*scale)); dy=int(round(y*scale))
        dw=int(round(w*scale)); dh=int(round(h*scale))
        cv2.rectangle(out,(dx,dy),(dx+dw,dy+dh),(196,114,68),3)
        if labels and i<len(labels):
            cv2.putText(out,labels[i],(dx+5,max(22,dy+24)),cv2.FONT_HERSHEY_SIMPLEX,.7,(196,114,68),2,cv2.LINE_AA)
    return out

def polygon_mask(shape, pts):
    m=np.zeros(shape[:2],np.uint8)
    if len(pts)>=3: cv2.fillPoly(m,[np.array(pts,np.int32)],255)
    return m

def mask_overlay(img,mask):
    out=img.copy(); cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out,cnts,-1,(0,0,255),3); return out



def heatmap_plate_overlay(plate, frac, footprint_masks, alpha=0.55):
    """Unannotated heatmap, coloured only inside the final analysis masks."""
    base=plate.copy()
    vals=np.clip(np.nan_to_num(frac,nan=0.0,posinf=1.0,neginf=0.0),0.0,1.0)
    heat=cv2.applyColorMap(np.uint8(np.round(vals*255)),cv2.COLORMAP_TURBO)

    union=np.zeros(plate.shape[:2],dtype=bool)
    for m in footprint_masks:
        if m is not None:
            union |= (m>0)

    if np.any(union):
        base_f=base.astype(np.float32)
        heat_f=heat.astype(np.float32)
        base_f[union]=(1.0-alpha)*base_f[union] + alpha*heat_f[union]
        base=np.clip(base_f,0,255).astype(np.uint8)
    return base

def crop_pooling_band(img, fraction=0.05):
    """Physically remove the bottom pooling band from the rectified plate."""
    h=img.shape[0]
    keep=max(1,int(round(h*(1.0-fraction))))
    return img[:keep].copy()

def fit_mask_to_plate(mask, plate_shape):
    """Keep masks compatible if the app reruns after introducing the physical crop."""
    h,w=plate_shape[:2]
    if mask.shape[:2]==(h,w): return mask
    if mask.shape[1]==w and mask.shape[0]>=h: return mask[:h].copy()
    return cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)

def png_bytes(img):
    ok,b=cv2.imencode('.png',img)
    if not ok: raise RuntimeError('Could not encode PNG')
    return b.tobytes()

def prepare_plate(img, cfg):
    plate=crop_pooling_band(rectify(img,np.float32(cfg['corners']))); lab=lab_float(plate)
    dirty_m=roi_mask(plate.shape,tuple(cfg['dirty'])); clean_m=roi_mask(plate.shape,tuple(cfg['clean'])); clean_lab=robust_lab(lab,clean_m)
    excluded=clean_m.copy()
    for gx,gy,gw,gh in cfg['guides']:
        # The generated guide is a broad horizontal corridor, not a contact mask.
        # Keep a surrounding dirty-background margin for baseline estimation.
        pad=max(2,int(.09*gw))
        r=(gx+pad,gy,max(1,gw-2*pad),gh)
        excluded=cv2.bitwise_or(excluded,roi_mask(plate.shape,r))
    excluded[dirty_m>0]=0
    dirty_model=spatial_dirty_model(lab,excluded); frac=cleaning_fraction(lab,dirty_model,clean_lab)
    soil=originally_soiled_mask(dirty_model,clean_lab); valid=np.full(plate.shape[:2],255,np.uint8)
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
        accepted_masks=[]
        for i in range(exp['n_products']):
            key=(name,i); accepted=fit_mask_to_plate(exp['accepted'][key],plate.shape)
            # Clip every accepted footprint to its own product-guide ROI.
            # Automatic proposals/manual polygons can otherwise extend outside the
            # intended track and produce large cross-product heatmap artefacts.
            guide_mask=roi_mask(plate.shape,tuple(cfg['guides'][i]))
            accepted_clipped=cv2.bitwise_and(accepted,guide_mask)
            # Keep scoring restricted to originally-soiled pixels and the bottom exclusion.
            am=cv2.bitwise_and(cv2.bitwise_and(accepted_clipped,soil),valid)

            # Heatmap geometry is different from the scoring mask: show the whole accepted
            # contact footprint, clipped only to its product guide and to the straight
            # bottom-exclusion band. Intersecting with `soil` caused the irregular curved
            # and notched lower edges seen in the heatmap.
            heatmap_mask=cv2.bitwise_and(accepted_clipped,valid)
            accepted_masks.append(heatmap_mask)

            # Keep true contact/wetting area separate from the originally-soiled
            # pixels that are eligible for quantitative cleaning analysis.
            pname=f"Product {chr(65+i)}"
            row=metrics(pname,frac,am,contact_mask=accepted_clipped,
                        plate_area_px=plate.shape[0]*plate.shape[1])
            row['Replicate']=ri; rows.append(row)
            stem=pname.replace(' ','_')
            files[f'{prefix}/{stem}_contact_mask.png']=png_bytes(accepted_clipped)
            files[f'{prefix}/{stem}_analysis_mask.png']=png_bytes(am)
            files[f'{prefix}/{stem}_heatmap.png']=png_bytes(heatmap_plate_overlay(plate,frac,[heatmap_mask]))
        files[f'{prefix}/all_products_heatmap.png']=png_bytes(heatmap_plate_overlay(plate,frac,accepted_masks))
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
    st.info("The bottom 5% of the rectified plate is cropped away immediately after corner correction because product can pool against the rack/stand.")
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
        corner_widget=f"corners_{name}"
        # Fold the most recent component click into state before drawing the overlay.
        prev=st.session_state.get(corner_widget)
        prev_sig=_click_signature(prev) if isinstance(prev,dict) else None
        last_key=corner_state+"_last"
        if prev_sig and prev_sig != st.session_state.get(last_key) and len(corner_pts)<4:
            st.session_state[last_key]=prev_sig
            corner_pts.append((float(prev["x"]),float(prev["y"])))
        corner_vis=draw_points(shown,corner_pts,["TL","TR","BR","BL"])
        corners_disp=collect_click(pil_bgr(corner_vis),corner_widget,corner_state,max_points=4)
        b1,b2=st.columns([1,3])
        if b1.button("Undo corner",key=f"undo_corner_{name}",disabled=not corners_disp):
            corners_disp.pop(); st.rerun()
        b2.caption(f"Corner points: {len(corners_disp)} / 4")
        if len(corners_disp)!=4:
            st.warning(f"Place exactly 4 corner points. Currently: {len(corners_disp)}")
            continue
        corners=[(x/s,y/s) for x,y in corners_disp]
        plate=crop_pooling_band(rectify(img,np.float32(corners))); pshow,ps=display_fit(plate)

        labels=["Dirty control", "Clean control"]
        st.markdown("**B. Reference areas** — select two opposite corners of the dirty control and then of the clean control.")
        rect_state=f"controls_rects_{name}"
        rects=st.session_state.setdefault(rect_state,[])
        current=len(rects)
        if current<2:
            st.info(f"Now select **{labels[current]}** ({current+1}/2)")
            pair_state=f"control_pair_{name}_{current}"
            pair=st.session_state.setdefault(pair_state,[])
            pair_widget=f"control_click_{name}_{current}"
            previous=st.session_state.get(pair_widget)
            signature=_click_signature(previous) if isinstance(previous,dict) else None
            marker_key=pair_state+"_last"
            if signature and signature!=st.session_state.get(marker_key) and len(pair)<2:
                st.session_state[marker_key]=signature
                pair.append((float(previous['x']),float(previous['y'])))
            vis=draw_rects(pshow,rects,labels,scale=ps)
            vis=draw_points(vis,pair,["1","2"])
            pair=collect_click(pil_bgr(vis),pair_widget,pair_state,max_points=2)
            if len(pair)==2:
                rr=rect_from_two_points((pair[0][0]/ps,pair[0][1]/ps),(pair[1][0]/ps,pair[1][1]/ps))
                rects.append(rr)
                st.session_state.pop(pair_state,None)
                st.session_state.pop(marker_key,None)
                st.rerun()
        # Once the two controls are selected, draw them on the larger interactive
        # top-point image below instead of duplicating the entire plate here.
        r1,r2=st.columns([1,3])
        if r1.button("Undo last reference",key=f"undo_ctrl_{name}",disabled=not rects):
            rects.pop(); st.rerun()
        r2.caption(f"Reference areas: {len(rects)} / 2")

        if len(rects)==2:
            tip_state=f"tip_points_{name}"
            tips_display=st.session_state.setdefault(tip_state,[])

            # The rectified plate can be much narrower than the available viewport.
            # Enlarge it for the top-point selector; click coordinates remain in this
            # displayed image and are converted back to native coordinates when saved.
            tip_scale=650.0/plate.shape[1]
            if tip_scale>1.0:
                tip_shown=cv2.resize(plate,(650, max(1,int(round(plate.shape[0]*tip_scale)))),interpolation=cv2.INTER_CUBIC)
            else:
                tip_shown,tip_scale=display_fit(plate,maxw=650)
            tip_scale=tip_shown.shape[1]/plate.shape[1]

            # Migration for points already clicked under the old (smaller) UI. In
            # that version the display scale was ps, and points were stored in
            # display pixels. Convert once if this session survived redeployment.
            scale_key=f"tip_coordinate_scale_{name}"
            previous_scale=st.session_state.get(scale_key,ps)
            if tips_display and abs(previous_scale-tip_scale)>1e-8:
                tips_display[:]=[(float(px)*tip_scale/previous_scale,
                                 float(py)*tip_scale/previous_scale)
                                for px,py in tips_display]
            st.session_state[scale_key]=tip_scale

            if len(tips_display)<int(n):
                current_product=chr(65+len(tips_display))
                st.markdown(f"**C. Product top points** — click the rounded top of **Product {current_product}**. Just one click per product, left to right.")
            else:
                st.markdown("**C. Product top points** — all product tops are selected.")

            overlay=draw_rects(tip_shown,rects,labels,scale=tip_scale)
            tip_labels=[chr(65+i) for i in range(int(n))]
            overlay=draw_points(overlay,tips_display,tip_labels)

            if len(tips_display)<int(n):
                # A fresh component key at each point prevents an old click from
                # being replayed when the annotated image changes on rerun.
                epoch=st.session_state.setdefault(f"top_point_epoch_{name}",0)
                widget_key=f"top_click_{name}_{epoch}_{len(tips_display)}"
                click=streamlit_image_coordinates(
                    pil_bgr(overlay),key=widget_key,
                    width=tip_shown.shape[1],cursor="crosshair"
                )
                if isinstance(click,dict) and 'x' in click and 'y' in click:
                    px,py=float(click['x']),float(click['y'])
                    if 0<=px<tip_shown.shape[1] and 0<=py<tip_shown.shape[0]:
                        tips_display.append((px,py))
                        # Rendering a fresh component avoids the old image-disappearing
                        # problem caused by rerunning a component with the same key.
                        st.rerun()
            else:
                st.image(pil_bgr(overlay),width=tip_shown.shape[1])

            t1,t2=st.columns([1,3])
            if t1.button("Undo last top point",key=f"undo_tip_{name}",disabled=not tips_display):
                tips_display.pop()
                st.session_state[f"top_point_epoch_{name}"]=st.session_state.get(f"top_point_epoch_{name}",0)+1
                st.rerun()
            t2.caption(f"Product top points: {len(tips_display)} / {int(n)}")
            if len(tips_display)==int(n):
                tips=[(px/tip_scale,py/tip_scale) for px,py in tips_display]
                if not all(tips[i][0]<tips[i+1][0] for i in range(len(tips)-1)):
                    st.error("Product top points must be selected from left to right. Undo and correct the last point.")
                elif st.button(f"Save setup for replicate {ri}",key=f"save_{name}",type="primary"):
                    guides=lane_guides(tips,plate.shape,plate_bgr=plate)
                    st.session_state.exp['configured'][name]={
                        'corners':corners,'dirty':rects[0],'clean':rects[1],
                        'guides':guides,'tips':tips
                    }
                    for i in range(int(n)):st.session_state.exp['accepted'].pop((name,i),None)
                    st.session_state['footprints_confirmed']=False
                    st.rerun()

if len(st.session_state.exp['configured'])<len(names):
    st.info("Finish and save the setup for every replicate to continue.")
    st.stop()

st.markdown('<div class="step">STEP 3 · FOOTPRINT VERIFICATION</div>',unsafe_allow_html=True)
st.caption("The red outline is the automatic contact-footprint proposal. If it looks correct, leave it as-is. Only choose **Correct manually** when the proposal needs adjustment. Clicking **Continue to results** automatically accepts every unchanged proposal.")

proposals={}
for ri,name in enumerate(names,1):
    plate,dirty_model,frac,soil,valid=prepare_plate(st.session_state.exp['files'][name],st.session_state.exp['configured'][name])
    st.subheader(f"Replicate {ri}")
    cols=st.columns(2)
    for i,r in enumerate(st.session_state.exp['configured'][name]['guides']):
        pname=f"Product {chr(65+i)}"; key=(name,i); proposal=propose_footprint(frac,tuple(r),plate_bgr=plate,tip_xy=st.session_state.exp['configured'][name]['tips'][i]); proposals[key]=proposal
        with cols[i%2]:
            st.markdown(f"**{pname}**")
            if key in st.session_state.exp['accepted']:
                st.image(pil_bgr(mask_overlay(plate,st.session_state.exp['accepted'][key])),caption="Manually corrected footprint",use_container_width=True)
                if st.button("Change manual correction",key=f'change_{ri}_{i}',use_container_width=True):
                    st.session_state.exp['accepted'].pop(key,None)
                    st.session_state[f'manual_{ri}_{i}']=True
                    st.rerun()
            else:
                st.image(pil_bgr(mask_overlay(plate,proposal)),caption="Automatic proposal — used unless corrected manually",use_container_width=True)
                if st.button("Correct manually",key=f'corr_{ri}_{i}',use_container_width=True):
                    st.session_state[f'manual_{ri}_{i}']=True
                    st.rerun()
                if st.session_state.get(f'manual_{ri}_{i}',False):
                    st.caption("Click around the actual footprint. Use at least 3 points, then press **Use manual footprint**.")
                    pshow,ps=display_fit(plate,maxw=600)
                    poly_state=f"poly_pts_{ri}_{i}"
                    poly_pts=st.session_state.setdefault(poly_state,[])
                    poly_widget=f'poly_{ri}_{i}'
                    prev=st.session_state.get(poly_widget)
                    prev_sig=_click_signature(prev) if isinstance(prev,dict) else None
                    last_key=poly_state+"_last"
                    if prev_sig and prev_sig != st.session_state.get(last_key):
                        st.session_state[last_key]=prev_sig
                        poly_pts.append((float(prev["x"]),float(prev["y"])))
                    pvis=draw_points(pshow,poly_pts)
                    if len(poly_pts)>=2:
                        cv2.polylines(pvis,[np.array(poly_pts,np.int32)],False,(0,0,255),3)
                    poly_pts=collect_click(pil_bgr(pvis),poly_widget,poly_state)
                    pc1,pc2=st.columns(2)
                    if pc1.button("Undo point",key=f'undopoly_{ri}_{i}',disabled=not poly_pts):
                        poly_pts.pop(); st.rerun()
                    if pc2.button("Clear points",key=f'clearpoly_{ri}_{i}',disabled=not poly_pts):
                        st.session_state[poly_state]=[]; st.session_state.pop(poly_state+"_last",None); st.rerun()
                    mm=polygon_mask(plate.shape,[(x/ps,y/ps) for x,y in poly_pts])
                    if len(poly_pts)>=3 and st.button("Use manual footprint",key=f'usepoly_{ri}_{i}',type="primary"):
                        st.session_state.exp['accepted'][key]=mm
                        st.session_state[f'manual_{ri}_{i}']=False
                        st.rerun()

if not st.session_state.get('footprints_confirmed',False):
    st.info("Automatic proposals do not need individual approval. Correct only the footprints that need it, then continue.")
    if st.button("Continue to results",type="primary",use_container_width=True):
        for key,proposal in proposals.items():
            if key not in st.session_state.exp['accepted']:
                st.session_state.exp['accepted'][key]=proposal
        st.session_state['footprints_confirmed']=True
        st.rerun()
    st.stop()

st.markdown('<div class="step">STEP 4 · RESULTS</div>',unsafe_allow_html=True)
df,xlsx,zipbytes=build_outputs()
summary=df.groupby('Product',sort=False).agg(
    n=('Replicate','count'),
    mean_cleaning=('Mean cleaning depth (%)','mean'),
    sd_cleaning=('Mean cleaning depth (%)','std'),
    contact=('Contact area (% plate)','mean'),
    sd_contact=('Contact area (% plate)','std'),
    total_removal=('Integrated optical removal (% plate)','mean'),
    sd_total_removal=('Integrated optical removal (% plate)','std'),
    footprint=('Footprint area (px)','mean'),
    coverage50=('Footprint >=50% cleaned (%)','mean'),
    coverage75=('Footprint >=75% cleaned (%)','mean'),
).reset_index()
summary['Rank (mean cleaning)']=summary['mean_cleaning'].rank(ascending=False,method='min').astype('Int64')
summary['Rank (total removal)']=summary['total_removal'].rank(ascending=False,method='min').astype('Int64')

st.subheader("Cleaning performance across replicate plates")
display_summary=summary.rename(columns={
    'mean_cleaning':'Mean cleaning (%)','sd_cleaning':'Cleaning SD',
    'contact':'Contact area (% plate)','sd_contact':'Contact area SD',
    'total_removal':'Total optical removal (% plate)','sd_total_removal':'Total removal SD',
    'footprint':'Mean contact area (px)',
    'coverage50':'Area ≥50% cleaned (%)','coverage75':'Area ≥75% cleaned (%)',
})
st.dataframe(
    display_summary,hide_index=True,use_container_width=True,
    column_config={
        **{label:st.column_config.NumberColumn(format='%.2f') for label in [
            'Mean cleaning (%)','Cleaning SD','Contact area (% plate)','Contact area SD',
            'Total optical removal (% plate)','Total removal SD',
            'Area ≥50% cleaned (%)','Area ≥75% cleaned (%)']},
        'Mean contact area (px)':st.column_config.NumberColumn(format='%.0f'),
    },
)
st.caption("Mean cleaning describes cleaning intensity within the originally soiled part of the footprint. "
           "Contact area describes spreading. Total optical removal combines cleaning intensity "
           "and the analyzed area, expressed as equivalent clean area relative to the rectified plate.")

fig,ax=plt.subplots(figsize=(9,4.8))
x=np.arange(len(summary))
means=summary['mean_cleaning'].to_numpy(dtype=float)
sds=summary['sd_cleaning'].fillna(0).to_numpy(dtype=float)
labels=summary['Product'].astype(str).tolist()
ax.bar(x,means,yerr=sds,capsize=6)
ax.set_ylabel('Cleaning (%)')
ax.set_ylim(0,100)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('Mean cleaning intensity ± SD')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',alpha=.2)
ax.set_axisbelow(True)
fig.tight_layout()
st.pyplot(fig,use_container_width=True)
plt.close(fig)

# Unlike mean cleaning, this score rewards spreading over a larger soiled area.
# Expressing the integral as a percentage of plate area makes it comparable when
# replicate photographs have different pixel dimensions.
st.subheader('Total optical removal — cleaning intensity × area')
fig_total,ax_total=plt.subplots(figsize=(9,4.8))
integrated=summary['total_removal'].to_numpy(dtype=float)
integrated_sd=summary['sd_total_removal'].fillna(0).to_numpy(dtype=float)
ax_total.bar(x,integrated,yerr=integrated_sd,capsize=6)
ax_total.set_ylabel('Equivalent cleaned area (% of plate)')
ax_total.set_ylim(bottom=0)
ax_total.set_xticks(x)
ax_total.set_xticklabels(labels)
ax_total.set_title('Integrated optical removal ± SD')
ax_total.spines[['top','right']].set_visible(False)
ax_total.grid(axis='y',alpha=.2)
ax_total.set_axisbelow(True)
fig_total.tight_layout()
st.pyplot(fig_total,use_container_width=True)
plt.close(fig_total)
st.caption('Each pixel contributes its estimated cleaning fraction (0–1). The sum is ' 
           'divided by the total cropped plate pixels. For example, a footprint covering ' 
           '20% of the plate at 50% mean cleaning of originally soiled pixels contributes ' 
           'approximately 10% equivalent clean area when the whole footprint was originally soiled. ' 
           'This is an optical proxy, **not** a measurement of soil mass.')

st.subheader('Product spreading')
fig_contact,ax_contact=plt.subplots(figsize=(9,4.2))
ax_contact.bar(x,summary['contact'].to_numpy(dtype=float),
               yerr=summary['sd_contact'].fillna(0).to_numpy(dtype=float),capsize=6)
ax_contact.set_ylabel('Contacted area (% of plate)')
ax_contact.set_ylim(bottom=0)
ax_contact.set_xticks(x)
ax_contact.set_xticklabels(labels)
ax_contact.set_title('Contact area ± SD')
ax_contact.spines[['top','right']].set_visible(False)
ax_contact.grid(axis='y',alpha=.2)
ax_contact.set_axisbelow(True)
fig_contact.tight_layout()
st.pyplot(fig_contact,use_container_width=True)
plt.close(fig_contact)

coverage=summary.set_index('Product')[['coverage50','coverage75']].rename(columns={'coverage50':'≥50% cleaned','coverage75':'≥75% cleaned'})
st.markdown('**Cleaning coverage within the originally soiled footprint**')
st.bar_chart(coverage,y_label='Footprint meeting threshold (%)',height=360)


st.subheader("Cleaning heatmaps")
st.caption("Each replicate shows the cleaning performance of all products on the same plate.")

# Colour scale matching OpenCV COLORMAP_TURBO used in the heatmaps.
scale_vals = np.linspace(0, 255, 512, dtype=np.uint8).reshape(1, -1)
scale_bgr = cv2.applyColorMap(scale_vals, cv2.COLORMAP_TURBO)
scale_rgb = cv2.cvtColor(scale_bgr, cv2.COLOR_BGR2RGB)
fig_cb, ax_cb = plt.subplots(figsize=(9, 0.75))
ax_cb.imshow(scale_rgb, aspect='auto', extent=[0, 100, 0, 1])
ax_cb.set_yticks([])
ax_cb.set_xlim(0, 100)
ax_cb.set_xticks([0, 25, 50, 75, 100])
ax_cb.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
ax_cb.set_xlabel('Cleaning')
for spine in ax_cb.spines.values():
    spine.set_visible(False)
fig_cb.tight_layout(pad=0.4)
st.pyplot(fig_cb, use_container_width=True)
plt.close(fig_cb)
for ri,name in enumerate(names,1):
    plate,dirty_model,frac,soil,valid=prepare_plate(st.session_state.exp['files'][name],st.session_state.exp['configured'][name])
    masks=[]
    cfg=st.session_state.exp['configured'][name]
    for i in range(st.session_state.exp['n_products']):
        accepted=fit_mask_to_plate(st.session_state.exp['accepted'][(name,i)],plate.shape)
        guide_mask=roi_mask(plate.shape,tuple(cfg['guides'][i]))
        accepted_clipped=cv2.bitwise_and(accepted,guide_mask)
        # The pooling band was physically cropped before any ROIs or footprints were created.
        # Therefore the heatmap needs no irregular bottom mask at all.
        heatmap_mask=accepted_clipped
        masks.append(heatmap_mask)
    combined=heatmap_plate_overlay(plate,frac,masks)
    st.markdown(f"**Replicate {ri}**")
    st.image(pil_bgr(combined),use_container_width=True)

c1,c2=st.columns(2)
c1.download_button("Download Excel report",xlsx,file_name='Cleaning_Analyzer_results.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
c2.download_button("Download complete analysis ZIP",zipbytes,file_name='Cleaning_Analyzer_analysis.zip',mime='application/zip',use_container_width=True)
st.caption("Across-plate values are arithmetic means with each replicate weighted equally. "
           "SD is the sample standard deviation between replicate plates; with n=1, SD is not reported. "
           "Percent-of-plate measurements assume the photographs show equivalent physical plate regions "
           "after perspective correction and the fixed bottom 5% crop. Full contact area includes pixels "
           "that were not originally soiled; total removal only sums pixels assessed as originally soiled.")
