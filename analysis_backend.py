import cv2
import numpy as np
import pandas as pd
from pathlib import Path

BOTTOM_MARGIN_FRAC = 0.05

# ------------------------- geometry -------------------------

def order4(pts):
    s=pts.sum(axis=1); d=np.diff(pts,axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]],np.float32)

def rectify(img, pts):
    tl,tr,br,bl=order4(pts)
    W=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl)))
    H=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
    dst=np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    M=cv2.getPerspectiveTransform(np.float32([tl,tr,br,bl]),dst)
    return cv2.warpPerspective(img,M,(W,H))

# ------------------------- color / baseline -------------------------

def lab_float(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2LAB).astype(np.float32)

def roi_mask(shape, r):
    x,y,w,h=r
    m=np.zeros(shape[:2],np.uint8); m[y:y+h,x:x+w]=255
    return m

def robust_lab(img_lab, mask):
    p=img_lab[mask>0]
    return np.median(p,axis=0)

def spatial_dirty_model(lab, excluded, sigma_frac=.10):
    """
    Estimate the local dirty appearance from untouched pixels.
    Normalized convolution prevents excluded/cleaned pixels from bleeding in.
    """
    h,w=lab.shape[:2]
    valid=(excluded==0).astype(np.float32)
    sigma=max(15.0, sigma_frac*min(h,w))
    den=cv2.GaussianBlur(valid,(0,0),sigmaX=sigma,sigmaY=sigma)
    out=np.zeros_like(lab,np.float32)
    for c in range(3):
        num=cv2.GaussianBlur(lab[:,:,c]*valid,(0,0),sigmaX=sigma,sigmaY=sigma)
        out[:,:,c]=num/np.maximum(den,1e-4)
    # Fill poorly supported pixels with global dirty median.
    g=np.median(lab[valid>0.5],axis=0)
    bad=den<0.05
    out[bad]=g
    return out

def cleaning_fraction(lab, dirty_model, clean_lab):
    """
    Project each pixel from its local dirty endpoint toward the clean endpoint
    in 3D CIELAB. 0 = dirty, 1 = clean. This uses color direction, not whiteness.
    """
    v=clean_lab.reshape(1,1,3)-dirty_model
    q=lab-dirty_model
    den=np.sum(v*v,axis=2)
    f=np.sum(q*v,axis=2)/np.maximum(den,1e-6)
    return np.clip(f,0,1)

def originally_soiled_mask(dirty_model, clean_lab, clean_distance_fraction=0.45):
    """
    Estimate substrate that was originally covered by soil.
    Uses the estimated pre-cleaning dirty endpoint, not the observed cleaned colour.
    """
    clean=clean_lab.reshape(1,1,3)
    d=np.linalg.norm(dirty_model-clean,axis=2)
    positive=d[d>1e-6]
    if positive.size==0:
        return np.ones(d.shape,np.uint8)*255
    dirty_sep=float(np.percentile(positive,75))
    threshold=clean_distance_fraction*dirty_sep
    m=(d>=threshold).astype(np.uint8)*255
    k=max(3,int(0.006*min(d.shape))); k += 1-k%2
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,kernel,iterations=1)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,kernel,iterations=1)
    return m

# ------------------------- footprint proposal -------------------------

def regularize_tongue(mask, guide):
    """Smooth a detected track into a rounded tongue without large side notches.

    The rounded tip is estimated from the detected footprint, while the lower
    portion permits mild tapering. All geometry stays in the guide corridor.
    """
    H, W = mask.shape
    x, y, gw, gh = map(int, guide)
    yy, xx = np.where(mask > 0)
    if not len(xx):
        return mask

    top = int(yy.min())
    # Estimate the footprint's stable sides from row-wise envelopes.
    left = np.full(H, np.nan)
    right = np.full(H, np.nan)
    for row in range(top, H):
        cols = np.flatnonzero(mask[row] > 0)
        if len(cols) >= 3:
            left[row] = np.percentile(cols, 3)
            right[row] = np.percentile(cols, 97)
    valid = np.flatnonzero(np.isfinite(left))
    if len(valid) < 5:
        return mask
    rows = np.arange(top, H)
    left_vals = np.interp(rows, valid, left[valid])
    right_vals = np.interp(rows, valid, right[valid])
    sigma = max(3.0, min(14.0, gh * 0.025))
    def smooth(values):
        return cv2.GaussianBlur(values.astype(np.float32)[:, None],
                                (1, 0), sigmaX=0, sigmaY=sigma).ravel()
    left_vals, right_vals = smooth(left_vals), smooth(right_vals)
    centers = (left_vals + right_vals) / 2
    widths = np.maximum(1, right_vals - left_vals)

    # Use the first substantial band below the tip for a reliable body width.
    body_start = min(len(rows)-1, max(1, int(0.16 * gh)))
    body_end = min(len(rows), max(body_start+1, int(0.50 * gh)))
    body_width = float(np.percentile(widths[body_start:body_end], 65))
    body_width = max(6.0, body_width)

    # Suppress inward notches: each row below the rounded cap retains at
    # least 82% of the established body width; small bottom taper is allowed.
    widths = np.maximum(widths, 0.82 * body_width)
    # Round the cap with a half-ellipse: broad shoulders, no pointed triangle.
    cap_height = max(6, min(int(0.18 * gh), int(0.38 * body_width)))
    cap_height = min(cap_height, len(rows))
    cap_width = float(np.median(widths[min(cap_height, len(rows)-1):
                                       min(len(rows), cap_height + max(3, int(0.1*gh)))]))
    cap_width = max(cap_width, 0.82 * body_width)
    for i in range(cap_height):
        t = (i + 0.5) / cap_height
        rounded = cap_width * np.sqrt(max(0.0, 1.0 - (1.0-t)**2))
        widths[i] = rounded
    widths = smooth(widths)

    # Prevent strong lateral jumps without shifting the entire footprint.
    centers = smooth(centers)
    corridor_l = max(0, int(x - 0.10*gw))
    corridor_r = min(W, int(x + 1.10*gw))
    out = np.zeros_like(mask)
    for i, row in enumerate(rows):
        half = widths[i] / 2
        l = max(corridor_l, int(round(centers[i] - half)))
        r = min(corridor_r, int(round(centers[i] + half)) + 1)
        if r > l:
            out[row, l:r] = 255
    return out

def propose_footprint(clean_frac, guide_roi):
    """Estimate a continuous rounded track, without connected-component selection.

    The guide fixes product identity and approximate horizontal position. Row-wise
    boundaries are estimated across the whole height, then regularized so a weak
    contrast region cannot delete the top or lower half of a track.
    """
    H, W = clean_frac.shape
    x, y, w, h = map(int, guide_roi)
    if w < 5 or h < 8:
        return roi_mask(clean_frac.shape, guide_roi)
    cx = x + w / 2.0
    # A guide is a rough bounding rectangle, not a footprint segmentation.
    lo = max(0, int(x - .12*w)); hi = min(W, int(x + 1.12*w))
    top = max(0, int(y - .25*h))
    bottom = min(H, max(y+h, int(H-.015*H)))
    if bottom <= top + 8 or hi <= lo + 5:
        return roi_mask(clean_frac.shape, guide_roi)

    # A continuous side envelope is more reliable than thresholded components.
    # In each row find the broad high-cleaning band near the guide centre.
    sm = cv2.GaussianBlur(clean_frac.astype(np.float32), (0, 0),
                          sigmaX=max(2, .025*w), sigmaY=max(2, .012*h))
    centers=[]; widths=[]; strengths=[]
    search_x=np.arange(lo,hi)
    for row in range(top,bottom):
        v=sm[row,lo:hi]
        # Limit lateral drift; do not allow a neighbour to win the search.
        proximity=np.exp(-.5*((search_x-cx)/max(.38*w,1))**2)
        weight=np.maximum(v-.025,0)*proximity
        if weight.sum()>1e-6:
            center=float((search_x*weight).sum()/weight.sum())
        else:
            center=cx
        center=np.clip(center, cx-.13*w, cx+.13*w)
        # Edge estimate uses a moderate threshold; noisy/missing rows fall back
        # to the guide width rather than collapsing to a disconnected fragment.
        threshold=max(.055, float(np.percentile(v,75))*.32)
        candidate=np.where((v>=threshold)&(np.abs(search_x-center)<.60*w))[0]
        if len(candidate)>4:
            left=float(search_x[candidate[0]]); right=float(search_x[candidate[-1]])
            width=right-left
        else:
            width=.88*w
        centers.append(center); widths.append(width)
        strengths.append(float(np.mean(v)))
    centers=np.asarray(centers,np.float32)
    widths=np.asarray(widths,np.float32)
    # Establish stable body width from the guide and multiple rows, not a
    # single selected component or the potentially noisy cap.
    body_start=int(.25*len(widths)); body_end=max(body_start+1,int(.75*len(widths)))
    typical=float(np.median(widths[body_start:body_end]))
    typical=float(np.clip(typical,.72*w,1.08*w))
    widths=np.clip(widths,.83*typical,1.13*typical)
    sigma=max(3.,min(16.,.035*h))
    def smooth(v):
        return cv2.GaussianBlur(v[:,None],(1,0),sigmaX=0,sigmaY=sigma).ravel()
    centers=smooth(centers)
    widths=smooth(widths)
    # Use the guide's upper edge as a conservative tip anchor, with limited
    # adjustment toward an above-guide high-contrast band.
    tip=max(top, y-int(.07*h))
    cap=max(8,int(min(.22*h,.52*typical)))
    out=np.zeros((H,W),np.uint8)
    for row in range(tip,bottom):
        i=row-top
        if i<0 or i>=len(widths): continue
        width=float(widths[i]); center=float(centers[i])
        if row<tip+cap:
            t=(row-tip+.5)/cap
            width=typical*np.sqrt(max(0.,1.-(1.-t)**2))
        left=max(lo,int(round(center-width/2)))
        right=min(hi,int(round(center+width/2))+1)
        if right>left: out[row,left:right]=255
    return out


# ------------------------- analysis -------------------------

def heatmap_overlay(img, frac, mask, name):
    heat=(np.clip(frac*255,0,255)).astype(np.uint8)
    heat=cv2.applyColorMap(heat,cv2.COLORMAP_TURBO)
    out=img.copy()
    alpha=.55
    ix=mask>0
    out[ix]=(alpha*heat[ix]+(1-alpha)*out[ix]).astype(np.uint8)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out,cnts,-1,(0,0,255),3)
    cv2.putText(out,"Blue = low cleaning   Red = high cleaning",
                (20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA)
    return out

def bottom_exclusion_mask(shape, frac=BOTTOM_MARGIN_FRAC):
    """Mask of the valid plate area after removing a fixed bottom margin."""
    h, w = shape[:2]
    m = np.ones((h, w), np.uint8) * 255
    cut = int(round(h * (1.0 - float(frac))))
    cut = max(0, min(h, cut))
    m[cut:, :] = 0
    return m

def metrics(name, frac, mask):
    vals=frac[mask>0]
    area=int(np.count_nonzero(mask))
    return {
        "Product":name,
        "Footprint area (px)":area,
        "Mean cleaning depth (%)":100*float(np.mean(vals)),
        "Median cleaning depth (%)":100*float(np.median(vals)),
        "25th percentile cleaning (%)":100*float(np.percentile(vals,25)),
        "75th percentile cleaning (%)":100*float(np.percentile(vals,75)),
        "Footprint >=50% cleaned (%)":100*float(np.mean(vals>=.50)),
        "Footprint >=75% cleaned (%)":100*float(np.mean(vals>=.75)),
        "Integrated optical removal (equiv clean px)":float(np.sum(vals)),
    }


def save_report_workbook(df, outpath):
    """Create report workbook with replicate-level results and across-replicate mean/SD."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.chart import BarChart, Reference
    from openpyxl.chart.label import DataLabelList
    from openpyxl.formatting.rule import DataBarRule
    from openpyxl.utils import get_column_letter

    metric_map = {
        "Cleaning performance (%)": "Mean cleaning depth (%)",
        "Area >=50% cleaned (%)": "Footprint >=50% cleaned (%)",
        "Area >=75% cleaned (%)": "Footprint >=75% cleaned (%)",
        "Footprint area (px)": "Footprint area (px)",
    }
    g=df.groupby("Product", sort=False)
    rows=[]
    for product, sub in g:
        row={"Product":product,"n":len(sub)}
        for label,col in metric_map.items():
            row[label]=sub[col].mean()
            row[label+" SD"]=sub[col].std(ddof=1) if len(sub)>1 else np.nan
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary["Rank"]=summary["Cleaning performance (%)"].rank(ascending=False,method="min").astype(int)

    wb=Workbook(); ws=wb.active; ws.title="Summary"
    rep=wb.create_sheet("Replicate Results"); detail=wb.create_sheet("Detailed Results")
    navy="17365D"; blue="4472C4"; pale="D9EAF7"; white="FFFFFF"; thin=Side(style="thin",color="D9D9D9")

    ws["A1"]="Cleaning Performance Summary"; ws["A1"].font=Font(bold=True,size=18,color=navy)
    ws.merge_cells("A1:K1")
    ws["A2"]="Mean ± SD across independent plate/image replicates. Each replicate is weighted equally."; ws["A2"].font=Font(size=10,color="666666")
    ws.merge_cells("A2:K2")
    cols=["Product","n","Cleaning performance (%)","Cleaning performance (%) SD","Area >=50% cleaned (%)","Area >=50% cleaned (%) SD","Area >=75% cleaned (%)","Area >=75% cleaned (%) SD","Footprint area (px)","Footprint area (px) SD","Rank"]
    ws["A4"]="Comparative results"; ws["A4"].font=Font(bold=True,color=white); ws["A4"].fill=PatternFill("solid",fgColor=navy); ws.merge_cells("A4:K4")
    for c,col in enumerate(cols,1):
        cell=ws.cell(5,c,col); cell.font=Font(bold=True,color=white); cell.fill=PatternFill("solid",fgColor=blue); cell.alignment=Alignment(horizontal="center",vertical="center",wrap_text=True); cell.border=Border(left=thin,right=thin,top=thin,bottom=thin)
    for r,row in enumerate(summary[cols].itertuples(index=False,name=None),6):
        for c,val in enumerate(row,1):
            if pd.isna(val): val=None
            cell=ws.cell(r,c,val.item() if hasattr(val,"item") else val); cell.border=Border(left=thin,right=thin,top=thin,bottom=thin); cell.alignment=Alignment(horizontal="left" if c==1 else "center")
            if c in (3,4,5,6,7,8): cell.number_format="0.0"
            elif c in (2,9,10,11): cell.number_format="0"
    last=5+len(summary)
    for col in ("C","E","G"):
        ws.conditional_formatting.add(f"{col}6:{col}{last}",DataBarRule(start_type="num",start_value=0,end_type="num",end_value=100,color=pale,showValue=True))

    # Clean report-style charts. SD is shown numerically in the adjacent summary columns.
    ch=BarChart(); ch.type="col"; ch.style=10; ch.title="Mean cleaning performance"; ch.y_axis.title="Cleaning (%)"; ch.y_axis.scaling.min=0; ch.y_axis.scaling.max=100; ch.height=8; ch.width=14; ch.add_data(Reference(ws,min_col=3,min_row=5,max_row=last),titles_from_data=True); ch.set_categories(Reference(ws,min_col=1,min_row=6,max_row=last)); ch.legend=None; ch.dLbls=DataLabelList(); ch.dLbls.showVal=True; ws.add_chart(ch,"M2")
    ch2=BarChart(); ch2.type="col"; ch2.style=10; ch2.title="Cleaning coverage"; ch2.y_axis.title="Footprint meeting threshold (%)"; ch2.y_axis.scaling.min=0; ch2.y_axis.scaling.max=100; ch2.height=8; ch2.width=14; ch2.add_data(Reference(ws,min_col=5,max_col=7,min_row=5,max_row=last),titles_from_data=True,from_rows=False); ch2.series=[ch2.series[0],ch2.series[2]] if len(ch2.series)>=3 else ch2.series; ch2.set_categories(Reference(ws,min_col=1,min_row=6,max_row=last)); ch2.legend.position="b"; ws.add_chart(ch2,"U2")
    note=ws.cell(last+3,1); ws.merge_cells(start_row=last+3,start_column=1,end_row=last+6,end_column=11); note.value="Interpretation: values are arithmetic means across replicate plates/images; SD is the sample standard deviation between replicates (n=1: SD not reported). Replicates are weighted equally rather than by footprint pixel count. Footprint area is a diagnostic of contact/wetting and should not by itself be interpreted as cleaning efficacy."; note.font=Font(size=9,color="666666"); note.alignment=Alignment(wrap_text=True,vertical="top")
    for c,w in {"A":18,"B":8,"C":22,"D":18,"E":22,"F":18,"G":22,"H":18,"I":20,"J":18,"K":9}.items(): ws.column_dimensions[c].width=w
    ws.row_dimensions[5].height=42; ws.sheet_view.showGridLines=False

    # Replicate-level sheet: one row per product per plate.
    repdf=df.copy()
    for c,col in enumerate(repdf.columns,1):
        cell=rep.cell(1,c,col); cell.font=Font(bold=True,color=white); cell.fill=PatternFill("solid",fgColor=navy); cell.alignment=Alignment(horizontal="center",wrap_text=True)
    for r,vals in enumerate(repdf.itertuples(index=False,name=None),2):
        for c,val in enumerate(vals,1): rep.cell(r,c,val.item() if hasattr(val,"item") else val).number_format="0.00" if c>2 else "General"
    rep.freeze_panes="C2"; rep.auto_filter.ref=rep.dimensions; rep.sheet_view.showGridLines=False

    # Detailed results currently equal replicate results, retained for compatibility/report traceability.
    for c,col in enumerate(df.columns,1):
        cell=detail.cell(1,c,col); cell.font=Font(bold=True,color=white); cell.fill=PatternFill("solid",fgColor=navy); cell.alignment=Alignment(horizontal="center",wrap_text=True)
    for r,vals in enumerate(df.itertuples(index=False,name=None),2):
        for c,val in enumerate(vals,1): detail.cell(r,c,val.item() if hasattr(val,"item") else val).number_format="0.00" if c>2 else "General"
    for sh in (rep,detail):
        for c in range(1,len(df.columns)+1): sh.column_dimensions[get_column_letter(c)].width=18 if c<=2 else 24
        sh.freeze_panes="C2"; sh.auto_filter.ref=sh.dimensions; sh.sheet_view.showGridLines=False
    wb.save(outpath)

