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

# Changing this identifier causes the Streamlit app to refresh previously
# calculated automatic footprints without removing plate/ROI selections.
FOOTPRINT_ALGORITHM_VERSION = "photo_edges_20261008_v1"


def _smooth_trace(values, sigma):
    """One-dimensional Gaussian smoothing without a scipy dependency."""
    vector = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if len(vector) < 2:
        return vector.ravel()
    return cv2.GaussianBlur(vector, (1, 0), sigmaX=0, sigmaY=max(0.5, sigma)).ravel()


def _edge_guided_tongue(plate_bgr, guide):
    """Trace both contact boundaries using photographic edge evidence.

    Each edge follows the bright perimeter ridge, not the cleaned interior.
    An approximately tongue-shaped contour is used as a soft constraint. This
    continues the track through weakly visible/dirty regions and prevents the
    old connected-component algorithm from selecting only its upper/lower part.
    """
    height, width = plate_bgr.shape[:2]
    x, y, gw, gh = map(float, guide)
    if gw < 12 or gh < 12:
        return None
    cx = x + gw / 2.0

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.4)
    # A bright, narrow outline is visible around even very poorly cleaned
    # product tracks. The broad background is subtracted to expose the ridge.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(7.0, 0.035 * gw))
    ridge = np.maximum(gray - background, 0)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    edge_score = cv2.GaussianBlur(ridge, (3, 3), 0.8)
    edge_score += 0.20 * cv2.GaussianBlur(gx, (3, 3), 0.8)
    edge_score = np.clip(edge_score, 0, 36)

    # Locate the rounded cap near the top of the marked guide. Horizontal
    # ridge evidence makes this work for dark interiors as well as white ones.
    xa = max(0, int(cx - .20 * gw))
    xb = min(width, int(cx + .20 * gw))
    ya = max(5, int(y - .24 * gh))
    yb = min(int(.55 * height), int(y + .19 * gh))
    if xb <= xa + 3 or yb <= ya + 3:
        return None
    cap_scores = _smooth_trace(np.mean(ridge[ya:yb, xa:xb], axis=1), 4)
    candidates_y = np.arange(ya, yb)
    prior = np.exp(-.5 * ((candidates_y - y) / max(25., .17 * gh))**2)
    tip = int(candidates_y[np.argmax(cap_scores * (.50 + .50 * prior))])
    tip = max(0, min(tip, height - 8))

    # The user guide defines the horizontal corridor. Restrict each traced
    # side so strong dirt streaks or neighbouring track edges cannot take over.
    left_limit = max(0, int(x - 0.045 * gw))
    right_limit = min(width, int(x + 1.045 * gw))
    if right_limit <= left_limit + 10:
        return None
    # A rounded cap widens into the normal track width over this distance.
    cap_height = max(14, int(np.clip(.35 * gw, 25, 125)))
    shoulder_height = max(cap_height, int(.80 * gw))

    def predicted_width(row):
        t = np.clip((row - tip + 1) / shoulder_height, 0, 1)
        return max(.08, np.sqrt(max(0., 1. - (1. - t)**2))) * gw

    paths = []
    for side in (-1, 1):
        lo = left_limit if side == -1 else max(left_limit, int(cx + .035 * gw))
        hi = min(right_limit, int(cx - .035 * gw)) if side == -1 else right_limit
        cols = np.arange(lo, hi, dtype=np.float32)
        if len(cols) < 8:
            return None
        energies = []
        for row in range(tip, height):
            target = cx + side * predicted_width(row) / 2.
            deviations = np.abs(cols - target) / gw
            # Ridge is important, but not important enough to permit sudden
            # jumps from the correct outline to a different product/droplet.
            evidence = edge_score[row, lo:hi].copy()
            evidence -= 18.0 * (deviations / .25)**2
            energies.append(evidence)
        energy = np.asarray(energies, dtype=np.float32)
        row_count, nstates = energy.shape
        value = energy[0].copy()
        predecessor = np.zeros((row_count, nstates), dtype=np.int32)
        state_ix = np.arange(nstates)
        for ri in range(1, row_count):
            best = np.full(nstates, -1.e9, dtype=np.float32)
            back = np.zeros(nstates, dtype=np.int32)
            for shift in range(-3, 4):
                preceding = np.clip(state_ix + shift, 0, nstates - 1)
                objective = value[preceding] - (1.2 * abs(shift) + .4 * shift**2)
                improved = objective > best
                best[improved] = objective[improved]
                back[improved] = preceding[improved]
            value = best + energy[ri]
            predecessor[ri] = back
        end_index = int(np.argmax(value))
        path = np.empty(row_count, dtype=np.float32)
        for ri in range(row_count - 1, -1, -1):
            path[ri] = cols[end_index]
            end_index = predecessor[ri, end_index]
        paths.append(_smooth_trace(path, sigma=11))

    left, right = paths
    if np.median(right - left) < 4:
        return None
    trace_widths = np.maximum(right - left, 1.)
    trace_centers = _smooth_trace((right + left) / 2., 12)
    trace_widths = np.maximum(trace_widths, .88 * _smooth_trace(trace_widths, 23))
    trace_widths = _smooth_trace(trace_widths, 12)
    body = trace_widths[min(len(trace_widths)-1, cap_height):]
    if len(body):
        reference = float(np.median(body))
        # Small bottom taper is permitted; gross inward notches are not.
        trace_widths[cap_height:] = np.maximum(trace_widths[cap_height:], .80 * reference)
        trace_widths = _smooth_trace(trace_widths, 8)

    join = min(len(trace_widths)-1, cap_height)
    left_join = max(0, join - 8)
    right_join = min(len(trace_widths), join + 8)
    joined_width = float(np.median(trace_widths[left_join:right_join]))
    joined_center = float(np.median(trace_centers[left_join:right_join]))
    out = np.zeros((height, width), dtype=np.uint8)
    for relative_y, row in enumerate(range(tip, height)):
        half_width = trace_widths[relative_y] / 2.
        center = trace_centers[relative_y]
        if relative_y < cap_height:
            t = (relative_y + .5) / cap_height
            # Rounded, broad cap (superellipse), not a triangular point.
            cap_fraction = (1 - (1 - t)**3)**(1 / 3)
            half_width = (joined_width / 2.) * cap_fraction
            center = joined_center
        xmin = max(left_limit, int(round(center - half_width)))
        xmax = min(right_limit, int(round(center + half_width)) + 1)
        if xmax > xmin:
            out[row, xmin:xmax] = 255
    return out


def propose_footprint(clean_frac, guide_roi, plate_bgr=None):
    """Suggest the complete product-contact tongue, including poorly cleaned soil.

    The application should pass the *cropped rectified plate* as plate_bgr.
    For backward compatibility the clean_fraction map can be used as a fallback,
    but photographic boundary evidence is much more reliable.
    """
    if plate_bgr is None:
        # Older callers may not provide a photograph. Retain usable proposals.
        v = np.clip(np.nan_to_num(clean_frac, nan=0), 0, 1)
        plate_bgr = cv2.cvtColor(np.uint8(v * 255), cv2.COLOR_GRAY2BGR)
    if plate_bgr.shape[:2] != clean_frac.shape[:2]:
        raise ValueError('Photo and cleaning fraction dimensions must match.')
    # Lower-resolution edge tracing keeps Streamlit reruns reasonably fast.
    H, W = clean_frac.shape[:2]
    scale = min(1., 900. / W)
    if scale < 1:
        smaller = cv2.resize(plate_bgr, (round(W*scale), round(H*scale)),
                             interpolation=cv2.INTER_AREA)
        smaller_guide = tuple(float(v)*scale for v in guide_roi)
    else:
        smaller = plate_bgr
        smaller_guide = tuple(float(v) for v in guide_roi)
    result = _edge_guided_tongue(smaller, smaller_guide)
    if result is None or not np.any(result):
        result = roi_mask(smaller.shape, tuple(map(int, smaller_guide)))
    if scale < 1:
        result = cv2.resize(result, (W, H), interpolation=cv2.INTER_NEAREST)
    return result


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

