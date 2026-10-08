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

# Used by Streamlit to invalidate old automatic masks after a detector update.
FOOTPRINT_ALGORITHM_VERSION = "tip_anchored_edge_refined_20261008_v4_bright_sides"

import cv2
import numpy as np

def _sm(v,sigma):
    return cv2.GaussianBlur(np.asarray(v,np.float32).reshape(-1,1),(1,0),sigmaX=0,sigmaY=max(.5,sigma)).ravel()

def lane_guides(tips, shape, plate_bgr=None):
    H,W=shape[:2]
    tips=[(float(px),float(py)) for px,py in tips]
    xx=[p[0] for p in tips]
    if len(xx)<1 or not all(a<b for a,b in zip(xx,xx[1:])):
        raise ValueError('Product tops must be provided from left to right.')
    if len(xx)==1:
        borders=[0,W]
    else:
        borders=[max(0,xx[0]-.52*(xx[1]-xx[0]))]
        image_gray=None
        if plate_bgr is not None:
            image_gray=cv2.cvtColor(plate_bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
            image_gray=cv2.GaussianBlur(image_gray,(0,0),1.2)
            ridge=np.maximum(image_gray-cv2.GaussianBlur(image_gray,(0,0),10),0)
            dx=cv2.Sobel(image_gray,cv2.CV_32F,1,0,ksize=3)
        for i in range(len(xx)-1):
            a,b=xx[i],xx[i+1];gap=b-a;boundary=(a+b)*.5
            if image_gray is not None and gap>=50:
                probe_y=int(max(tips[i][1],tips[i+1][1])+.40*(H-max(tips[i][1],tips[i+1][1])))
                ya=max(0,probe_y-12);yb=min(H,probe_y+13)
                right_signal=np.mean((ridge+.24*np.maximum(0,-dx))[ya:yb],axis=0)
                left_signal=np.mean((ridge+.24*np.maximum(0,dx))[ya:yb],axis=0)
                def peak(sig,frac0,frac1):
                    lo=max(0,int(a+frac0*gap));hi=min(W-1,int(a+frac1*gap))
                    if hi<lo+5:return None,0
                    loc=lo+int(np.argmax(sig[lo:hi+1]))
                    return loc,float(sig[loc])
                right,rs=peak(right_signal,.26,.55)
                left,ls=peak(left_signal,.64,.94)
                if (right is not None and left is not None and ls>14 and rs>12
                        and right <= boundary + .05*gap and 7 < left-right < .35*gap):
                    measured=(right+left)/2
                    boundary=.85*measured+.15*boundary
                    boundary=np.clip(boundary,a+.40*gap,a+.72*gap)
            borders.append(boundary)
        borders.append(min(W,xx[-1]+.52*(xx[-1]-xx[-2])))
    out=[]
    for i,(px,py) in enumerate(tips):
        x0=max(0,int(np.floor(borders[i])));x1=min(W,int(np.ceil(borders[i+1])))
        top=max(0,int(py-12))
        out.append((x0,top,max(1,x1-x0),H-top))
    return out

def follow_tongue(img, tip, guide):
    H,W=img.shape[:2];tx,ty=map(float,tip); x,y,gw,gh=guide
    left_bound=max(0,int(x));right_bound=min(W,int(x+gw))
    if not(left_bound+5<tx<right_bound-5):return None
    ytop=int(np.clip(round(ty),0,H-20))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    sm=cv2.GaussianBlur(gray,(0,0),1.2)
    ridge=np.maximum(0,sm-cv2.GaussianBlur(sm,(0,0),max(7,.04*gw)))
    signed_dx=cv2.Sobel(sm,cv2.CV_32F,1,0,ksize=3)
    # Evidence is positive for a bright perimeter ridge or for the correct
    # signed contrast transition at the left/right edge.
    cap_h=int(np.clip(.30*gw,18,90))
    shoulder=min(H-2,ytop+cap_h)
    rows=np.arange(ytop,H)
    edge_paths=[]
    for side in (-1,1):
        strength=ridge + .24*np.maximum(0,-side*signed_dx)
        # Shoulder candidates constrained to the selected lane and a physically
        # plausible distance from the manually marked rounded tip.
        minsep=max(5.,.11*gw)
        maxsep=min(.49*gw,(tx-left_bound-2 if side==-1 else right_bound-tx-2))
        xc1=int(max(left_bound+2,tx-maxsep)) if side==-1 else int(tx+minsep)
        xc2=int(tx-minsep) if side==-1 else int(min(right_bound-2,tx+maxsep))
        if xc2<=xc1+3: return None
        col=np.arange(xc1,xc2+1)
        nearby=strength[max(0,shoulder-9):min(H,shoulder+10), :]
        profile=np.mean(nearby[:,col],axis=0)
        # Prevent a far plate edge/droplet from beating an actual boundary.
        rough=tx+side*.28*gw
        prior_penalty=3.*((col-rough)/(.30*gw))**2
        col0=int(col[np.argmax(profile-prior_penalty)])

        # Trace downwards from the confident shoulder anchor. The tracker only
        # searches close to its previous boundary: it cannot jump to the next
        # product or to an isolated patch of high cleaning intensity.
        path=np.empty(H-shoulder,np.float32)
        pos=float(col0)
        v=side*.14
        confidence=np.empty(len(path),np.float32)
        for j,row in enumerate(range(shoulder,H)):
            forecast=np.clip(pos+v,left_bound+2,right_bound-3)
            search=max(6,int(.055*gw))
            lo=max(left_bound+2,int(forecast-search));hi=min(right_bound-2,int(forecast+search))
            if lo>=hi:
                path[j]=pos;confidence[j]=0;continue
            cc=np.arange(lo,hi+1)
            line=cv2.GaussianBlur(strength[max(row-1,0):min(row+2,H),:].mean(axis=0)[None,:],(5,1),0).ravel()
            sc=line[cc]-.24*((cc-forecast)**2)
            winner=int(cc[np.argmax(sc)])
            peak=float(line[winner]);local_med=float(np.median(line[cc]))
            credible=peak>max(4.5,local_med+1.5)
            if credible:
                shift=float(np.clip(winner-forecast,-2.4,2.4))
                new=forecast+.40*shift
                confidence[j]=min(1.,(peak-4.5)/10.)
                # The predicted velocity learns slowly from credible edges.
                v=np.clip(.90*v+.10*(new-pos),-.65,.65)
            else:
                # Weak evidence: continue smoothly based on the last observed
                # trajectory rather than narrowing or finding another edge.
                new=forecast
                v=np.clip(.995*v + .005*side*.10,-.35,.35)
                confidence[j]=0.
            pos=float(np.clip(new,left_bound+2,right_bound-3))
            path[j]=pos
        # smooth high-frequency scanner noise; preserve large-scale contours
        path=_sm(path,max(4,.035*gw))
        # Gentle one-sided shape constraint: no substantial inward notches.
        peak=path[0]
        for j in range(1,len(path)):
            if side<0:
                peak=min(peak,path[j]);path[j]=min(path[j],peak+4.)
            else:
                peak=max(peak,path[j]);path[j]=max(path[j],peak-4.)
        edge_paths.append((col0,path))
    (left0,left),(right0,right)=edge_paths
    if len(left)!=len(right): return None
    left=np.clip(left,left_bound,right_bound)
    right=np.clip(right,left_bound,right_bound)
    bodycenter=_sm(.5*(left+right),max(5,.026*gw))
    widths=np.maximum(right-left,5)
    widths=_sm(widths,max(4,.026*gw))
    mask=np.zeros((H,W),np.uint8)
    # Elliptical rounding at the top: widest at shoulder, curved rather than flat.
    ctr=float(.5*(left0+right0))
    rad=float(.5*(right0-left0))
    if rad<5:return None
    for row in range(ytop,shoulder):
        t=(row-ytop+.35)/max(1,shoulder-ytop)
        half=rad*np.sqrt(max(0.,1.-(1.-t)**2))
        c=ctr
        l=max(left_bound,int(round(c-half)));r=min(right_bound,int(round(c+half))+1)
        if l<r:mask[row,l:r]=255
    for j,row in enumerate(range(shoulder,H)):
        l=max(left_bound,int(round(bodycenter[j]-widths[j]/2)))
        r=min(right_bound,int(round(bodycenter[j]+widths[j]/2))+1)
        if l<r:mask[row,l:r]=255
    return mask



# ---------------------- image-boundary refinement ----------------------
# First build a continuous tongue using the existing top-point follower. Then
# locally optimize each side toward vertically persistent bright rim evidence.
# Uncertain rows interpolate back to that tongue, rather than chasing unrelated
# white cleaning streaks. The cap is separately matched to the bright top edge.

def _dp_refined_edge(evidence, base, leftlim, rightlim, tip_x, side, shoulder, gw):
    H,W=evidence.shape
    margin=int(max(8,.09*gw))
    rows=np.arange(shoulder,H)
    if len(rows)<4:return base
    lo=max(leftlim+2,int(np.floor(np.min(base[rows])-margin)))
    hi=min(rightlim-3,int(np.ceil(np.max(base[rows])+margin)))
    if side<0: hi=min(hi,int(tip_x+5))
    else:lo=max(lo,int(tip_x-5))
    if hi-lo<6:return base
    cols=np.arange(lo,hi+1)
    n=len(cols)
    # From original signal, use its vertical persistence to suppress horizontal markings.
    scores=evidence[rows][:,cols]
    # Contrast and local vertical persistence in natural BGR luminance units.
    scores=np.minimum(scores,30)
    costs=np.empty_like(scores,dtype=np.float32)
    diff=cols[None,:]-base[rows,None]
    costs[:]=1.1*scores - .030*diff*diff
    # gradually clamp baseline at upper shoulder
    cost0=costs[0]-.20*(cols-base[shoulder])**2
    prev=np.zeros((len(rows),n),dtype=np.int16)
    opts=np.arange(n)
    best=cost0.copy()
    for j in range(1,len(rows)):
        win=np.full(n,-1e10,np.float32); parent=np.zeros(n,np.int16)
        for delta in range(-3,4):
            source=np.clip(opts+delta,0,n-1)
            possible=best[source] - (.65*np.abs(delta)+.42*delta**2)
            good=possible>win
            win[good]=possible[good];parent[good]=source[good]
        best=win+costs[j]
        prev[j]=parent
    ix=int(np.argmax(best))
    path=np.empty(len(rows),np.float32)
    for j in range(len(rows)-1,-1,-1):
        path[j]=cols[ix];ix=prev[j,ix]
    path=_sm(path,max(7,.055*gw))
    # Blend in observations depending on local relative strength vs nearby alternatives
    corrected=base.copy()
    strength=evidence[rows,np.clip(np.rint(path).astype(int),0,W-1)]
    reference=np.quantile(scores,.50,axis=1)
    rel=strength-reference
    conf=np.clip((rel-1.0)/7.,0,1)
    conf=np.minimum(_sm(conf,12),.42)
    corrected[rows]=base[rows]+np.clip(conf*(path-base[rows]),-8,8)
    return corrected


def _photo_refined(img,tip,guide,base):
    if base is None:return None
    H,W=base.shape;x,y,gw,gh=guide
    tx,ty=tip
    cap=int(np.clip(.30*gw,18,90)); shoulder=min(H-2,int(ty)+cap)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    fine=cv2.GaussianBlur(gray,(0,0),1.25)
    long=cv2.GaussianBlur(fine,(0,0),max(6,.035*gw))
    ridge=np.maximum(0,fine-long)
    grad=cv2.Sobel(fine,cv2.CV_32F,1,0,ksize=3)
    current=[]
    for side in [-1,1]:
        vals=np.empty(H,np.float32)
        for j in range(H):
            xx=np.flatnonzero(base[j]); vals[j] = (xx.min() if side==-1 else xx.max()) if xx.size else np.nan
        vv=np.where(np.isfinite(vals))[0]
        if vv.size<10: return base
        vals=np.interp(np.arange(H),vv,vals[vv]).astype(np.float32)
        # Positive ridge with directional contrast.
        directed=np.maximum(0,-side*grad)
        raw=np.clip(ridge+ .18*directed,0,30)
        persistence=cv2.GaussianBlur(raw,(0,0),sigmaX=1.7,sigmaY=9)
        evidence=(.48*raw+.52*persistence).astype(np.float32)
        current.append(_dp_refined_edge(evidence,vals,max(0,int(x)),min(W,int(x+gw)),tx,side,shoulder,gw))
    L,R=current
    # combine and suppress sharp inward errors, modest natural narrowing allowed
    width=np.maximum(2,R-L)
    average=_sm(width,max(9,.06*gw))
    width=np.maximum(width,.92*average)
    width=_sm(width,max(6,.045*gw))
    center=_sm((L+R)/2,max(7,.05*gw))
    out=base.copy()
    for j in range(shoulder,H):
        out[j]=0
        lo=max(int(x),int(round(center[j]-width[j]/2)))
        hi=min(int(x+gw),int(round(center[j]+width[j]/2))+1)
        if hi>lo:out[j,lo:hi]=255
    return out


def _photo_refined_cap(img,mask,tip,guide):
    H,W=mask.shape
    tx,ty=tip;x,y,gw,gh=guide
    cap=int(np.clip(.30*gw,18,90));shoulder=min(H-2,int(ty)+cap)
    candidate=mask[max(0,int(ty)-5):shoulder+2,:]
    if not candidate.any():return mask
    line=np.flatnonzero(mask[shoulder,:])
    if len(line)<25:return mask
    xl,xr=int(line[0]),int(line[-1])
    coords=np.arange(xl,xr+1)
    baseline=[]
    for c in coords:
        yy=np.flatnonzero(mask[:shoulder+3,c])
        baseline.append(yy[0] if yy.size else shoulder)
    baseline=np.asarray(baseline,np.float32)
    smooth=cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),1.3)
    ridge=np.maximum(0,smooth-cv2.GaussianBlur(smooth,(0,0),9))
    gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
    ev=np.maximum(0,ridge+.17*np.maximum(gy,0))
    persist=cv2.GaussianBlur(ev,(0,0),sigmaX=5,sigmaY=2.1)
    ev=.65*ev+.35*persist
    radius=max(8,min(19,int(.12*gw)))
    ymin=max(0,int(np.floor(np.min(baseline)-radius)))
    ymax=min(shoulder+radius,int(np.ceil(np.max(baseline)+radius)))
    ry=np.arange(ymin,ymax+1)
    prev=np.zeros((len(coords),len(ry)),np.int16)
    init=ev[ry,coords[0]]-.05*(ry-baseline[0])**2
    best=init.copy()
    idx=np.arange(len(ry))
    for j,c in enumerate(coords[1:],1):
        win=np.full(len(ry),-1e9,dtype=np.float32); parent=np.zeros(len(ry),np.int16)
        for shift in range(-3,4):
            old=np.clip(idx+shift,0,len(idx)-1)
            val=best[old]-(.50*np.abs(shift)+.35*shift*shift)
            good=val>win;win[good]=val[good];parent[good]=old[good]
        best=win+ev[ry,c]-.09*(ry-baseline[j])**2
        prev[j]=parent
    ix=int(np.argmax(best))
    path=np.empty(len(coords),np.float32)
    for j in range(len(coords)-1,-1,-1):
        path[j]=ry[ix];ix=prev[j,ix]
    path=_sm(path,3.4)
    # confidence weighting reduces noise in weak top patches
    conf=np.clip((ev[np.clip(np.round(path).astype(int),0,H-1),coords]-1)/10,0,1)
    conf=_sm(conf,9)
    path=.35*path+.65*baseline
    path=(.2+.7*conf)*path+(.8-.7*conf)*baseline
    out=mask.copy()
    out[:shoulder+1,xl:xr+1]=0
    for i,c in enumerate(coords):
        yy=int(np.clip(round(path[i]),0,shoulder+1))
        out[yy:shoulder+1,c]=255
    return out



# ------------------- high-contrast track side refinement -------------------
# Run only when the inside of a product track is uniformly far brighter than
# the background (e.g. nearly fully cleaned product D). The algorithm searches
# outward from the existing, geometrically stable footprint side and accepts
# nearby white-to-dirty transitions, with a conservative displacement limit.
# Otherwise return the original footprint completely unchanged. In particular,
# never move the rounded tip selected by the user or let bright artefacts above
# that point pull the mask upwards.

def _refine_bright_track_sides(img,mask,tip,guide):
    """Adjust side edges of bright, high-contrast tracks without changing user tip/cap."""
    H,W=mask.shape
    gx,gy,gw,gh=guide
    tipx,tipy=tip
    shoulder=min(H-2,int(round(tipy))+int(np.clip(.30*gw,18,90)))
    if shoulder>=H-15:return mask
    lum=cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),2.0)
    rows=np.arange(shoulder+6,H-7, max(2,int(.015*H)))
    inside=[];outside=[]
    for y in rows:
        xx=np.flatnonzero(mask[y])
        if xx.size<14:continue
        a,b=int(xx[0]),int(xx[-1]);center=(a+b)//2; rad=(b-a)*.2
        inside.extend(lum[y,max(a,int(center-rad)):min(b+1,int(center+rad)+1):max(1,int(rad//3))].tolist())
        for l,r in ((max(int(gx),a-22),max(int(gx),a-8)),(min(int(gx+gw),b+8),min(int(gx+gw),b+22))):
            if r>l:outside.extend(lum[y,l:r:4].tolist())
    if not inside or not outside:return mask
    inn=float(np.median(inside));out=float(np.median(outside))
    # only high-contrast, near-white contact tracks. Leave less-cleaned tracks alone.
    # Do not treat ordinary bright streaks (A) like a fully cleaned white track.
    if inn < 185 or np.percentile(inside,10) < 175 or inn-out < 43:
        return mask
    thr=out+.58*(inn-out)
    thr=np.clip(thr,138,210)
    maxd=max(7,int(.105*gw))
    L=[];R=[];qL=[];qR=[]
    for y in range(shoulder,H):
        xx=np.flatnonzero(mask[y]);
        if len(xx)==0:return mask
        l,r=int(xx[0]),int(xx[-1]);p=lum[y];
        for side,base,vals,qual in ((-1,l,L,qL),(1,r,R,qR)):
            lo=max(int(gx)+2,base-maxd);hi=min(int(gx+gw)-3,base+maxd)
            crosses=[]
            for z in range(lo+5,hi-5):
                a=np.mean(p[z-5:z]);b=np.mean(p[z+1:z+6]);contrast=b-a if side<0 else a-b
                if contrast>9 and ((side<0 and a<thr and b>thr) or (side>0 and a>thr and b<thr)):
                    score=contrast - .30*abs(z-base)
                    crosses.append((score,z,contrast))
            if crosses:
                _,best,contrast=max(crosses)
                vals.append(float(best));qual.append(min(1.,max(0.,(contrast-8)/20)))
            else:
                vals.append(float(base));qual.append(0.)
    n=len(L); baseL=np.array([np.flatnonzero(mask[y])[0] for y in range(shoulder,H)],float);baseR=np.array([np.flatnonzero(mask[y])[-1] for y in range(shoulder,H)],float)
    def sm(x,s):return cv2.GaussianBlur(np.asarray(x,np.float32)[:,None],(1,0),sigmaY=s,sigmaX=0).ravel()
    ql=sm(qL,6);qr=sm(qR,6)
    cl=sm(L,4);cr=sm(R,4)
    l=baseL+.85*ql*(cl-baseL);r=baseR+.85*qr*(cr-baseR)
    l=sm(l,3);r=sm(r,3)
    # fade correction in at cap/body junction
    for k in range(min(14,n)):
        a=k/14;l[k]=baseL[k]*(1-a)+l[k]*a;r[k]=baseR[k]*(1-a)+r[k]*a
    outmask=mask.copy()
    for j,y in enumerate(range(shoulder,H)):
        a=max(int(gx),int(round(l[j])));b=min(int(gx+gw),int(round(r[j]))+1)
        if b>a+4:outmask[y,:]=0;outmask[y,a:b]=255
    return outmask

def propose_footprint(clean_frac, guide_roi, plate_bgr=None, tip_xy=None):
    """Trace the contacted tongue downwards from its user-marked rounded top.

    Use photographic rim contrast, not high cleaning fraction (dark interiors are
    still contacted). When the rim weakens, propagate the smooth boundary.
    """
    H, W = clean_frac.shape[:2]
    if plate_bgr is None:
        val = np.uint8(255*np.clip(np.nan_to_num(clean_frac),0,1))
        plate_bgr = cv2.cvtColor(val,cv2.COLOR_GRAY2BGR)
    if plate_bgr.shape[:2] != (H,W):
        raise ValueError('Plate and cleaning fraction must have identical dimensions')
    x,y,w,h=guide_roi
    if tip_xy is None:
        tip_xy=(x+.5*w, y+12.)
    scale=min(1.,900./max(W,1))
    if scale<1.:
        small=cv2.resize(plate_bgr,(max(2,round(W*scale)),max(2,round(H*scale))),interpolation=cv2.INTER_AREA)
    else: small=plate_bgr
    small_guide=tuple(float(v)*scale for v in guide_roi)
    small_tip=tuple(float(v)*scale for v in tip_xy)
    mask=follow_tongue(small,small_tip,small_guide)
    if mask is not None and np.any(mask):
        refined=_photo_refined(small,small_tip,small_guide,mask)
        if refined is not None and np.any(refined):
            mask=_photo_refined_cap(small,refined,small_tip,small_guide)
            mask=_refine_bright_track_sides(small,mask,small_tip,small_guide)
    if mask is None or not np.any(mask):
        # Safe fallback still has a rounded tip and stays in its own corridor.
        hs,ws=small.shape[:2]; gx,gy,gw,gh=small_guide; tx,ty=small_tip
        mask=np.zeros((hs,ws),np.uint8)
        cap=max(15,int(.29*gw))
        left=max(0,int(gx));right=min(ws,int(gx+gw))
        rad=max(5,min(.35*gw,tx-left,right-tx))
        for iy in range(max(0,int(ty)),hs):
            t=min(1.,(iy-ty+.35)/cap)
            rr=rad*np.sqrt(max(0,1-(1-t)**2))
            l=max(left,int(tx-rr));r=min(right,int(tx+rr)+1)
            if l<r:mask[iy,l:r]=255
    if scale<1.:
        mask=cv2.resize(mask,(W,H),interpolation=cv2.INTER_NEAREST)
    for row in mask:
        active=np.flatnonzero(row)
        if active.size:
            row[active[0]:active[-1]+1]=255
    return mask

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

