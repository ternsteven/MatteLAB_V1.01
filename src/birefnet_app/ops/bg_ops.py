import numpy as np, cv2
from PIL import Image

def hex_to_rgb(x):
    if isinstance(x, (tuple, list)) and len(x) == 3:
        r,g,b = [int(v) for v in x]
        return max(0,min(r,255)), max(0,min(g,255)), max(0,min(b,255))
    s = (x or "").strip(); s = s[1:] if s.startswith("#") else s
    if len(s)==3: s = "".join(c*2 for c in s)
    try: return int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)
    except: return (0,255,0)

def resize_bg_keep_aspect(bg, w, h, mode="cover"):
    H,W = bg.shape[:2]; src = W/H; dst = w/h
    if mode=="contain":
        if src>dst: new_w=w; new_h=int(new_w/src)
        else: new_h=h; new_w=int(new_h*src)
        r = cv2.resize(bg,(new_w,new_h))
        canvas = np.zeros((h,w,bg.shape[2]), bg.dtype); canvas[...] = r[0,0]
        y=(h-new_h)//2; x=(w-new_w)//2; canvas[y:y+new_h,x:x+new_w]=r; return canvas
    else:
        if src<dst: new_h=h; new_w=int(new_h*src)
        else: new_w=w; new_h=int(new_w/src)
        r = cv2.resize(bg,(new_w,new_h))
        y=max(0,(new_h-h)//2); x=max(0,(new_w-w)//2)
        return r[y:y+h,x:x+w]

def create_background(kind, data, size_wh):
    w,h = size_wh
    if kind=="image" and data is not None:
        arr = np.array(data) if isinstance(data, Image.Image) else np.asarray(data)
        if arr.shape[:2]!=(h,w):
            arr = resize_bg_keep_aspect(arr,w,h,"cover")
        return arr
    if kind=="color" and data is not None:
        r,g,b = hex_to_rgb(data); return np.full((h,w,3),(r,g,b),np.uint8)
    return None  # transparent

def _estimate_background_inpaint(rgb_u8: np.ndarray, bin_mask: np.ndarray, radius:int)->np.ndarray:
    img8 = rgb_u8 if rgb_u8.dtype==np.uint8 else np.clip(rgb_u8,0,255).astype(np.uint8)
    H,W = img8.shape[:2]
    band_px = max(2, min(6, int(2+0.5*radius)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band_px+1,2*band_px+1))
    dil = cv2.dilate((bin_mask>0).astype(np.uint8),ker,1)
    ero = cv2.erode((bin_mask>0).astype(np.uint8),ker,1)
    band = cv2.subtract(dil,ero)
    ys,xs = np.where(band>0)
    if ys.size==0:
        return cv2.GaussianBlur(img8,(9,9),3)
    pad=8; y0,y1=max(0,ys.min()-pad), min(H,ys.max()+pad+1)
    x0,x1=max(0,xs.min()-pad), min(W,xs.max()+pad+1)
    roi_img = img8[y0:y1,x0:x1]; roi_m = (band[y0:y1,x0:x1]*255).astype(np.uint8)
    max_side = max(roi_img.shape[:2]); scale=0.5 if max_side>800 else 1.0
    if scale<1.0:
        nw,nh=int((x1-x0)*scale), int((y1-y0)*scale)
        roi_s = cv2.resize(roi_img,(nw,nh)); m_s = cv2.resize(roi_m,(nw,nh),interpolation=cv2.INTER_NEAREST)
    else:
        roi_s, m_s = roi_img, roi_m
    bgr = cv2.cvtColor(roi_s,cv2.COLOR_RGB2BGR)
    r = max(3,min(12,int(radius)))
    bgr_bg = cv2.inpaint(bgr,m_s,r,cv2.INPAINT_TELEA)
    if scale<1.0: bgr_bg = cv2.resize(bgr_bg,(x1-x0,y1-y0))
    out = img8.copy(); out[y0:y1,x0:x1]=cv2.cvtColor(bgr_bg,cv2.COLOR_BGR2RGB); return out

def _srgb_to_linear(x): x=np.clip(x,0,1); return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def _linear_to_srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,x*12.92,1.055*(x**(1/2.4))-0.055)

def _map_defringe_strength(s: float):
    s=float(max(0.0,min(1.0,s)))
    if s<0.30: band_px=1
    elif s<0.60: band_px=2
    elif s<0.85: band_px=3
    else: band_px=4
    if s<0.35: erode_px=0
    elif s<0.55: erode_px=1
    elif s<0.75: erode_px=2
    elif s<0.90: erode_px=3
    else: erode_px=4
    strength = 0.45 + 0.50*s
    return dict(strength=strength, band_px=band_px, erode_px=erode_px)

def _color_decontam_edge(rgb_u8: np.ndarray, mask_u8: np.ndarray, band_px=2, strength=0.7):
    H,W = rgb_u8.shape[:2]
    m = mask_u8 if mask_u8.dtype==np.uint8 else (np.clip(mask_u8,0,1)*255+0.5).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band_px+1,2*band_px+1))
    fg = (m>0).astype(np.uint8)
    dil = cv2.dilate(fg,ker,1); ero = cv2.erode(fg,ker,1); band = cv2.subtract(dil,ero)
    ys,xs = np.where(band>0)
    if ys.size==0: return rgb_u8
    pad=8; y0,y1=max(0,ys.min()-pad), min(H,ys.max()+pad+1)
    x0,x1=max(0,xs.min()-pad), min(W,xs.max()+pad+1)
    fg_roi = rgb_u8[y0:y1,x0:x1]; m_roi = m[y0:y1,x0:x1]
    B = _estimate_background_inpaint(rgb_u8, (m>0).astype(np.uint8), radius=max(3,int(3+4*band_px+6*float(strength))))
    B_roi = B[y0:y1,x0:x1]
    a = (m_roi.astype(np.float32)/255.0)
    C = fg_roi.astype(np.float32)/255.0; Bl = _srgb_to_linear(B_roi.astype(np.float32)/255.0)
    eps=1e-4
    Fl = ( _srgb_to_linear(C) - (1.0 - a)[...,None]*Bl )/np.maximum(a[...,None], eps)
    F = _linear_to_srgb(np.clip(Fl,0,1))
    S=float(strength)
    out = (1.0 - S)*C + S*F
    out = np.clip(out*255+0.5,0,255).astype(np.uint8)
    ret = rgb_u8.copy(); ret[y0:y1,x0:x1]=out; return ret

def _remove_white_halo_rgba(rgba: np.ndarray, mask: np.ndarray, band_px=2, strength=0.7, erode_px=1):
    H,W = rgba.shape[:2]
    m = mask if mask.dtype==np.uint8 else (np.clip(mask,0,1)*255+0.5).astype(np.uint8)
    rgb = rgba[:,:,:3]
    rgb_fixed = _color_decontam_edge(rgb, m, band_px=max(1,band_px), strength=float(strength))
    if erode_px>0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*erode_px+1,2*erode_px+1))
        m = cv2.erode(m, ker, 1)
    a = cv2.GaussianBlur(m,(0,0),0.6)
    return np.dstack([rgb_fixed, a]).astype(np.uint8)

def replace_background_with_mask(image_array, background_array, mask, *, remove_white_halo=False, defringe_strength=0.7):
    fg = np.asarray(image_array); bg = np.asarray(background_array); H,W = fg.shape[:2]
    if bg.shape[:2]!=(H,W): bg = resize_bg_keep_aspect(bg,W,H,"cover")
    m = mask; 
    if m.ndim==3: m = m[:,:,0] if m.shape[2]==1 else cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    a_u8 = m if m.dtype==np.uint8 else (np.clip(m,0,1)*255+0.5).astype(np.uint8)
    if remove_white_halo:
        params=_map_defringe_strength(defringe_strength)
        fg = _color_decontam_edge(fg, a_u8, band_px=params["band_px"], strength=params["strength"])
        if params["erode_px"]>0:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*params["erode_px"]+1,2*params["erode_px"]+1))
            a_u8 = cv2.erode(a_u8,ker,1)
    a = (a_u8.astype(np.float32)/255.0)[...,None]
    out = fg.astype(np.float32)*a + bg.astype(np.float32)*(1.0-a)
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))

def create_transparent_result(image_array, mask, *, remove_white_halo=False, defringe_strength=0.7):
    img = np.asarray(image_array)
    if img.ndim==2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2]>=4: img = img[:,:,:3]
    m = mask; 
    if m.ndim==3: m = m[:,:,0] if m.shape[2]==1 else cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    a = m if m.dtype==np.uint8 else (np.clip(m,0,1)*255+0.5).astype(np.uint8)
    rgba = np.dstack([img, a]).astype(np.uint8)
    if remove_white_halo:
        p=_map_defringe_strength(defringe_strength)
        rgba = _remove_white_halo_rgba(rgba, a, band_px=p["band_px"], strength=p["strength"], erode_px=p["erode_px"])
    return Image.fromarray(rgba, mode="RGBA")
