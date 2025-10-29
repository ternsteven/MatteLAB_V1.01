import numpy as np, cv2
from PIL import Image
from .bg_ops import _estimate_background_inpaint  # from bg_ops

def to_binary_mask(mask: np.ndarray, *, use_otsu=True) -> np.ndarray:
    m = mask if mask.dtype==np.uint8 else (np.clip(mask,0,1)*255+0.5).astype(np.uint8)
    if use_otsu:
        _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _, m = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, 1)
    return m.astype(np.uint8)

def estimate_soft_alpha_inside_mask(image_or_array, base_mask: np.ndarray, *, strength:float=0.5, mode:str="auto")->np.ndarray:
    I = np.array(image_or_array.convert("RGB")) if isinstance(image_or_array, Image.Image) else np.asarray(image_or_array)
    if I.ndim==3 and I.shape[2]==4: I = I[:,:,:3]
    m8 = base_mask if base_mask.dtype==np.uint8 else (np.clip(base_mask,0,1)*255+0.5).astype(np.uint8)
    binm = (m8>=128).astype(np.uint8)
    inpaint_r = 2 + int(8*float(strength))
    B = _estimate_background_inpaint(I, binm, radius=inpaint_r)

    I_f = I.astype(np.float32)/255.0
    B_f = B.astype(np.float32)/255.0
    eps=1e-5

    er_r = max(1,int(2+6*float(strength)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*er_r+1,2*er_r+1))
    fg_core = cv2.erode(binm, ker, 1); 
    if fg_core.sum()<10: fg_core = binm.copy()
    F_ref = np.array([np.median(I_f[:,:,c][fg_core>0]) if (I_f[:,:,c][fg_core>0]).size>0 else 0.5 for c in range(3)], np.float32)

    num = np.abs(I_f - B_f).sum(axis=2)
    den = (np.abs(F_ref.reshape(1,1,3) - B_f)).sum(axis=2) + eps
    alpha = np.clip(num/den, 0.0, 1.0)

    if mode in ("暗部优先","dark"):
        gamma = max(0.4, 1.0-0.5*float(strength))
        alpha = np.power(alpha, gamma)
    elif mode in ("透色优先","bleed"):
        gamma = 1.2 + 0.8*float(strength)
        alpha = np.power(alpha, gamma)

    try:
        import cv2.ximgproc as xip
        gf_r = max(3, int(3+8*float(strength)))
        alpha = xip.fastGuidedFilter((I_f*255).astype(np.uint8), alpha.astype(np.float32), radius=gf_r, eps=1e-4)
    except Exception:
        d = 5 + 2*int(3+8*float(strength))
        alpha = cv2.bilateralFilter(alpha.astype(np.float32), d=d, sigmaColor=0.08, sigmaSpace=max(5,d))

    alpha = alpha * (binm>0).astype(np.float32)
    return (np.clip(alpha,0,1)*255).astype(np.uint8)

def refine_alpha_with_channel(image_or_array, base_mask: np.ndarray, mode="auto", strength=0.5)->np.ndarray:
    img = np.array(image_or_array.convert("RGB")) if isinstance(image_or_array, Image.Image) else np.asarray(image_or_array)
    if img.ndim==3 and img.shape[2]==4: img = img[:,:,:3]
    H,W = img.shape[:2]
    base = base_mask if base_mask.dtype==np.uint8 else (np.clip(base_mask,0,1)*255+0.5).astype(np.uint8)
    base_alpha = base.astype(np.float32)/255.0
    binary = (base>=128).astype(np.uint8)
    radius = max(1, int(2+strength*10))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    dil = cv2.dilate(binary, ker, 1); ero = cv2.erode(binary, ker, 1)
    unknown = cv2.subtract(dil, ero)
    solid_r = max(1, radius*2)
    ker_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*solid_r+1,2*solid_r+1))
    fg_solid = cv2.erode(binary, ker_s, 1)
    bg_solid = cv2.erode(1-binary, ker_s, 1)

    I = img.astype(np.float32)/255.0
    F = np.array([np.median(I[:,:,c][fg_solid>0]) if (I[:,:,c][fg_solid>0]).size>0 else 0.8 for c in range(3)], np.float32)
    B = np.array([np.median(I[:,:,c][bg_solid>0]) if (I[:,:,c][bg_solid>0]).size>0 else 0.2 for c in range(3)], np.float32)
    den = F-B; w = np.abs(den); w = w/(w.sum()+1e-6)
    alpha_u = np.zeros((H,W), np.float32); eps=1e-4
    for c in range(3):
        if w[c]<=1e-6: continue
        alpha_u += w[c]*( (I[:,:,c]-B[c])/(den[c]+eps) )
    alpha_u = np.clip(alpha_u, 0, 1)
    if mode in ("暗部优先","dark"):
        gamma = 1.2 + 0.8*(1-strength); alpha_u = np.power(alpha_u, gamma)
    elif mode in ("透色优先","bleed"):
        gamma = max(0.5, 1.0-0.5*strength); alpha_u = np.power(alpha_u, gamma)
    mix = 0.35 + 0.55*strength
    u = (unknown>0).astype(np.float32)
    final = base_alpha*(1.0-u) + ((1-mix)*base_alpha + mix*alpha_u)*u
    try:
        import cv2.ximgproc as xip
        final = xip.guidedFilter(I, final.astype(np.float32), radius=radius*2+1, eps=1e-4)
    except Exception:
        d = 5 + 2*radius
        final = cv2.bilateralFilter(final.astype(np.float32), d=d, sigmaColor=0.1, sigmaSpace=radius*2+1)
    return (np.clip(final,0,1)*255).astype(np.uint8)
