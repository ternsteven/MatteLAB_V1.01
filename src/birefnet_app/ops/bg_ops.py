import numpy as np, cv2
from PIL import Image
from PIL import Image as _Image

def hex_to_rgb(hex_color=None):
    if isinstance(hex_color, (tuple, list)) and len(hex_color) == 3:
        r, g, b = [int(x) for x in hex_color]
        return (max(0, min(r,255)), max(0, min(g,255)), max(0, min(b,255)))
    s = (hex_color or "").strip()
    if s.startswith("#"): s = s[1:]
    if len(s) == 3: s = "".join(c * 2 for c in s)
    try: return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except: return (0,255,0)

def _resize_bg_keep_aspect(bg_array: np.ndarray, target_w: int, target_h: int, mode: str = "cover") -> np.ndarray:
    h, w = bg_array.shape[:2]; src = w / h; dst = target_w / target_h
    if mode == "contain":
        if src > dst: new_w, new_h = target_w, int(target_w / src)
        else:         new_h, new_w = target_h, int(target_h * src)
        resized = cv2.resize(bg_array, (new_w, new_h))
        canvas = np.full((target_h, target_w, bg_array.shape[2]), resized[0,0], dtype=bg_array.dtype)
        y0 = (target_h - new_h)//2; x0 = (target_w - new_w)//2
        canvas[y0:y0+new_h, x0:x0+new_w] = resized
        return canvas
    else:
        if src < dst: new_h, new_w = target_h, int(target_h * src)
        else:         new_w, new_h = target_w, int(target_w / src)
        resized = cv2.resize(bg_array, (new_w, new_h))
        y0 = max(0, (new_h - target_h)//2); x0 = max(0, (new_w - target_w)//2)
        return resized[y0:y0+target_h, x0:x0+target_w]

def create_background(kind: str, data, image_size):
    w, h = image_size
    if kind == "image" and data is not None:
        arr = np.array(data) if isinstance(data, Image.Image) else data
        if arr.ndim == 2: arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.ndim == 3 and arr.shape[2] >= 4: arr = arr[:, :, :3]
        if arr.shape[:2] != (h, w): arr = _resize_bg_keep_aspect(arr, w, h, mode="cover")
        return arr
    if kind == "color" and data is not None:
        r,g,b = hex_to_rgb(data); return np.full((h, w, 3), (r,g,b), dtype=np.uint8)
    return None  # transparent

# —— 去白边/发丝保护（精简要点）——
def _map_defringe_strength(s: float):
    s = float(max(0.0, min(1.0, s)))
    if   s < 0.30: band_px = 1
    elif s < 0.60: band_px = 2
    elif s < 0.85: band_px = 3
    else:          band_px = 4
    if   s < 0.35: erode_px = 0
    elif s < 0.55: erode_px = 1
    elif s < 0.75: erode_px = 2
    elif s < 0.90: erode_px = 3
    else:          erode_px = 4
    strength = 0.45 + 0.50 * s
    return dict(strength=strength, band_px=band_px, erode_px=erode_px)

def _color_decontam_edge(rgb_u8: np.ndarray, mask_u8: np.ndarray, band_px=2, strength=0.7):
    # 仅演示关键思路：边带 ROI 反解前景颜色并线性混合，发丝处降低强度
    H, W = rgb_u8.shape[:2]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band_px+1, 2*band_px+1))
    fg = (mask_u8 > 0).astype(np.uint8)
    dil = cv2.dilate(fg, ker, 1); ero = cv2.erode(fg, ker, 1)
    band = cv2.subtract(dil, ero)
    if band.max() == 0: return rgb_u8
    # 简化：用高斯模糊的近似背景
    bg_est = cv2.GaussianBlur(rgb_u8, (11,11), 4)
    a = (mask_u8.astype(np.float32)/255.0)[...,None]
    F = (rgb_u8.astype(np.float32) - (1.0 - a) * bg_est.astype(np.float32)) / np.maximum(a, 1e-3)
    F = np.clip(F, 0, 255).astype(np.uint8)
    S = float(strength) * (band.astype(np.float32)/255.0)[...,None]
    out = (1.0 - S) * (rgb_u8.astype(np.float32)) + S * F.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

def _remove_white_halo_rgba(rgba: np.ndarray, mask_u8: np.ndarray, *, band_px=2, strength=0.7, erode_px=1):
    rgb = rgba[:, :, :3].copy(); a = rgba[:, :, 3].copy()
    rgb = _color_decontam_edge(rgb, mask_u8, band_px=band_px, strength=strength)
    if erode_px > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
        a = cv2.erode(a, ker, 1)
    a = cv2.GaussianBlur(a, (0,0), 0.6)
    return np.dstack([rgb, a]).astype(np.uint8)

def create_transparent_result(image_array, mask, *, remove_white_halo=False, defringe_strength=0.7):
    img = np.asarray(image_array); 
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] >= 4: img = img[:, :, :3]
    H, W = img.shape[:2]
    m = np.asarray(mask)
    if m.ndim == 3: m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY) if m.shape[2] != 1 else m[...,0]
    if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    a = m.astype(np.uint8)
    rgba = np.dstack([img, a]).astype(np.uint8)
    if remove_white_halo:
        p = _map_defringe_strength(defringe_strength)
        rgba = _remove_white_halo_rgba(rgba, a, band_px=p["band_px"], strength=p["strength"], erode_px=p["erode_px"])
    return Image.fromarray(rgba)

def replace_background_with_mask(image_array, background_array, mask, *, remove_white_halo=False, defringe_strength=0.7):
    fg = np.asarray(image_array); 
    if fg.ndim == 2: fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2RGB)
    if fg.ndim == 3 and fg.shape[2] >= 4: fg = fg[:, :, :3]
    H, W = fg.shape[:2]
    bg = np.asarray(background_array)
    if bg.ndim == 2: bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
    if bg.ndim == 3 and bg.shape[2] >= 4: bg = bg[:, :, :3]
    if bg.shape[:2] != (H, W): bg = _resize_bg_keep_aspect(bg, W, H, mode="cover")

    m = np.asarray(mask)
    if m.ndim == 3: m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY) if m.shape[2] != 1 else m[...,0]
    if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    a = (m.astype(np.float32) / 255.0)[..., None]
    out = fg.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if remove_white_halo:
        p = _map_defringe_strength(defringe_strength)
        rgba = _remove_white_halo_rgba(np.dstack([out, m]).astype(np.uint8), m, 
                                       band_px=p["band_px"], strength=p["strength"], erode_px=p["erode_px"])
        out = rgba[:, :, :3]
    return Image.fromarray(out)

def _estimate_background_inpaint(
    rgb_u8,                # HxWx3 或 HxW 或 HxWx4
    mask_u8,               # HxW 或 HxWx{1,3,4}；非零=前景（需要被修复掉）
    radius_px: int | None = None,
    method: str = "telea",
    **kwargs,              # 兼容未来多余参数，避免再次报签名不匹配
):
    """
    用 OpenCV inpaint 估计“抠除前景后的背景”。返回 RGB uint8, shape=(H,W,3)。
    """
    # -- 规范化图像为 RGB uint8 --
    if isinstance(rgb_u8, _Image):
        rgb = _np.array(rgb_u8.convert("RGB"))
    else:
        rgb = _np.asarray(rgb_u8)
        if rgb.ndim == 2:
            rgb = _cv2.cvtColor(rgb, _cv2.COLOR_GRAY2RGB)
        elif rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        elif rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"_estimate_background_inpaint: unexpected image shape {rgb.shape}")

    H, W = rgb.shape[:2]

    # -- 规范化 mask 为单通道 0/255 --
    m = _np.asarray(mask_u8)
    if m.ndim == 3:
        if m.shape[2] == 1:
            m = m[:, :, 0]
        elif m.shape[2] == 4:
            m = m[:, :, 3]  # 优先 alpha
        else:
            m = _cv2.cvtColor(m, _cv2.COLOR_RGB2GRAY)
    elif m.ndim != 2:
        raise ValueError(f"_estimate_background_inpaint: unexpected mask shape {m.shape}")
    if m.dtype != _np.uint8:
        if _np.issubdtype(m.dtype, _np.floating):
            m = (_np.clip(m, 0, 1) * 255.0 + 0.5).astype(_np.uint8)
        else:
            m = m.astype(_np.uint8)
    if m.shape != (H, W):
        m = _cv2.resize(m, (W, H), interpolation=_cv2.INTER_NEAREST)
    m = (m > 0).astype(_np.uint8) * 255

    # -- 自适应半径 & 方法 --
    if radius_px is None:
        # 与图像长边成比例，设一个温和的默认
        radius_px = max(3, int(max(H, W) / 512))
    radius_px = int(max(1, radius_px))
    flag = _cv2.INPAINT_TELEA if str(method).lower() == "telea" else _cv2.INPAINT_NS

    # -- inpaint 在 BGR 域执行 --
    bgr = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)
    out_bgr = _cv2.inpaint(bgr, m, radius_px, flag)
    out_rgb = _cv2.cvtColor(out_bgr, _cv2.COLOR_BGR2RGB)
    return out_rgb

# 公开别名（可选，供其他模块按无下划线名使用）
def estimate_background_inpaint(*args, **kwargs):
    return _estimate_background_inpaint(*args, **kwargs)