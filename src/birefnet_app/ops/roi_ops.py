
from typing import Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image


# --- helpers to access Gradio ImageEditor EditorValue (dict or pydantic object) ---
def _get_ev_field(ev, name, default=None):
    try:
        if isinstance(ev, dict):
            return ev.get(name, default)
        # Gradio v5 EditorValue is a pydantic model with attributes
        val = getattr(ev, name, default)
        # Some fields (background/composite) may be nested objects with a "image" or "value" attribute
        if hasattr(val, "image"):
            return val.image
        if hasattr(val, "value"):
            return val.value
        return val
    except Exception:
        return default

def _to_pil_or_np(x):
    try:
        if x is None:
            return None
        # EditorValue background/composite may already be numpy
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, Image.Image):
            return x
        # dict with "image"/"array" keys
        if isinstance(x, dict):
            arr = x.get("image") or x.get("array") or x.get("background")
            if arr is not None:
                return arr
        # Fallback: try PIL conversion from bytes-like
        return Image.fromarray(np.asarray(x))
    except Exception:
        return None
# --- 将缩略图长边压到 long_side，用于 ImageEditor 初值 ---
def make_editor_thumbnail(img_pil: Image.Image, long_side: int = 640) -> Tuple[Dict[str, Any], Dict[str, int]]:
    w, h = img_pil.size
    scale = min(float(long_side) / max(w, h), 1.0)
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    thumb = img_pil.resize((tw, th), Image.BILINEAR)
    ev = {"background": thumb, "layers": [], "composite": thumb}
    meta = {"ori_w": w, "ori_h": h, "thumb_w": tw, "thumb_h": th}
    return ev, meta

# --- 从 ImageEditor 的 value+meta 合成全尺寸 ROI 掩码（HxW, uint8 0/255） ---
def editor_layers_to_mask_fullres(editor_value: Dict[str, Any], meta: Dict[str, int]) -> np.ndarray | None:
    if not editor_value or not meta:
        return None
    tw, th, W, H = meta["thumb_w"], meta["thumb_h"], meta["ori_w"], meta["ori_h"]
    mask_thumb = np.zeros((th, tw), np.uint8)
    layers = _get_ev_field(editor_value, "layers") or []
    for layer in layers:
        if layer is None: 
            continue
        arr = np.array(layer)
        if arr.ndim == 3 and arr.shape[2] == 4:
            alpha = arr[..., 3]
            mask_thumb = np.maximum(mask_thumb, alpha.astype(np.uint8))
        elif arr.ndim == 3 and arr.shape[2] == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            mask_thumb = np.maximum(mask_thumb, (gray > 0).astype(np.uint8) * 255)
    if mask_thumb.max() == 0:
        comp = _get_ev_field(editor_value, "composite")
        bg = _get_ev_field(editor_value, "background")
        if comp is not None and bg is not None:
            # Normalize to RGBA arrays
            comp_img = comp if isinstance(comp, Image.Image) else Image.fromarray(np.asarray(comp)) if not isinstance(comp, np.ndarray) else Image.fromarray(comp)
            bg_img = bg if isinstance(bg, Image.Image) else Image.fromarray(np.asarray(bg)) if not isinstance(bg, np.ndarray) else Image.fromarray(bg)
            bg_rgba = np.array(bg_img.convert("RGBA"))
            comp_rgba = np.array(comp_img.convert("RGBA"))
            diff = np.abs(comp_rgba[..., :3].astype(np.int16) - bg_rgba[..., :3].astype(np.int16)).sum(axis=2)
            mask_thumb = (diff > 5).astype(np.uint8) * 255
    if mask_thumb.max() == 0:
        return None
    mask_full = cv2.resize(mask_thumb, (W, H), interpolation=cv2.INTER_NEAREST)
    return (mask_full > 0).astype(np.uint8) * 255

# --- 兼容函数：把任意图像转单通道 uint8（若工程内其它地方还在 import） ---
def to_single_channel_uint8(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype != np.uint8:
        if a.dtype.kind == "f":
            a = (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        else:
            a = a.astype(np.uint8)
    if a.ndim == 2: return a
    if a.ndim == 3:
        if a.shape[2] == 4:
            alpha = a[..., 3]
            return alpha if alpha.max() > 0 else cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
        if a.shape[2] == 3:
            return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported ndim: {a.ndim}")

# --- 兼容函数：compose 可能 import 的旧函数（若不再用可删） ---
def merge_with_brush(base_mask, fg_add=None, bg_erase=None, *, feather_px: int = 0, mode: str = "augment"):
    m = (to_single_channel_uint8(base_mask) >= 128).astype(np.uint8) * 255
    def bin8(z): 
        return (to_single_channel_uint8(z) >= 128).astype(np.uint8) * 255
    if mode == "replace":
        rep = np.zeros_like(m)
        if fg_add is not None: rep = np.maximum(rep, bin8(fg_add))
        if bg_erase is not None: rep = np.where(bin8(bg_erase) > 0, 0, rep).astype(np.uint8)
        out = rep
    elif mode == "intersect":
        if fg_add is None: 
            out = m
        else:
            out = cv2.bitwise_and(m, bin8(fg_add))
    else:
        out = m
        if fg_add is not None: out = np.maximum(out, bin8(fg_add))
        if bg_erase is not None: out = np.where(bin8(bg_erase) > 0, 0, out).astype(np.uint8)
    if feather_px and feather_px > 0:
        k = max(1, feather_px * 2 + 1)
        out = cv2.GaussianBlur(out, (k, k), sigmaX=feather_px)
    return out
