from __future__ import annotations
import numpy as np
from typing import Any, Dict, Tuple, Optional
from PIL import Image
import cv2  # 必须导入

# —— 半透明扣除说明文案（用于 UI 展示一致性） ——
SEMI_TIP = """
**扣除半透明**  
- 开关：默认关闭以保持旧版本行为。  

**力度 / 区域大小（0–1）** 影响 inpaint 半径、融合强度、平滑半径。  
建议：**烟雾** 0.6–0.8；**薄纱/纱网** 0.4–0.6；**玻璃/水面** 0.3–0.5。

**模式**  
- **auto**：自动选择，不再额外弯曲 α 曲线。  
- **暗部优先**：适合阴影、烟雾略压暗背景（更保守，防止过度透明）。  
- **透色优先**：适合薄纱、雾气高亮/低饱和（更开放，通透感更强）。  
半身人像/发丝建议先选 **Matting**，再开启本功能。
"""



__all__ = ["make_editor_thumbnail", "editor_layers_to_mask_fullres"]

# ---------- 工具 ----------

def _to_np(x: Any) -> Optional[np.ndarray]:
    """将 PIL / ndarray / dict(image/mask) 统一为 ndarray；失败返 None。"""
    if x is None:
        return None
    if isinstance(x, dict):
        # gr.ImageEditor 的 layer 可能是 dict
        if "mask" in x and x["mask"] is not None:
            return _to_np(x["mask"])
        if "image" in x and x["image"] is not None:
            return _to_np(x["image"])
        return None
    if isinstance(x, Image.Image):
        return np.array(x)
    arr = np.asarray(x)
    return arr

def _sizes_from_meta(meta: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """
    从 meta 中稳健取出 (tw, th, W, H)。
    兼容:
      - {"thumb_w","thumb_h","ori_w","ori_h"}
      - {"thumb_size": (tw,th), "full_size": (W,H)}
    """
    tw = meta.get("thumb_w"); th = meta.get("thumb_h")
    W  = meta.get("ori_w");   H  = meta.get("ori_h")
    if tw is None or th is None:
        ts = meta.get("thumb_size")
        if ts and len(ts) >= 2:
            tw, th = int(ts[0]), int(ts[1])
    if W is None or H is None:
        fs = meta.get("full_size")
        if fs and len(fs) >= 2:
            W, H = int(fs[0]), int(fs[1])
    return int(tw or 0), int(th or 0), int(W or 0), int(H or 0)

# ---------- API：生成缩略图 ----------

def make_editor_thumbnail(img: Image.Image, long_side: int):
    """
    根据原图生成缩略图，返回 editor_value(PIL) 和 meta。
    meta 同时提供两套键，避免调用处 KeyError：
      - "full_size": (W, H), "thumb_size": (tw, th), "scale"
      - 兼容键: "ori_w","ori_h","thumb_w","thumb_h"
    """
    w, h = img.size
    if max(w, h) <= 0:
        thumb = img
        tw, th = w, h
        scale = 1.0
    else:
        scale = float(long_side) / float(max(w, h))
        if scale >= 1.0:
            thumb = img.copy()
            tw, th = w, h
            scale = 1.0
        else:
            tw = max(1, int(round(w * scale)))
            th = max(1, int(round(h * scale)))
            thumb = img.resize((tw, th), Image.LANCZOS)

    ev = {"background": thumb, "layers": [], "composite": thumb}
    meta = {
        "full_size": (w, h),
        "thumb_size": (tw, th),
        "scale": scale,
        # 兼容键
        "ori_w": w, "ori_h": h,
        "thumb_w": tw, "thumb_h": th,
    }
    return ev, meta

# ---------- API：从 ImageEditor 值中恢复全尺寸 ROI ----------

def editor_layers_to_mask_fullres(editor_value: Dict[str, Any], meta: Dict[str, Any], thr: int = 5) -> Optional[np.ndarray]:
    """
    从 ImageEditor 的编辑值提取 ROI（二值, 0/255）并映射回原图尺寸。
    - 兼容 PIL / numpy / dictLayer（含 image/mask）三种形式
    - 优先聚合 layers 的 alpha/mask；为空则用 composite vs background 的像素差异兜底
    """
    if not editor_value or not meta:
        return None

    tw, th, W, H = _sizes_from_meta(meta)
    if tw <= 0 or th <= 0 or W <= 0 or H <= 0:
        return None

    # ① 聚合图层
    acc = np.zeros((th, tw), dtype=np.uint8)
    layers = editor_value.get("layers") or []
    for L in layers:
        # 先尝试显式 mask
        m = _to_np(L["mask"]) if isinstance(L, dict) and "mask" in L else None
        if m is None:
            # 再尝试 image 的 alpha
            im = _to_np(L["image"] if isinstance(L, dict) else L)
            if im is not None:
                if im.ndim == 3 and im.shape[2] >= 4:
                    m = im[..., 3]  # alpha
                elif im.ndim == 3 and im.shape[2] >= 3:
                    m = cv2.cvtColor(im[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    m = (m > 0).astype(np.uint8) * 255
        if m is not None:
            if m.ndim == 3:
                m = m[..., 0]
            if m.shape[:2] != (th, tw):
                m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
            acc = np.maximum(acc, m.astype(np.uint8))

    # ② 图层全空 → 用合成差异兜底
    if int(acc.max()) == 0:
        bg   = _to_np(editor_value.get("background"))
        comp = _to_np(editor_value.get("composite"))
        if bg is not None and comp is not None:
            if bg.shape[:2] != (th, tw):
                bg = cv2.resize(bg, (tw, th), interpolation=cv2.INTER_NEAREST)
            if comp.shape[:2] != (th, tw):
                comp = cv2.resize(comp, (tw, th), interpolation=cv2.INTER_NEAREST)
            if bg.ndim == 2:
                bg = np.stack([bg] * 3, axis=2)
            if comp.ndim == 2:
                comp = np.stack([comp] * 3, axis=2)
            bg3 = bg[..., :3].astype(np.int16)
            cp3 = comp[..., :3].astype(np.int16)
            diff = np.abs(cp3 - bg3).mean(axis=2)
            acc = (diff > thr).astype(np.uint8) * 255

    if int(acc.max()) == 0:
        return None

    # ③ 映射回原图（最近邻，保持硬边）
    mask_full = cv2.resize(acc, (W, H), interpolation=cv2.INTER_NEAREST)
    return (mask_full > 0).astype(np.uint8) * 255


