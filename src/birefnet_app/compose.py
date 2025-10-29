from typing import Optional, Tuple, Any
import numpy as np
from PIL import Image
import cv2

from .ops.roi_ops import editor_layers_to_mask_fullres  # 如 UI 已产出全尺寸，可直接透传
from .ops.mask_ops import to_binary_mask, estimate_soft_alpha_inside_mask, refine_alpha_with_channel
from .ops import bg_ops as bg

def _bbox_from_mask(mask_u8: np.ndarray):
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0 or ys.size == 0: return None
    x0, x1 = xs.min(), xs.max() + 1; y0, y1 = ys.min(), ys.max() + 1
    return int(x0), int(y0), int(x1), int(y1)

def _expand_box(x0, y0, x1, y1, pad: int, W: int, H: int):
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    return x0, y0, x1, y1

def apply_background_replacement(
    engine,  # BiRefEngine
    image: Any,
    *,
    background_image: Optional[Image.Image] = None,
    semi_transparent: bool = False,
    semi_strength: float = 0.5,
    semi_mode: str = "auto",
    remove_white_halo: bool = False,
    defringe_strength: float = 0.7,
    model_name: Optional[str] = None,
    input_size: Optional[Tuple[int,int]] = None,
    # === ROI 新增 ===
    roi_mask_fullres: Optional[np.ndarray] = None,  # HxW 0/255
    roi_crop_before: bool = True,
    roi_pad_px: int = 16,
):
    img_arr = np.array(image) if isinstance(image, Image.Image) else np.asarray(image)
    if img_arr.ndim == 2: img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    if img_arr.ndim == 3 and img_arr.shape[2] >= 4: img_arr = img_arr[:, :, :3]
    H, W = img_arr.shape[:2]

    # 1) ROI 裁剪（可选）
    crop_box = None
    if roi_mask_fullres is not None and roi_crop_before:
        roi_bin = (roi_mask_fullres > 0).astype(np.uint8)
        bbox = _bbox_from_mask(roi_bin)
        if bbox is not None:
            x0, y0, x1, y1 = _expand_box(*bbox, pad=int(roi_pad_px), W=W, H=H)
            crop_box = (x0, y0, x1, y1)
            im_for_seg = img_arr[y0:y1, x0:x1]
        else:
            im_for_seg = img_arr
    else:
        im_for_seg = img_arr

    # 2) 分割
    raw_mask = engine.segment(
        im_for_seg,
        model_name=model_name or engine.cfg.model_name,
        input_size=input_size or engine.cfg.input_size,
    )
    if raw_mask is None:
        raise RuntimeError("无法生成分割mask")

    # 3) 回贴到全图
    if crop_box is not None:
        (x0, y0, x1, y1) = crop_box
        m_full = np.zeros((H, W), np.uint8)
        m_crop = raw_mask
        if m_crop.ndim == 3:  # 安全转灰度
            m_crop = cv2.cvtColor(m_crop, cv2.COLOR_RGB2GRAY) if m_crop.shape[2] != 1 else m_crop[...,0]
        m_crop = cv2.resize(m_crop, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
        m_full[y0:y1, x0:x1] = m_crop
        raw_mask = m_full

    # 4) ROI 限域（即使未裁剪，仍可强制仅保留 ROI 内前景）
    if roi_mask_fullres is not None:
        roi_bin = (roi_mask_fullres > 0).astype(np.uint8)
        if roi_bin.shape != (H, W):
            roi_bin = cv2.resize(roi_bin, (W, H), interpolation=cv2.INTER_NEAREST)
        raw_mask = (raw_mask.astype(np.float32) * (roi_bin > 0)).astype(np.uint8)

    # 5) 半透明 vs 二值
    if semi_transparent:
        hard = to_binary_mask(raw_mask, use_otsu=True)
        soft_alpha = estimate_soft_alpha_inside_mask(
            img_arr, hard, strength=float(semi_strength), mode=semi_mode
        )
        mask_u8 = refine_alpha_with_channel(
            img_arr, soft_alpha, mode=semi_mode, strength=float(semi_strength)
        )
    else:
        mask_u8 = to_binary_mask(raw_mask, use_otsu=True)

    # 6) 合成/透明导出
    bg_img = bg.create_background('image' if background_image is not None else 'transparent',
                                  background_image, (W, H))

    if bg_img is not None:
        out = bg.replace_background_with_mask(
            img_arr, bg_img, mask_u8,
            remove_white_halo=remove_white_halo,
            defringe_strength=float(defringe_strength),
        )
    else:
        out = bg.create_transparent_result(
            img_arr, mask_u8,
            remove_white_halo=remove_white_halo,
            defringe_strength=float(defringe_strength),
        )

    return out, Image.fromarray(mask_u8).convert("RGB")
