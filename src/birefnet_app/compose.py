
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
    input_size: Optional[Tuple[int, int]] = None,
    # === ROI ===
    roi_mask_fullres: Optional[np.ndarray] = None,  # HxW 0/255
    roi_crop_before: bool = True,
    roi_pad_px: int = 16,
):
    # --- 0) 规范输入图 ---
    img_arr = np.array(image) if isinstance(image, Image.Image) else np.asarray(image)
    if img_arr.ndim == 2:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    if img_arr.ndim == 3 and img_arr.shape[2] >= 4:
        img_arr = img_arr[:, :, :3]
    H, W = img_arr.shape[:2]

    # --- 1) ROI 预裁剪（可选） ---
    crop_box = None
    im_for_seg = img_arr
    if roi_mask_fullres is not None and roi_crop_before:
        roi_bin = (roi_mask_fullres > 0).astype(np.uint8)
        bbox = _bbox_from_mask(roi_bin)
        if bbox is not None:
            x0, y0, x1, y1 = _expand_box(*bbox, pad=int(roi_pad_px), W=W, H=H)
            crop_box = (x0, y0, x1, y1)
            im_for_seg = img_arr[y0:y1, x0:x1]

    # --- 2) 分割 ---
    raw_mask = engine.segment(
        im_for_seg,
        model_name=model_name or getattr(engine.cfg, "model_name", None),
        input_size=input_size or getattr(engine.cfg, "input_size", None),
    )
    if raw_mask is None:
        raise RuntimeError("无法生成分割 mask")

    # 统一成单通道 uint8（0/255）
    if raw_mask.ndim == 3:
        if raw_mask.shape[2] == 1:
            raw_mask = raw_mask[..., 0]
        else:
            raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_RGB2GRAY)
    raw_mask = np.asarray(raw_mask)
    if raw_mask.dtype != np.uint8:
        # 常见返回：bool/0-1/float
        m = raw_mask.astype(np.float32)
        m = np.clip(m, 0, 1) if m.max() <= 1.0 else np.clip(m, 0, 255) / 255.0
        raw_mask = (m * 255.0 + 0.5).astype(np.uint8)

    # --- 3) 回贴到全图（若有裁剪） ---
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        m_full = np.zeros((H, W), np.uint8)
        m_crop = cv2.resize(raw_mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        m_full[y0:y1, x0:x1] = m_crop
        raw_mask = m_full

    # --- 4) ROI 限域（即使未裁剪，也可只保留 ROI 内前景） ---
    if roi_mask_fullres is not None:
        roi_bin = (roi_mask_fullres > 0).astype(np.uint8)
        if roi_bin.shape != (H, W):
            roi_bin = cv2.resize(roi_bin, (W, H), interpolation=cv2.INTER_NEAREST)
        raw_mask = (raw_mask.astype(np.float32) * (roi_bin > 0)).astype(np.uint8)

        # 兜底1：启用 ROI 但掩码全 0 → 回退到全图分割（不加 ROI）
        if int(raw_mask.max()) == 0:
            full_mask = engine.segment(
                img_arr,
                model_name=model_name or getattr(engine.cfg, "model_name", None),
                input_size=input_size or getattr(engine.cfg, "input_size", None),
            )
            if full_mask is None:
                raise RuntimeError("ROI 为空且全图分割失败")
            if full_mask.ndim == 3:
                if full_mask.shape[2] == 1:
                    full_mask = full_mask[..., 0]
                else:
                    full_mask = cv2.cvtColor(full_mask, cv2.COLOR_RGB2GRAY)
            full_mask = np.asarray(full_mask)
            if full_mask.dtype != np.uint8:
                fm = full_mask.astype(np.float32)
                fm = np.clip(fm, 0, 1) if fm.max() <= 1.0 else np.clip(fm, 0, 255) / 255.0
                full_mask = (fm * 255.0 + 0.5).astype(np.uint8)
            raw_mask = full_mask

    # --- 5) 半透明 vs 二值（带兜底） ---
    # 二值硬边（Otsu/阈值）——软 α 输入
    hard = to_binary_mask(raw_mask, use_otsu=True)

    if semi_transparent:
        # 软 α 估计
        soft_alpha = estimate_soft_alpha_inside_mask(
            img_arr, hard, strength=float(semi_strength), mode=str(semi_mode or "auto")
        )

        # 兜底2：软 α 为空或几乎全空 → 回退二值
        if soft_alpha is None or float(np.max(soft_alpha)) < 1.0:
            mask_u8 = hard
        else:
            # refine + 规范化
            mask_refined = refine_alpha_with_channel(
                img_arr, soft_alpha, mode=str(semi_mode or "auto"), strength=float(semi_strength)
            )
            mask_refined = np.asarray(mask_refined)
            if mask_refined.dtype != np.uint8:
                mr = mask_refined.astype(np.float32)
                mr = np.clip(mr, 0, 1) if mr.max() <= 1.0 else np.clip(mr, 0, 255) / 255.0
                mask_u8 = (mr * 255.0 + 0.5).astype(np.uint8)
            else:
                mask_u8 = mask_refined

            # 兜底3：refine 结果仍全 0 → 回退二值
            if int(mask_u8.max()) == 0:
                mask_u8 = hard
    else:
        mask_u8 = hard

    # --- 6) 合成/透明导出 ---
    bg_img = bg.create_background(
        "image" if background_image is not None else "transparent",
        background_image,
        (W, H),
    )

    if bg_img is not None:
        out = bg.replace_background_with_mask(
            img_arr, bg_img, mask_u8,
            remove_white_halo=bool(remove_white_halo),
            defringe_strength=float(defringe_strength),
        )
    else:
        out = bg.create_transparent_result(
            img_arr, mask_u8,
            remove_white_halo=bool(remove_white_halo),
            defringe_strength=float(defringe_strength),
        )

    return out, Image.fromarray(mask_u8).convert("RGB")


