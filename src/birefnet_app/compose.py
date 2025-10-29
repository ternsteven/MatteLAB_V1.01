# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Any
import numpy as np
from PIL import Image

from .ops.mask_ops import (
    to_binary_mask,
    estimate_soft_alpha_inside_mask,
    refine_alpha_with_channel,
)
from .ops import bg_ops as bg

def apply_background_replacement(
    engine,  # BiRefEngine 实例，仅用于调用 engine.segment()
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
):
    img_arr = np.array(image) if isinstance(image, Image.Image) else np.asarray(image)
    raw_mask = engine.segment(
        img_arr,
        model_name=model_name or engine.cfg.model_name,
        input_size=input_size or engine.cfg.input_size,
    )
    if raw_mask is None:
        raise RuntimeError("无法生成分割mask")

    if semi_transparent:
        hard = ((raw_mask.astype(np.uint8)) > 127).astype(np.uint8) * 255
        soft_alpha = estimate_soft_alpha_inside_mask(
            img_arr, hard, strength=float(semi_strength), mode=semi_mode
        )
        mask_u8 = refine_alpha_with_channel(
            img_arr, soft_alpha, mode=semi_mode, strength=float(semi_strength)
        )
    else:
        mask_u8 = to_binary_mask(raw_mask, use_otsu=True)

    bg_img = bg.create_background(
        'image' if background_image is not None else 'transparent',
        background_image,
        (img_arr.shape[1], img_arr.shape[0]),
    )

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
    # 返回：结果图（RGB/RGBA）和可视化mask（RGB）
    return out, Image.fromarray(mask_u8).convert("RGB")
