# -*- coding: utf-8 -*-
import traceback
from typing import Optional, Any
from PIL import Image
from ..logging_utils import get_logger
from ..ops.roi_ops import editor_layers_to_mask_fullres

logger = get_logger("MatteLAB.Single")


def parse_model_choice(selected: str) -> str:
    """把 'General - xxx' 这种下拉值取前半段短名。"""
    if not selected:
        return "General"
    return str(selected).split(" - ")[0].strip()


def run_single_image(
    engine,
    img: Image.Image,
    bg_img: Optional[Image.Image],
    semi_en: bool, semi_str: float, semi_md: str,
    def_en: bool, def_str: float,
    roi_en: bool, roi_ev: dict, roi_meta: dict, roi_crop: bool, roi_pad: int,
    selected_model: str, res_value: int
):
    """单图处理入口：负责把 UI 的参数整理后传给引擎/合成函数。"""
    if img is None:
        return (
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)),
            Image.new("RGB", (1, 1), (0, 0, 0)),
            "⚠️ 请先上传图片",
        )

    try:
        short = parse_model_choice(selected_model)
        res = int(res_value)
        engine.load_model(short, (res, res))

        # ---- ROI 解析（有则用；失败则回退全图） ----
        roi_mask_full = None
        if bool(roi_en) and roi_ev and roi_meta:
            try:
                roi_mask_full = editor_layers_to_mask_fullres(roi_ev, roi_meta)
                if roi_mask_full is None:
                    logger.warning("ROI 解析为空，按全图处理")
            except Exception as e:
                logger.warning(f"ROI 解析失败，将按全图处理：{e}")

        # ---- 调用后端主流程（半透明三参原样透传） ----
        result, mask = engine.apply_background_replacement(
            image=img,
            background_image=bg_img,
            model_name=short,
            input_size=(res, res),
            semi_transparent=bool(semi_en),
            semi_strength=float(semi_str or 0.5),
            semi_mode=str(semi_md or "auto"),
            remove_white_halo=bool(def_en),
            defringe_strength=float(def_str or 0.7),
            roi_mask_fullres=roi_mask_full,
            roi_crop_before=bool(roi_crop),
            roi_pad_px=int(roi_pad or 0),
        )

        # ---- 输出规范化 ----
        if result is None:
            result = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        elif result.mode != "RGBA":
            result = result.convert("RGBA")

        if mask is None:
            mask = Image.new("RGB", (1, 1), (0, 0, 0))
        elif mask.mode != "RGB":
            mask = mask.convert("RGB")

        w, h = result.size
        return result, mask, f"✅ 完成（模型：{short}，输入：{res}×{res}，输出：{w}×{h}）"

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"单图处理失败：{e}\n{tb}")
        return (
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)),
            Image.new("RGB", (1, 1), (0, 0, 0)),
            f"❌ 处理失败：{e}\n```\n{tb}\n```",
        )
