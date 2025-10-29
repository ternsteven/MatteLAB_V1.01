# -*- coding: utf-8 -*-
"""
ops/image_io.py
公开稳定的 I/O 工具：
- load_image_safe(path)        安全读取图片（带 EXIF 旋转）
- preprocess_image(image)      统一预处理（含超大图等比缩放与 RGB 规范化）
- force_png_path(path)         把任意扩展名改为 .png
- save_image_safe(img, path)   安全保存：带透明→强制 PNG；JPG 非 RGB→转 RGB
"""

from __future__ import annotations
import os
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps, ImageFile

# ---- 超大图/截断图容错 ----
# 可通过环境变量覆盖：MAX_IMAGE_PIXELS, BIRE_MAX_MP, BIRE_MAX_LONG
try:
    _env_max_px = os.getenv("MAX_IMAGE_PIXELS")
    Image.MAX_IMAGE_PIXELS = int(_env_max_px) if _env_max_px else None  # None = 取消限制
except Exception:
    Image.MAX_IMAGE_PIXELS = None

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许载入被截断的图片

# ---- 公共 API：建议保持这些名字不变 ----
__all__ = [
    "load_image_safe",
    "preprocess_image",
    "force_png_path",
    "save_image_safe",
    # 兼容旧代码的内部别名（有文件 import 了下划线版本）
    "_force_png_path",
    "_save_image_safe",
]


# ========== 读取 ==========
def load_image_safe(path: str) -> Image.Image:
    """
    安全读取一张图片，自动处理 EXIF 旋转；支持大图/截断图。
    """
    img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img


# ========== 预处理 ==========
def _downscale_if_too_large(
    img: Image.Image,
    *,
    max_megapixels: Optional[float] = None,
    max_long_side: Optional[int] = None,
) -> Image.Image:
    """
    对超大图做等比缩小，避免内存暴涨或 PIL 抛炸弹。
    默认阈值：48MP / 长边 12000px（可用环境变量覆盖）
    """
    try:
        if max_megapixels is None:
            max_megapixels = float(os.getenv("BIRE_MAX_MP", "48"))
        if max_long_side is None:
            max_long_side = int(os.getenv("BIRE_MAX_LONG", "12000"))
    except Exception:
        max_megapixels, max_long_side = 48.0, 12000

    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    need_scale = False
    scale = 1.0

    # 限制像素总量
    mp = (w * h) / 1_000_000.0
    if mp > max_megapixels:
        scale = min(scale, (max_megapixels / mp) ** 0.5)
        need_scale = True

    # 限制长边
    long_side = max(w, h)
    if long_side > max_long_side:
        scale = min(scale, max_long_side / float(long_side))
        need_scale = True

    if need_scale and scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.LANCZOS)
    return img


def preprocess_image(image: Union[str, np.ndarray, Image.Image]) -> Optional[Image.Image]:
    """
    统一预处理输入：
    - 路径 / numpy / PIL 均可；
    - 自动等比缩小超大图；
    - 转换为 RGB。
    """
    try:
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"图像文件不存在: {image}")
            img = load_image_safe(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] >= 4:
                image = image[:, :, :3]
            img = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")

        img = _downscale_if_too_large(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None


# ========== 保存 ==========
def force_png_path(path: str) -> str:
    """把任意路径的扩展名改成 .png"""
    root, _ = os.path.splitext(path)
    return root + ".png"


def save_image_safe(img: Union[Image.Image, np.ndarray], save_path: str) -> str:
    """
    安全保存图像：
    - RGBA/LA 或带 transparency → 强制 PNG；
    - 目标是 jpg/jpeg，但图像不是 RGB → 转 RGB；
    - 如仍报错则回退为 PNG。
    返回最终保存的路径（可能被改为 .png）。
    """
    # numpy → PIL
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))

    # 有透明信息 → 强制改为 PNG
    if getattr(img, "mode", "") in ("RGBA", "LA") or ("transparency" in getattr(img, "info", {})):
        save_path = force_png_path(save_path)
        img.save(save_path, "PNG")
        return save_path

    ext = os.path.splitext(save_path)[1].lower()
    if ext in (".jpg", ".jpeg") and img.mode != "RGB":
        img = img.convert("RGB")

    try:
        img.save(save_path)
        return save_path
    except Exception:
        save_path = force_png_path(save_path)
        img.save(save_path, "PNG")
        return save_path


# —— 兼容旧代码的内部别名（有人用 `_force_png_path` / `_save_image_safe`）——
_force_png_path = force_png_path
_save_image_safe = save_image_safe
