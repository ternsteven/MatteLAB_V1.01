# -*- coding: utf-8 -*-
from __future__ import annotations
import os, numpy as np
from typing import Optional, Union
from PIL import Image, ImageOps, ImageFile



try:
    _env_max_px = os.getenv("MAX_IMAGE_PIXELS")
    Image.MAX_IMAGE_PIXELS = int(_env_max_px) if _env_max_px else None
except Exception:
    Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
__all__ = ["load_image_safe","preprocess_image","force_png_path","save_image_safe","_force_png_path","_save_image_safe"]
def load_image_safe(path: str) -> Image.Image:
    img = Image.open(path)
    try: img = ImageOps.exif_transpose(img)
    except Exception: pass
    return img
def _downscale_if_too_large(img: Image.Image, *, max_megapixels: float = None, max_long_side: int = None):
    try:
        if max_megapixels is None: max_megapixels = float(os.getenv("BIRE_MAX_MP", "48"))
        if max_long_side  is None: max_long_side  = int(os.getenv("BIRE_MAX_LONG", "12000"))
    except Exception:
        max_megapixels, max_long_side = 48.0, 12000
    w,h = img.size; scale=1.0; need=False
    mp=(w*h)/1e6
    if mp>max_megapixels: scale=min(scale,(max_megapixels/mp)**0.5); need=True
    long_side=max(w,h)
    if long_side>max_long_side: scale=min(scale,max_long_side/float(long_side)); need=True
    if need and scale<1.0:
        nw,nh=max(1,int(round(w*scale))), max(1,int(round(h*scale)))
        return img.resize((nw,nh), Image.LANCZOS)
    return img
def preprocess_image(image: Union[str, np.ndarray, Image.Image]) -> Optional[Image.Image]:
    try:
        if isinstance(image, str):
            if not os.path.exists(image): raise FileNotFoundError(image)
            img = load_image_safe(image)
        elif isinstance(image, np.ndarray):
            if image.ndim==3 and image.shape[2]>=4: image=image[...,:3]
            img = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        img = _downscale_if_too_large(img)
        if img.mode!="RGB": img = img.convert("RGB")
        return img
    except Exception:
        return None
def force_png_path(path: str) -> str:
    root,_=os.path.splitext(path); return root + ".png"
def save_image_safe(img: Union[Image.Image, np.ndarray], save_path: str) -> str:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    if img.mode in ("RGBA","LA") or ("transparency" in getattr(img,"info",{})):
        save_path = force_png_path(save_path); img.save(save_path,"PNG"); return save_path
    ext = os.path.splitext(save_path)[1].lower()
    if ext in (".jpg",".jpeg") and img.mode!="RGB": img = img.convert("RGB")
    try:
        img.save(save_path); return save_path
    except Exception:
        save_path = force_png_path(save_path); img.save(save_path,"PNG"); return save_path
_force_png_path = force_png_path
_save_image_safe = save_image_safe
