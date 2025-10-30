# -*- coding: utf-8 -*-
import os, time, numpy as np
from PIL import Image
from .image_io import save_image_safe
from ..settings import OUT_DIR, MASK_SUBDIR, ensure_dirs
def _to_pil_rgba(x):
    if isinstance(x, Image.Image): return x if x.mode=="RGBA" else x.convert("RGBA")
    arr=np.asarray(x)
    if arr.ndim==2: return Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    if arr.ndim==3:
        im=Image.fromarray(arr.astype(np.uint8))
        return im if im.mode=="RGBA" else im.convert("RGBA")
    return Image.new("RGBA",(1,1),(0,0,0,0))
def _to_pil_rgb(x):
    if isinstance(x, Image.Image): return x if x.mode=="RGB" else x.convert("RGB")
    arr=np.asarray(x)
    if arr.ndim==2: return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if arr.ndim==3:
        im=Image.fromarray(arr.astype(np.uint8))
        return im if im.mode=="RGB" else im.convert("RGB")
    return Image.new("RGB",(1,1),(0,0,0))
def save_result_and_mask(result_img, mask_img):
    ensure_dirs()
    result_img=_to_pil_rgba(result_img)
    mask_img=_to_pil_rgb(mask_img)
    ts=time.strftime("%Y%m%d-%H%M%S")
    out_dir=OUT_DIR; mask_dir=os.path.join(OUT_DIR,MASK_SUBDIR)
    os.makedirs(out_dir,exist_ok=True); os.makedirs(mask_dir,exist_ok=True)
    out_path=os.path.join(out_dir,f"cutout_{ts}.png")
    mask_path=os.path.join(mask_dir,f"mask_{ts}.png")
    try:
        save_image_safe(result_img,out_path)
        save_image_safe(mask_img,mask_path)
        md=f"💾 已保存：`{os.path.basename(out_path)}`；蒙版：`{os.path.basename(mask_path)}`\n📂 目录：`{os.path.abspath(out_dir)}`"
    except Exception as e:
        md=f"⚠️ 保存失败：{e}"
    return mask_img, md

