from PIL import Image, ImageOps
import os
import numpy as np

def load_image_safe(path):
    im = Image.open(path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im

def force_png_path(path: str) -> str:
    root, _ = os.path.splitext(path)
    return root + ".png"

def save_image_safe(img, save_path: str):
    # 透明 → 强制 PNG
    if getattr(img, "mode", "") in ("RGBA", "LA") or ("transparency" in getattr(img, "info", {})):
        save_path = force_png_path(save_path)
        img.save(save_path, "PNG"); return save_path
    ext = os.path.splitext(save_path)[1].lower()
    if ext in (".jpg", ".jpeg") and getattr(img, "mode", "") != "RGB":
        img = img.convert("RGB")
    try:
        img.save(save_path); return save_path
    except Exception:
        save_path = force_png_path(save_path)
        img.save(save_path, "PNG"); return save_path

def preprocess_image(image):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def resize_keep_ratio(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    s = target / max(w, h)
    return img.resize((int(w*s), int(h*s)), Image.BILINEAR)
