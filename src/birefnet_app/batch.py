from __future__ import annotations

import os, io, zipfile
from datetime import datetime
from typing import Iterable, Tuple, List, Callable, Optional, Any
from PIL import Image, UnidentifiedImageError

from .engine import BiRefEngine
from .settings import PRED_OUTPUT_DIR, ensure_dirs
from .ops.image_io import save_image_safe

ensure_dirs()

def _open_as_pil(x: Any) -> Image.Image:
    if isinstance(x, Image.Image): return x
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if hasattr(x, "name"):
        return Image.open(x.name).convert("RGB")
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    raise UnidentifiedImageError(f"不支持的文件类型: {type(x)}")

def process_batch_images(
    engine: BiRefEngine,
    files: Iterable[Any],
    *,
    background_image: Optional[Image.Image] = None,
    input_size: Tuple[int,int] = (1024,1024),
    semi_transparent: bool = False,
    semi_strength: float = 0.5,
    semi_mode: str = "auto",
    remove_white_halo: bool = False,
    defringe_strength: float = 0.7,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Optional[str], List[str], str]:
    ensure_dirs()
    os.makedirs(PRED_OUTPUT_DIR, exist_ok=True)

    saved: List[str] = []
    files = list(files)
    total = len(files)
    if total == 0:
        return None, [], "⚠️ 未选择任何文件"

    for i, f in enumerate(files, 1):
        try:
            img = _open_as_pil(f)
            result, _ = engine.apply_background_replacement(
                image=img,
                background_image=background_image,
                model_name=engine.cfg.model_name,
                input_size=input_size,
                semi_transparent=semi_transparent,
                semi_strength=semi_strength,
                semi_mode=semi_mode,
                remove_white_halo=remove_white_halo,
                defringe_strength=defringe_strength,
            )
            base = os.path.splitext(getattr(f, "name", (f if isinstance(f, str) else f"img_{i}.png")))[0]
            out_path = os.path.join(PRED_OUTPUT_DIR, f"{os.path.basename(base)}_out.png")
            out_path = save_image_safe(result, out_path)
            saved.append(out_path)
        except Exception as e:
            saved.append(f"[ERROR]{e}")

        if progress_cb:
            progress_cb(i/total, f"{i}/{total}")

    ok_count = sum(1 for p in saved if isinstance(p, str) and not p.startswith("[ERROR]"))
    if ok_count > 0:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = os.path.join(PRED_OUTPUT_DIR, f"batch_{stamp}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in saved:
                if isinstance(p, str) and not p.startswith("[ERROR]"):
                    z.write(p, arcname=os.path.basename(p))
        return zip_path, saved, f"✅ 批量完成：{ok_count}/{total}"
    else:
        return None, saved, "❌ 批量失败：未产生结果"

