from __future__ import annotations
import os
from datetime import datetime
from typing import Tuple, Optional, Callable, Any
import numpy as np

from .engine import BiRefEngine
from .settings import PRED_OUTPUT_DIR, ensure_dirs

def process_single_video(
    engine: BiRefEngine,
    input_video_path: str,
    *,
    background_image: Optional[Any] = None,
    input_size: Tuple[int,int] = (1024,1024),
    semi_transparent: bool = False,
    semi_strength: float = 0.5,
    semi_mode: str = "auto",
    remove_white_halo: bool = False,
    defringe_strength: float = 0.7,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Optional[str], str]:
    ensure_dirs()
    try:
        import moviepy.editor as mp
    except Exception as e:
        return None, f"请先安装 moviepy：{e}"

    clip = mp.VideoFileClip(input_video_path)
    total_frames = max(1, int(clip.fps * clip.duration))
    done = 0

    def _proc(frame: np.ndarray) -> np.ndarray:
        nonlocal done
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        result, _ = engine.apply_background_replacement(
            image=frame,
            background_image=background_image,
            model_name=engine.cfg.model_name,
            input_size=input_size,
            semi_transparent=semi_transparent,
            semi_strength=semi_strength,
            semi_mode=semi_mode,
            remove_white_halo=remove_white_halo,
            defringe_strength=defringe_strength,
        )
        done += 1
        if progress_cb and done % 5 == 0:
            progress_cb(min(0.999, done/total_frames), f"已处理 {done}/{total_frames}")
        return np.array(result)

    out_clip = clip.fl_image(_proc)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(input_video_path))[0]
    out_path = os.path.join(PRED_OUTPUT_DIR, f"{base}_out_{stamp}.mp4")
    try:
        out_clip.write_videofile(
            out_path,
            codec="libx264",
            audio=clip.audio is not None,
            audio_codec="aac" if clip.audio is not None else None,
            temp_audiofile=os.path.join(PRED_OUTPUT_DIR, f"temp_{stamp}.m4a") if clip.audio is not None else None,
            remove_temp=True,
            verbose=False,
            logger=None,
        )
    except Exception as e:
        return None, f"写出视频失败：{e}"
    finally:
        clip.close(); out_clip.close()

    if progress_cb:
        progress_cb(1.0, "完成")
    return out_path, "✅ 视频处理完成"
