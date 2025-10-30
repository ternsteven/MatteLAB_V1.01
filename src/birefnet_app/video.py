# -*- coding: utf-8 -*-
from __future__ import annotations

import os, zipfile
from datetime import datetime
from typing import Tuple, Optional, Callable, Any, List

import numpy as np
from PIL import Image
try:
    import moviepy.editor as mp
except Exception as _e:
    mp = None  # 会在函数里给出友好提示

from .engine import BiRefEngine
from .settings import PRED_OUTPUT_DIR, ensure_dirs


def _parse_hex_color(s: Optional[str]) -> tuple[int, int, int]:
    """支持 #RRGGBB / #RGB / 'rgb(r,g,b)' / None（None → #00FF00）"""
    if not s:
        s = "#00FF00"
    s = s.strip()
    if s.startswith("rgb(") and s.endswith(")"):
        try:
            r, g, b = [int(x) for x in s[4:-1].split(",")[:3]]
            return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
        except Exception:
            s = "#00FF00"
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except Exception:
        return (0, 255, 0)  # 兜底绿色


def process_single_video(
    engine: BiRefEngine,
    input_video_path: str,
    *,
    # 即便外部传了 background_image，这里也会忽略，一律用纯色背景
    background_image: Optional[Any] = None,
    background_color: Optional[str] = None,
    input_size: Tuple[int, int] = (1024, 1024),
    semi_transparent: bool = False,
    semi_strength: float = 0.5,
    semi_mode: str = "auto",
    remove_white_halo: bool = False,
    defringe_strength: float = 0.7,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Optional[str], str]:
    """单个视频：强制纯色背景（不再允许透明导出）。"""
    ensure_dirs()
    if mp is None:
        return None, "请先安装 moviepy：pip install moviepy"

    try:
        clip = mp.VideoFileClip(input_video_path)
    except Exception as e:
        return None, f"无法打开视频：{e}"

    total_frames = max(1, int(round(clip.fps * clip.duration)))
    done = 0

    def _proc(frame: np.ndarray) -> np.ndarray:
        nonlocal done
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)

        # 转 PIL，取尺寸
        if frame.ndim == 3 and frame.shape[2] > 3:
            frame = frame[:, :, :3]
        frame_pil = Image.fromarray(frame)
        W, H = frame_pil.size

        # ✅ 强制构造纯色背景（忽略 background_image）
        bg_rgb = _parse_hex_color(background_color)  # None -> 绿色
        bg_img = Image.new("RGB", (W, H), bg_rgb)

        # 调引擎做抠图与合成

        W,H = (frame.shape[1], frame.shape[0]) if isinstance(frame, np.ndarray) else (frame.size[0], frame.size[1])
        _rgb = _parse_hex_color(background_color)
        _bgimg = _Image.new('RGB', (W,H), _rgb)
        result, _ = engine.apply_background_replacement(
            image=frame_pil,
            background_image=bg_img,             # 永远是纯色背景
            model_name=getattr(engine.cfg, "model_name", None),
            input_size=input_size,
            semi_transparent=bool(semi_transparent),
            semi_strength=float(semi_strength),
            semi_mode=str(semi_mode or "auto"),
            remove_white_halo=bool(remove_white_halo),
            defringe_strength=float(defringe_strength),
        )

        done += 1
        if progress_cb and done % 5 == 0:
            progress_cb(min(0.999, done / total_frames), f"已处理 {done}/{total_frames}")
        return np.array(result.convert("RGB"))

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
        clip.close(); out_clip.close()
        return None, f"写出视频失败：{e}"
    clip.close(); out_clip.close()

    if progress_cb:
        progress_cb(1.0, "完成")
    return out_path, "✅ 视频处理完成"


def process_batch_videos(
    engine: BiRefEngine,
    files: List[str],
    *,
    background_image: Optional[Any] = None,   # 兼容签名，但忽略
    background_color: Optional[str] = None,
    input_size: Tuple[int, int] = (1024, 1024),
    semi_transparent: bool = False,
    semi_strength: float = 0.5,
    semi_mode: str = "auto",
    remove_white_halo: bool = False,
    defringe_strength: float = 0.7,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Optional[str], str]:
    """批量视频：逐个调用上面单视频函数，最后打包 zip。"""
    ensure_dirs()
    files = files or []
    if not files:
        return None, "⚠️ 未选择任何视频文件"

    outputs: List[str] = []
    n = len(files)

    for idx, f in enumerate(files):
        def _cb(p: float, msg: str):
            if progress_cb:
                progress_cb((idx + p) / max(1, n), f"[{idx+1}/{n}] {msg}")

        out, status = process_single_video(
            engine,
            f,
            background_image=None,                 # 忽略图片背景
            background_color=background_color,     # ✅ 纯色背景透传
            input_size=input_size,
            semi_transparent=semi_transparent,
            semi_strength=semi_strength,
            semi_mode=semi_mode,
            remove_white_halo=remove_white_halo,
            defringe_strength=defringe_strength,
            progress_cb=_cb,
        )
        if out:
            outputs.append(out)

    if not outputs:
        return None, "❌ 批量视频处理失败：没有成功输出"

    if len(outputs) == 1:
        return outputs[0], "✅ 批量视频处理完成（1 个文件）"

    zip_name = os.path.join(PRED_OUTPUT_DIR, f"videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in outputs:
            z.write(p, os.path.basename(p))
    return zip_name, f"✅ 批量视频处理完成（{len(outputs)} 个文件）"