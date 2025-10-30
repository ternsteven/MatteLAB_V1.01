# -*- coding: utf-8 -*-
import os, inspect, traceback
from typing import Any, List, Optional, Tuple
from ..logging_utils import get_logger
from PIL import Image

logger = get_logger("MatteLAB.Video")

# 尝试导入“核心”视频处理方法（按你的工程结构）
try:
    from ..video import process_single_video as _core_process_single_video
except Exception:
    _core_process_single_video = None

try:
    from ..video import process_batch_videos as _core_process_batch_videos
except Exception:
    _core_process_batch_videos = None

def _parse_hex_color(s: Optional[str]) -> Tuple[int, int, int]:
    """支持 #RRGGBB / #RGB / None（默认为绿色）"""
    if not s:
        s = "#00FF00"
    s = s.strip()
    if s.startswith("rgb(") and s.endswith(")"):
        # 兼容 'rgb(0,255,0)' 形式
        nums = s[4:-1].split(",")
        try:
            r, g, b = [int(x) for x in nums[:3]]
            return max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b))
        except Exception:
            s = "#00FF00"
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join([c*2 for c in s])
    try:
        return int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)
    except Exception:
        return (0,255,0)  # 兜底：绿色


def parse_model_choice(selected: str) -> str:
    if not selected:
        return "General"
    return str(selected).split(" - ")[0].strip()


def _normalize_path(f: Any) -> Optional[str]:
    """
    将 Gradio 的单个文件输入规范成可用路径。
    """
    if not f:
        return None
    if isinstance(f, str):
        return f if os.path.exists(f) else None
    p = getattr(f, "name", None) or getattr(f, "orig_name", None) or getattr(f, "path", None)
    return p if (p and os.path.exists(p)) else None


def _normalize_files(files: Any) -> List[str]:
    """
    将 Gradio 的多文件输入规范成路径列表。
    """
    out: List[str] = []
    if not files:
        return out
    seq = files if isinstance(files, (list, tuple)) else [files]
    for f in seq:
        p = _normalize_path(f)
        if p:
            out.append(p)
    return out


def run_single_video(
    engine,
    video_file,
    bg_image,                 # Optional[PIL.Image.Image]
    bg_color,                 # e.g. "#00FF00"
    semi_enable: bool,
    semi_strength: float,
    semi_mode: str,
    selected_model: str,
    res_value: int,
):
    """
    单个视频适配层：把 UI 的参数标准化后传给核心视频处理。
    - 半透明三参：semi_transparent / semi_strength / semi_mode
    - 背景图优先；无背景图时可用背景色（如果核心函数支持 background_color）
    """
    if _core_process_single_video is None:
        return None, "❌ 未找到实际的视频处理函数：video.process_single_video"

    path = _normalize_path(video_file)
    if not path:
        return None, "⚠️ 请先选择一个视频文件"

    try:
        short = parse_model_choice(selected_model)
        res = int(res_value)
        engine.load_model(short, (res, res))

        # 候选参数（后续按核心函数签名筛选）
        candidate_kwargs = {
            "background_image": bg_image,
            "background_color": (None if bg_image else str(bg_color or "#00FF00")),
            "input_size": (res, res),
            "model_name": short,
            "semi_transparent": bool(semi_enable),
            "semi_strength": float(semi_strength or 0.5),
            "semi_mode": str(semi_mode or "auto"),
        }

        sig = inspect.signature(_core_process_single_video)
        param_names = list(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}

        # 两类常见签名：
        #   A) process_single_video(engine, video_path, **kwargs)
        #   B) process_single_video(video_path, background_image=..., background_color=..., ...)
        if param_names and param_names[0] in ("engine", "eng"):
            out = _core_process_single_video(engine, path, **filtered_kwargs)
        else:
            out = _core_process_single_video(path, **filtered_kwargs)

        # 统一返回给 UI：优先 (out_path, status)；支持 (out_path, info, status) 或纯路径
        if isinstance(out, tuple):
            if len(out) == 2:
                return out
            if len(out) == 3:
                return out[0], out[2]
        return out, f"✅ 处理完成（模型：{short}，输入：{res}×{res}）"

    except Exception as e:
        tb = traceback.format_exc(limit=2)
        logger.error(f"单视频处理失败：{e}\n{tb}")
        return None, f"❌ 单视频处理失败：{e}\n```\n{tb}\n```"


def run_batch_videos(
    engine,
    files,
    bg_image,                 # Optional[PIL.Image.Image]
    bg_color,                 # e.g. "#00FF00"
    semi_enable: bool,
    semi_strength: float,
    semi_mode: str,
    selected_model: str,
    res_value: int,
):
    """
    批量视频适配层：把 UI 的参数标准化后传给核心批量视频处理。
    - 半透明三参同上
    - 同时兼容旧签名：process_batch_videos(files, background_image) / process_batch_videos(files, background_color)
    """
    if _core_process_batch_videos is None:
        return None, "❌ 未找到实际的批量视频处理函数：video.process_batch_videos"

    paths = _normalize_files(files)
    if not paths:
        return None, "⚠️ 请先选择至少一个视频文件"

    try:
        short = parse_model_choice(selected_model)
        res = int(res_value)
        engine.load_model(short, (res, res))

        candidate_kwargs = {
            "background_image": bg_image,
            "background_color": (None if bg_image else str(bg_color or "#00FF00")),
            "input_size": (res, res),
            "model_name": short,
            "semi_transparent": bool(semi_enable),
            "semi_strength": float(semi_strength or 0.5),
            "semi_mode": str(semi_mode or "auto"),
        }

        sig = inspect.signature(_core_process_batch_videos)
        param_names = list(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}

        # 常见签名：
        #   A) process_batch_videos(engine, files, **kwargs)
        #   B) process_batch_videos(files, background_image=..., background_color=..., ...)
        #   C) 旧版两参：(files, background_image) 或 (files, background_color)
        if param_names and param_names[0] in ("engine", "eng"):
            out = _core_process_batch_videos(engine, paths, **filtered_kwargs)
        else:
            if param_names[:2] == ["files", "background_image"] and len(sig.parameters) == 2:
                out = _core_process_batch_videos(paths, candidate_kwargs["background_image"])
            elif param_names[:2] == ["files", "background_color"] and len(sig.parameters) == 2:
                out = _core_process_batch_videos(paths, candidate_kwargs["background_color"])
            else:
                out = _core_process_batch_videos(paths, **filtered_kwargs)

        # 统一返回给 UI：优先 (zip_path, status)；支持 (zip_path, previews, status) 或纯路径
        if isinstance(out, tuple):
            if len(out) == 2:
                return out
            if len(out) == 3:
                return out[0], out[2]
        return out, f"✅ 批量处理完成（模型：{short}，输入：{res}×{res}）"

    except Exception as e:
        tb = traceback.format_exc(limit=2)
        logger.error(f"批量视频处理失败：{e}\n{tb}")
        return None, f"❌ 批量视频处理失败：{e}\n```\n{tb}\n```"
