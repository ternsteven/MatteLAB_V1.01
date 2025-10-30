# -*- coding: utf-8 -*-
import os, inspect, traceback
from typing import List, Optional, Tuple, Any
from ..logging_utils import get_logger

logger = get_logger("MatteLAB.Batch")

try:
    from ..batch import process_batch_images as _core_process_batch_images
except Exception:
    _core_process_batch_images = None


def parse_model_choice(selected: str) -> str:
    if not selected:
        return "General"
    return str(selected).split(" - ")[0].strip()


def _normalize_files(files: Any) -> List[str]:
    """
    将 Gradio 传入的 files（可能是 list[str] / list[File] / None）规范成路径列表。
    """
    paths: List[str] = []
    if not files:
        return paths

    # gr.File(..., file_count="multiple") 场景：列表里可能是 str 或对象
    for f in (files if isinstance(files, (list, tuple)) else [files]):
        # gradio File 对象可能有 .name 或 .orig_name、.path
        p = None
        if isinstance(f, str):
            p = f
        else:
            # 尝试常见属性
            p = getattr(f, "name", None) or getattr(f, "orig_name", None) or getattr(f, "path", None)
        if p and os.path.exists(p):
            paths.append(p)
    return paths


def run_batch_images(
    engine,
    files,
    bg_image,                       # Optional[PIL.Image.Image]
    semi_enable: bool,
    semi_strength: float,
    semi_mode: str,
    defringe_enable: bool,
    defringe_strength: float,
    selected_model: str,
    res_value: int
):
    """
    批量图片适配层：只负责把 UI 的参数“标准化 + 透传”到核心 batch 逻辑。
    半透明三参（semi_*）与去白边参数（defringe_*）均按签名探测后传入。
    """
    if _core_process_batch_images is None:
        return None, "❌ 未找到实际的批量图片处理函数：batch.process_batch_images"

    # 规范输入文件
    paths = _normalize_files(files)
    if not paths:
        return None, "⚠️ 请先选择至少一张图片文件"

    try:
        # 模型&分辨率
        short = parse_model_choice(selected_model)
        res = int(res_value)
        engine.load_model(short, (res, res))

        # 构造“可能”被接收的参数（后面按签名过滤）
        candidate_kwargs = {
            "background_image": bg_image,                  # 背景图，不上传时通常 None
            "input_size": (res, res),                      # 输入分辨率
            "model_name": short,                           # 可选：很多 batch 实现不需要，但有的话就传
            "semi_transparent": bool(semi_enable),         # 半透明开关
            "semi_strength": float(semi_strength or 0.5),  # 力度
            "semi_mode": str(semi_mode or "auto"),         # 模式
            "remove_white_halo": bool(defringe_enable),    # 去白边开关
            "defringe_strength": float(defringe_strength or 0.7),  # 去白边强度
        }

        # 根据实际实现的签名，筛选出它能接收的参数
        sig = inspect.signature(_core_process_batch_images)
        param_names = list(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}

        # 判断是否需要传 engine
        # 常见两类：
        #   A) process_batch_images(engine, files, **kwargs)
        #   B) process_batch_images(files, background_image=..., input_size=..., ...)
        if param_names and param_names[0] in ("engine", "eng"):
            out = _core_process_batch_images(engine, paths, **filtered_kwargs)
        else:
            # 老版本/精简签名：只收 (files, background_image)
            if param_names[:2] == ["files", "background_image"] and len(sig.parameters) == 2:
                out = _core_process_batch_images(paths, candidate_kwargs["background_image"])
            else:
                # 新版但不带 engine：把它支持的关键字尽可能传进去
                out = _core_process_batch_images(paths, **filtered_kwargs)

        # 统一整理返回值给 UI：
        # 约定：
        #   - 返回 (zip_path, status_str)
        #   - 返回 (zip_path, previews, status_str)
        #   - 返回 zip_path
        if isinstance(out, tuple):
            if len(out) == 2:
                return out  # (zip_path, status)
            if len(out) == 3:
                return out[0], out[2]  # (zip_path, previews, status) -> (zip_path, status)
        return out, f"✅ 批量处理完成（模型：{short}，输入：{res}×{res}）"

    except Exception as e:
        tb = traceback.format_exc(limit=2)
        logger.error(f"批量图片处理失败：{e}\n{tb}")
        return None, f"❌ 批量图片处理失败：{e}\n```\n{tb}\n```"
