# -*- coding: utf-8 -*-
"""
birefnet_app.ui.handlers
用于 Gradio UI 的小工具动作：打开目录、清理目录。
函数返回中文提示字符串，便于直接绑定到 UI 的状态文本组件。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Iterable

__all__ = ["open_dir", "clear_dir"]


def open_dir(path: str) -> str:
    """
    在系统文件管理器中打开指定目录；若目录不存在则自动创建。
    返回：适合显示在 UI 的状态字符串。
    """
    try:
        if not path:
            return "❌ 路径为空"
        abs_path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(abs_path, exist_ok=True)

        system = platform.system()
        if system == "Windows":
            try:
                os.startfile(abs_path)  # type: ignore[attr-defined]
            except Exception:
                # 某些受限环境下回退到 explorer
                subprocess.Popen(["explorer", abs_path])
        elif system == "Darwin":
            subprocess.Popen(["open", abs_path])
        else:
            # Linux / 其他 UNIX
            subprocess.Popen(["xdg-open", abs_path])

        return f"✅ 已打开目录：{abs_path}"
    except Exception as e:
        return f"❌ 打开目录失败：{e}"


def _safe_listdir(path: str) -> list[str]:
    try:
        return os.listdir(path)
    except (FileNotFoundError, PermissionError):
        return []


def clear_dir(path: str, keep: Iterable[str] = ()) -> str:
    """
    清空目录内容（保留目录本身）。
    keep: 需要保留的不删除的“名称”（与文件/子目录名精确匹配）。

    返回：适合显示在 UI 的状态字符串。
    """
    try:
        if not path:
            return "❌ 路径为空"
        abs_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(abs_path):
            return f"ℹ️ 目录不存在：{abs_path}"

        removed = 0
        failed = 0
        kept = set(keep or ())

        for name in _safe_listdir(abs_path):
            if name in kept:
                continue
            p = os.path.join(abs_path, name)
            try:
                if os.path.islink(p) or os.path.isfile(p):
                    os.unlink(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=False)
                else:
                    # 其他特殊类型（如 FIFO、设备文件等）
                    os.unlink(p)
                removed += 1
            except Exception:
                failed += 1

        suffix = f"（已保留：{', '.join(sorted(kept))}）" if kept else ""
        return f"🧹 已清理：{removed} 项；失败 {failed} 项。目录：{abs_path} {suffix}"
    except Exception as e:
        return f"❌ 清理失败：{e}"
