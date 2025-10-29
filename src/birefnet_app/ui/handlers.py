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

PRED_DIR = os.path.join(os.getcwd(), "preds-BiRefNet")
MODEL_DIR = os.path.join(os.getcwd(), "models_local")

def open_dir(path: str) -> str:
    abs_path = os.path.abspath(path); os.makedirs(abs_path, exist_ok=True)
    try:
        if platform.system() == "Windows":
            subprocess.Popen(f'explorer "{abs_path}"')
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", abs_path])
        else:
            subprocess.Popen(["xdg-open", abs_path])
        return f"📂 已打开：{abs_path}"
    except Exception as e:
        return f"⚠️ 无法打开目录：{e}"

def clear_dir(path: str) -> str:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return f"ℹ️ 目录不存在：{abs_path}"
    try:
        for name in os.listdir(abs_path):
            p = os.path.join(abs_path, name)
            try:
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            except Exception as e:
                return f"⚠️ 删除失败 {p}: {e}"
        return f"✅ 已清空：{abs_path}"
    except Exception as e:
        return f"⚠️ 清理失败：{e}"

def clear_cache_safe() -> str:
    cleared = []
    for path in ["weights", "preds-BiRefNet", "__pycache__"]:
        if os.path.exists(path):
            try: shutil.rmtree(path); cleared.append(path)
            except Exception as e: pass
    if os.path.exists(MODEL_DIR):
        for d in os.listdir(MODEL_DIR):
            if d.startswith("models--"):
                full = os.path.join(MODEL_DIR, d)
                try: shutil.rmtree(full); cleared.append(full)
                except Exception: pass
    return "✅ 已清理（保留离线模型）:\n" + ("\n".join(cleared) if cleared else "无可清理")

def clear_cache_full() -> str:
    cleared = []
    for path in ["weights", "preds-BiRefNet", "models_local", "__pycache__"]:
        if os.path.exists(path):
            try: shutil.rmtree(path); cleared.append(path)
            except Exception: pass
    return "🧨 已彻底清理：\n" + ("\n".join(cleared) if cleared else "无可清理")
