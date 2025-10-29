#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载（或更新）模型权重到仓库根目录的 ./models_local 目录（默认）。
默认从 Hugging Face Hub 拉取“快照”（snapshot），便于版本锁定与离线复现。

用法示例：
  # 默认下载到 项目根目录/models_local/<repo-id-安全化路径>/
  python scripts/download_weights.py --repo-id your-org/your-model

  # 指定 revision 与自定义目录（相对路径会视为相对“项目根目录”）
  python scripts/download_weights.py \
    --repo-id your-org/your-model \
    --revision main \
    --local-dir models_local/custom_name

如需访问私有模型，请在环境变量中设置 HF_TOKEN：
  export HF_TOKEN=hf_xxx
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# 计算“项目根目录”：当前文件（位于 scripts/）的上一级目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _resolve_local_dir(local_dir_arg: Optional[str]) -> Path:
    """
    把 --local-dir 解析为项目根目录下的绝对路径：
    - 若未提供参数，则使用 PROJECT_ROOT / "models_local"
    - 若提供的是相对路径，则视为相对 PROJECT_ROOT
    - 若提供的是绝对路径，则直接使用
    """
    if not local_dir_arg:
        return PROJECT_ROOT / "models_local"
    p = Path(local_dir_arg)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p

def _sanitize_subdir_name(repo_id: str) -> str:
    """
    将 repo_id (如 "org/name") 转为安全的子目录名，例如 "org__name"
    这样每个模型会有自己独立的子目录，避免混淆。
    """
    return repo_id.replace("/", "__")

def main():
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("❌ 需要先安装 huggingface-hub：pip install huggingface-hub", file=sys.stderr)
        raise

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face 仓库，如 your-org/your-model")
    parser.add_argument("--revision", default="main", help="分支/标签/commit，默认 main")
    parser.add_argument(
        "--local-dir",
        default=None,
        help="下载到的目录。若为相对路径，视为相对项目根目录。默认 PROJECT_ROOT/models_local",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=["*.safetensors", "*.bin", "*.pt", "*.json", "*.model", "*.txt", "*.md"],
        help="仅下载匹配的文件模式（默认覆盖常见权重/索引/配置文件）。传空列表表示拉取仓库全部内容。",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=None,
        help="忽略匹配的文件模式，如 *.md *.txt",
    )
    parser.add_argument(
        "--no-subdir",
        action="store_true",
        help="默认会在目标目录下以 repo-id 创建子目录；加上该参数则直接下载到目标目录根。",
    )
    args = parser.parse_args()

    # 通过环境变量读取 HF_TOKEN（若需要私库）
    hf_token: Optional[str] = os.environ.get("HF_TOKEN")

    base_dir = _resolve_local_dir(args.local_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 目标目录：默认 base_dir/<org__name>，可通过 --no-subdir 关闭
    target_dir = base_dir if args.no_subdir else base_dir / _sanitize_subdir_name(args.repo_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔽 正在从 Hugging Face 拉取: {args.repo_id} @ {args.revision}")
    print(f"📁 下载位置: {target_dir}")

    # 允许通过传入空列表来表示“拉取全部内容”
    allow_patterns = args.allow_patterns
    if allow_patterns == [""]:
        allow_patterns = None

    # snapshot_download 的 token 处理在新旧版本略有不同；这里做兼容
    try:
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            ignore_patterns=args.ignore_patterns,
            use_auth_token=hf_token,  # 兼容旧版 huggingface_hub
        )
    except TypeError:
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            ignore_patterns=args.ignore_patterns,
        )

    print("✅ 权重就绪！")

if __name__ == "__main__":
    sys.exit(main())
