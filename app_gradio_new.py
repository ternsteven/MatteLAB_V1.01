# app_gradio_new.py
# -*- coding: utf-8 -*-
"""
启动脚本只负责：
1) 设环境（HF_HOME等）
2) 确保目录
3) 构建并启动UI
"""

import os
import warnings


# 屏蔽 timm 的未来警告（保留你的原逻辑）
warnings.filterwarnings("ignore", message="Importing from timm", category=FutureWarning)

# —— 可选环境预设（更稳）——
os.environ.setdefault("GRADIO_USE_BROTLI", "0")  # 关闭 Gradio 的 Brotli 中间件以规避 h11 长度不一致问题
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "models_local"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main(
    host: str = "127.0.0.1",   # 若需局域网访问，改为 "0.0.0.0"
    port: int = 7860,
    open_browser: bool = False,
    share: bool = False,
):
    # 延迟导入，避免启动时牵出重依赖导致循环导入
    from src.birefnet_app.settings import ensure_dirs
    from src.birefnet_app.ui_gradio import create_interface

    ensure_dirs()
    demo = create_interface()

    # Gradio 5.x：queue() 不带并发参数；事件级并发请在 ui_gradio.py 的 .click() 上设置
    demo.queue().launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=open_browser,
    )


if __name__ == "__main__":
    main(
        host="127.0.0.1",
        port=7860,
        open_browser=True,  # True: 启动后自动打开浏览器
        share=False,
    )


