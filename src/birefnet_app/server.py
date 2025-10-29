# -*- coding: utf-8 -*-
from .ui_gradio import create_interface

def run_app(server_name="0.0.0.0", server_port=None, inbrowser=False, share=False):
    demo = create_interface()
    # gradio>=4/5 的 queue/launch 参数兼容处理
    demo.queue().launch(server_name=server_name, server_port=server_port, inbrowser=inbrowser, share=share)
