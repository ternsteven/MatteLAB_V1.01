
# —— 半透明扣除说明文案（用于 UI 展示一致性） ——
SEMI_TIP = """
**扣除半透明**  
- 开关：默认关闭以保持旧版本行为。  

**力度 / 区域大小（0–1）** 影响 inpaint 半径、融合强度、平滑半径。  
建议：**烟雾** 0.6–0.8；**薄纱/纱网** 0.4–0.6；**玻璃/水面** 0.3–0.5。

**模式**  
- **auto**：自动选择，不再额外弯曲 α 曲线。  
- **暗部优先**：适合阴影、烟雾略压暗背景（更保守，防止过度透明）。  
- **透色优先**：适合薄纱、雾气高亮/低饱和（更开放，通透感更强）。  
半身人像/发丝建议先选 **Matting**，再开启本功能。
"""

# -*- coding: utf-8 -*-
import gradio as gr

def _semi_mode_hint_text(mode: str) -> str:
    mapping = {
        "auto": "🧠 auto：自动折中。",
        "暗部优先": "🌑 暗部优先：适合阴影/烟雾，整体更保守。",
        "透色优先": "✨ 透色优先：适合薄纱/雾气，整体更通透。",
    }
    return mapping.get(mode, mapping["auto"])
def build_semi_controls():
    """半透明抠图 UI：默认隐藏选项，勾选后展开；返回 (semi_enable, semi_strength, semi_mode)"""
    semi_enable = gr.Checkbox(label="扣除半透明", value=False)

    with gr.Group(visible=False) as semi_opts:
        semi_strength = gr.Slider(
            label="半透明力度 / 区域大小", minimum=0.0, maximum=1.0, step=0.05, value=0.5,
            info="影响 inpaint 半径、融合强度、平滑半径。建议：玻璃/水面 0.3–0.5；薄纱 0.4–0.6；烟雾 0.6–0.8。"
        )
        semi_mode = gr.Radio(
            label="半透明处理模式",
            choices=["auto", "暗部优先", "透色优先"],
            value="auto",
        )
        mode_hint = gr.Markdown(_semi_mode_hint_text("auto"))

    # 勾选→显示/隐藏选项组
    semi_enable.change(
        fn=lambda on: gr.update(visible=on),
        inputs=semi_enable, outputs=semi_opts
    )
    # 模式变化→更新提示
    semi_mode.change(
        fn=_semi_mode_hint_text,
        inputs=semi_mode, outputs=mode_hint
    )
    return semi_enable, semi_strength, semi_mode

