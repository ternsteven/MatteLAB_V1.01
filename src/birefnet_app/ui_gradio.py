# -*- coding: utf-8 -*-
"""
BiRefNet Gradio 5.4 UI (original-style layout)
- Controls on the left, preview on the right
- Bottom toolbar: open output folder / open models folder / clear temp / clear outputs
- Collapsible option groups (click-to-expand)
- Live hints while dragging sliders (resolution VRAM estimate, semi-transparency strength)
- 模型切换后自动提示 + 懒加载提示小气泡
- 新增：⚡ 预加载当前模型 按钮（单图/批量/视频 三处）
"""

import os
import shutil
import gradio as gr

# ---- split project backends ----
from .engine import BiRefEngine, EngineConfig
from .config_models import model_descriptions
from .batch import process_batch_images
from .video import process_single_video
from .settings import PRED_OUTPUT_DIR, ensure_dirs
# 从 handlers.py 导入小工具动作（并用旧名别名，保持既有调用不变）
from .ui.handlers import open_dir as _open_dir, clear_dir as _clear_dir


# -------------------------
# Temp cache helper (UI 专用：一次清理多处缓存)
# -------------------------
def _clear_temp_cache() -> str:
    base = os.getcwd()
    candidates = [
        os.path.join(base, "gradio_cached_examples"),
        os.path.join(base, "__pycache__"),
        os.path.join(base, "src", "__pycache__"),
        os.path.join(base, "src", "birefnet_app", "__pycache__"),
        os.path.join(base, "temp"),
    ]
    removed = []
    for p in candidates:
        try:
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)
                removed.append(p)
        except Exception:
            pass
    return f"🧼 已清理临时缓存：{', '.join(removed) if removed else '无可清理项'}"


# -------------------------
# Hint helpers
# -------------------------
def _parse_model_choice(model) -> str:
    """将下拉框的 'Name - 描述' 解析为短名 'Name'；保持兼容纯 Name。"""
    if isinstance(model, str) and " - " in model:
        return model.split(" - ", 1)[0].strip()
    return str(model) if model is not None else ""

def _model_hint_text(model) -> str:
    """
    生成模型提示气泡文案：模型描述 + 懒加载提示 + 建议。
    该文案只在 UI 上显示，不改变任何功能。
    """
    short = _parse_model_choice(model)
    desc  = model_descriptions.get(short, "通用前景分割模型")
    lines = [
        f"**已选模型**：`{short}` — {desc}",
        "**提示**：首次使用该模型时会在“开始处理/预加载”阶段**自动加载/下载权重**（仅一次，之后复用，称为“懒加载”）。如网络受限，请先手动下载到 `models_local/`。",
        "**建议**：若频繁切换模型，可先用较小图片运行一次以完成缓存，再进行大图/批量/视频处理。",
    ]
    return "<br>".join(lines)


# -------------------------
# Main UI
# -------------------------
def create_interface():
    ensure_dirs()
    engine = BiRefEngine(EngineConfig("General", (1024, 1024)))

    # 预加载动作（供三个页签复用）
    def _preload_model(m, r):
        short = _parse_model_choice(m)
        engine.load_model(short, (int(r), int(r)))
        return _model_hint_text(m) + f"<br>✅ **已预加载**：`{short}` @ {int(r)}×{int(r)}。"

    # CSS close to the original taste
    custom_css = """
    .gradio-container { max-width: 1250px !important; margin: auto !important; }
    .biref-header h2 { margin: 0; }
    .hint-box { font-size: 0.92rem; line-height: 1.35; background: rgba(245,247,250,.8); border-left: 4px solid #7c3aed; padding: 8px 10px; border-radius: 6px; }
    .footer-toolbar { border-top: 1px solid #e5e7eb; margin-top: 8px; padding-top: 8px; }
    .footer-toolbar .gr-button { min-width: 180px; }
    .tight .gr-form { gap: 8px !important; }
    """

    with gr.Blocks(title="BiRefNet 背景移除（模块化）", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Row(elem_classes=["biref-header"]):
            gr.Markdown("## BiRefNet 背景移除（模块化 UI）")

        # ---------------- Left/Right layout
        with gr.Tabs():
            # ========== 单图 ==========
            with gr.Tab("🖼️ 单图"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["tight"]):
                        inp  = gr.Image(type="pil", label="输入图片", height=360)
                        bg   = gr.Image(type="pil", label="背景（可选，留空=>透明）", height=180)

                        # 基础参数
                        with gr.Accordion("⚙️ 模型与分辨率", open=True):
                            model_choices = [f"{k} - {v}" for k, v in model_descriptions.items()]
                            model_choice  = gr.Dropdown(choices=model_choices, value=model_choices[0], label="模型")
                            # 模型提示气泡
                            model_hint    = gr.Markdown(_model_hint_text(model_choices[0]), elem_classes=["hint-box"])

                            resolution    = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="输入分辨率")
                            res_hint      = gr.Markdown("", elem_classes=["hint-box"])

                            def _res_hint(res):
                                try:
                                    import torch
                                    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                                except Exception:
                                    gpu = "Unknown GPU/CPU"
                                base_mem = 2.3  # rough @1024
                                mem = base_mem * (int(res)/1024)**2
                                speed = "🚀 很快" if res<=512 else ("⚡ 推荐" if res<=1024 else ("🐢 稍慢" if res<=1536 else "🐌 慢"))
                                return f"**设备**：{gpu}<br>**分辨率**：{res}×{res} | **预估显存**≈{mem:.1f}GB | {speed}"

                            # 预加载按钮
                            preload_btn = gr.Button("⚡ 预加载当前模型")
                            preload_btn.click(_preload_model, inputs=[model_choice, resolution], outputs=[model_hint])

                            # 事件：分辨率拖动提示 / 模型切换提示
                            resolution.input(_res_hint, inputs=resolution, outputs=res_hint)
                            model_choice.change(_model_hint_text, inputs=model_choice, outputs=model_hint)

                        # 半透明与去白边（点击展开）
                        with gr.Accordion("🪄 边缘增强选项（点击展开）", open=False):
                            with gr.Row():
                                semi_enable      = gr.Checkbox(label="半透明边缘（发丝/薄纱）", value=False)
                                defringe_enable  = gr.Checkbox(label="去白边（防渗色）", value=False)

                            with gr.Group(visible=False) as semi_grp:
                                semi_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="半透明强度（越大越实）")
                                semi_mode     = gr.Radio(choices=["auto", "暗部优先", "透色优先"], value="auto", label="半透明模式")
                                semi_hint     = gr.Markdown("", elem_classes=["hint-box"])

                                def _semi_hint(val, mode):
                                    if val is None: val = 0.5
                                    band = int(2 + val*10)
                                    mode_t = {"auto":"自动","暗部优先":"更保守","透色优先":"更通透"}.get(mode, "自动")
                                    return f"**强度**：{val:.2f}（近似边带≈{band}px） · **模式**：{mode_t}。建议：烟雾 0.6–0.8；薄纱 0.4–0.6；玻璃/水面 0.3–0.5。"

                                semi_strength.input(lambda v, m: _semi_hint(v, m), inputs=[semi_strength, semi_mode], outputs=semi_hint)
                                semi_mode.change(lambda m, v: _semi_hint(v, m), inputs=[semi_mode, semi_strength], outputs=semi_hint)

                            with gr.Group(visible=False) as defringe_grp:
                                defringe_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.7, label="去白边力度")
                                gr.Markdown("白色毛边/渗色明显时提高该值（会做颜色回灌+轻微收边）。", elem_classes=["hint-box"])

                            semi_enable.change(lambda on: gr.update(visible=on), inputs=semi_enable, outputs=semi_grp)
                            defringe_enable.change(lambda on: gr.update(visible=on), inputs=defringe_enable, outputs=defringe_grp)

                        run_btn = gr.Button("🚀 开始处理", variant="primary")

                    with gr.Column(scale=5):
                        out  = gr.Image(label="合成结果 / 透明 PNG", height=360)
                        mask = gr.Image(label="Mask / Alpha 预览", height=180)

                def on_process(img, bg_img, model, res, semi_en, semi_str, semi_md, def_en, def_str):
                    if img is None:
                        return None, None
                    short = _parse_model_choice(model)
                    engine.load_model(short, (int(res), int(res)))
                    result, m = engine.apply_background_replacement(
                        image=img,
                        background_image=bg_img,
                        model_name=short,
                        input_size=(int(res), int(res)),
                        semi_transparent=bool(semi_en),
                        semi_strength=float(semi_str or 0.5),
                        semi_mode=str(semi_md),
                        remove_white_halo=bool(def_en),
                        defringe_strength=float(def_str or 0.7),
                    )
                    return result, m

                run_btn.click(
                    on_process,
                    [inp, bg, model_choice, resolution,
                     semi_enable, semi_strength, semi_mode,
                     defringe_enable, defringe_strength],
                    [out, mask]
                )

            # ========== 批量 ==========
            with gr.Tab("📁 批量"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["tight"]):
                        files_b = gr.Files(label="选择多张图片", file_count="multiple", type="filepath")
                        bg_b    = gr.Image(type="pil", label="背景（可选，留空=>透明）", height=180)

                        with gr.Accordion("⚙️ 模型与分辨率", open=True):
                            model_choices = [f"{k} - {v}" for k, v in model_descriptions.items()]
                            model_choice_b  = gr.Dropdown(choices=model_choices, value=model_choices[0], label="模型")
                            model_hint_b    = gr.Markdown(_model_hint_text(model_choices[0]), elem_classes=["hint-box"])

                            resolution_b    = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="输入分辨率")
                            res_hint_b      = gr.Markdown("", elem_classes=["hint-box"])

                            # 预加载按钮
                            preload_btn_b = gr.Button("⚡ 预加载当前模型")
                            preload_btn_b.click(_preload_model, inputs=[model_choice_b, resolution_b], outputs=[model_hint_b])

                            resolution_b.input(lambda r: f"批量分辨率：{int(r)}×{int(r)}，显存占用和速度与单图相当。", inputs=resolution_b, outputs=res_hint_b)
                            model_choice_b.change(_model_hint_text, inputs=model_choice_b, outputs=model_hint_b)

                        with gr.Accordion("🪄 边缘增强选项（点击展开）", open=False):
                            semi_enable_b   = gr.Checkbox(label="半透明边缘", value=False)
                            semi_strength_b = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="半透明强度")
                            semi_mode_b     = gr.Radio(choices=["auto", "暗部优先", "透色优先"], value="auto", label="半透明模式")
                            defringe_enable_b   = gr.Checkbox(label="去白边（防渗色）", value=False)
                            defringe_strength_b = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.7, label="去白边力度")

                        run_b = gr.Button("📦 开始批量处理", variant="primary")

                    with gr.Column(scale=5):
                        zip_out = gr.File(label="批量结果 ZIP")
                        gallery = gr.Gallery(label="预览（保存图在 ZIP 内与输出目录）", columns=4, height=320, preview=True)

                def on_batch(files, bg_img, model, res, semi_en, semi_str, semi_md, def_en, def_str, progress=gr.Progress()):
                    if not files:
                        return None, []
                    short = _parse_model_choice(model)
                    engine.load_model(short, (int(res), int(res)))
                    def cb(p, msg): progress(p, desc=msg)
                    zip_path, saved_paths, _msg = process_batch_images(
                        engine, files,
                        background_image=bg_img,
                        input_size=(int(res), int(res)),
                        semi_transparent=bool(semi_en),
                        semi_strength=float(semi_str or 0.5),
                        semi_mode=str(semi_md),
                        remove_white_halo=bool(def_en),
                        defringe_strength=float(def_str or 0.7),
                        progress_cb=cb
                    )
                    show = [p for p in saved_paths if isinstance(p, str) and not p.startswith("[ERROR]")]
                    return zip_path, show

                run_b.click(
                    on_batch,
                    [files_b, bg_b, model_choice_b, resolution_b,
                     semi_enable_b, semi_strength_b, semi_mode_b,
                     defringe_enable_b, defringe_strength_b],
                    [zip_out, gallery]
                )

            # ========== 视频 ==========
            with gr.Tab("🎬 视频"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["tight"]):
                        vid_in = gr.Video(label="输入视频", height=280)
                        bg_v   = gr.Image(type="pil", label="背景（可选，留空=>透明）", height=160)

                        with gr.Accordion("⚙️ 模型与分辨率", open=True):
                            model_choices = [f"{k} - {v}" for k, v in model_descriptions.items()]
                            model_choice_v  = gr.Dropdown(choices=model_choices, value=model_choices[0], label="模型")
                            model_hint_v    = gr.Markdown(_model_hint_text(model_choices[0]), elem_classes=["hint-box"])

                            resolution_v    = gr.Slider(minimum=256, maximum=1536, step=64, value=768, label="输入分辨率（视频建议≤1536）")
                            res_hint_v      = gr.Markdown("", elem_classes=["hint-box"])

                            # 预加载按钮
                            preload_btn_v = gr.Button("⚡ 预加载当前模型")
                            preload_btn_v.click(_preload_model, inputs=[model_choice_v, resolution_v], outputs=[model_hint_v])

                            resolution_v.input(lambda r: f"更高分辨率将显著降低速度。当前：{int(r)}×{int(r)}。", inputs=resolution_v, outputs=res_hint_v)
                            model_choice_v.change(_model_hint_text, inputs=model_choice_v, outputs=model_hint_v)

                        with gr.Accordion("🪄 边缘增强选项（点击展开）", open=False):
                            semi_enable_v   = gr.Checkbox(label="半透明边缘", value=False)
                            semi_strength_v = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="半透明强度")
                            semi_mode_v     = gr.Radio(choices=["auto", "暗部优先", "透色优先"], value="auto", label="半透明模式")
                            defringe_enable_v   = gr.Checkbox(label="去白边（防渗色）", value=False)
                            defringe_strength_v = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.7, label="去白边力度")

                        run_v = gr.Button("🎬 开始处理视频", variant="primary")

                    with gr.Column(scale=5):
                        vid_out = gr.Video(label="输出视频", height=280)

                def on_video(vpath, bg_img, model, res, semi_en, semi_str, semi_md, def_en, def_str, progress=gr.Progress()):
                    if not vpath:
                        return None
                    short = _parse_model_choice(model)
                    engine.load_model(short, (int(res), int(res)))
                    def cb(p, msg): progress(p, desc=msg)
                    out_path, _msg = process_single_video(
                        engine,
                        input_video_path=vpath,
                        background_image=bg_img,
                        input_size=(int(res), int(res)),
                        semi_transparent=bool(semi_en),
                        semi_strength=float(semi_str or 0.5),
                        semi_mode=str(semi_md),
                        remove_white_halo=bool(def_en),
                        defringe_strength=float(def_str or 0.7),
                        progress_cb=cb
                    )
                    return out_path

                run_v.click(
                    on_video,
                    [vid_in, bg_v, model_choice_v, resolution_v,
                     semi_enable_v, semi_strength_v, semi_mode_v,
                     defringe_enable_v, defringe_strength_v],
                    [vid_out]
                )

        # ------------- Footer toolbar (bottom) -------------
        with gr.Row(elem_classes=["footer-toolbar"]):
            btn_open_out   = gr.Button("📂 打开输出目录")
            btn_open_model = gr.Button("📦 打开模型目录")
            btn_clean_temp = gr.Button("🧼 清理临时缓存")
            btn_clean_out  = gr.Button("🧹 清理输出结果")
            tool_status    = gr.Markdown("")

        # 统一使用 handlers 中的实现（打开/清空），与工程内 settings 路径协同
        btn_open_out.click(lambda: _open_dir(PRED_OUTPUT_DIR), outputs=tool_status)

        # 模型目录：优先 HF_HOME，其次工程内 models_local
        btn_open_model.click(
            lambda: _open_dir(os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "models_local")),
            outputs=tool_status
        )

        # 临时缓存：UI 专用多目标清理
        btn_clean_temp.click(_clear_temp_cache, outputs=tool_status)

        # 输出目录清理：可替换为 _clear_dir(PRED_OUTPUT_DIR, keep=(".gitkeep",))
        btn_clean_out.click(lambda: _clear_dir(PRED_OUTPUT_DIR), outputs=tool_status)

    return demo
