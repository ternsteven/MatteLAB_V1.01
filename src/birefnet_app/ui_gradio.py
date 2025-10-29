\
# -*- coding: utf-8 -*-
"""
BiRefNet Gradio 5.4 UI (original-style layout)
- Controls on the left, preview on the right
- Bottom toolbar: open output folder / open models folder / clear temp / clear outputs
- Collapsible option groups (click-to-expand)
- Live hints while dragging sliders (resolution VRAM estimate, semi-transparency strength)
"""
import os
import platform
import shutil
import gradio as gr

# ---- split project backends ----
from .engine import BiRefEngine, EngineConfig
from .config_models import model_descriptions
from .batch import process_batch_images
from .video import process_single_video
from .settings import PRED_OUTPUT_DIR, ensure_dirs

# -------------------------
# Helper actions
# -------------------------
def _open_dir(path: str) -> str:
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if platform.system() == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
        return f"✅ 已打开目录：{path}"
    except Exception as e:
        return f"❌ 打开目录失败：{e}"

def _clear_dir(path: str) -> str:
    if not os.path.exists(path):
        return f"ℹ️ 目录不存在：{path}"
    removed, failed = 0, 0
    for name in os.listdir(path):
        p = os.path.join(path, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
            removed += 1
        except Exception:
            failed += 1
    return f"🧹 清理完成：删除 {removed} 个，失败 {failed} 个（{path}）"

def _open_output_dir() -> str:
    return _open_dir(PRED_OUTPUT_DIR)

def _clear_output_dir() -> str:
    return _clear_dir(PRED_OUTPUT_DIR)

def _open_models_dir() -> str:
    path = os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "models_local")
    return _open_dir(path)

def _clear_temp_cache() -> str:
    base = os.getcwd()
    candidates = [
        os.path.join(base, "gradio_cached_examples"),
        os.path.join(base, "__pycache__"),
        os.path.join(base, "src", "__pycache__"),
        os.path.join(base, "src", "birefnet_app", "__pycache__"),
    ]
    removed = []
    for p in candidates:
        if os.path.exists(p):
            try:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(p)
            except Exception:
                pass
    return f"🧼 已清理临时缓存：{', '.join(removed) if removed else '无可清理项'}"

# -------------------------
# Main UI
# -------------------------
def create_interface():
    ensure_dirs()
    engine = BiRefEngine(EngineConfig("General", (1024, 1024)))

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
                            resolution    = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="输入分辨率")
                            res_hint      = gr.Markdown("",
                                elem_classes=["hint-box"])

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
                            resolution.input(_res_hint, inputs=resolution, outputs=res_hint)

                        # 半透明与去白边（点击展开）
                        with gr.Accordion("🪄 边缘增强选项（点击展开）", open=False):
                            with gr.Row():
                                semi_enable   = gr.Checkbox(label="半透明边缘（发丝/薄纱）", value=False)
                                defringe_enable   = gr.Checkbox(label="去白边（防渗色）", value=False)
                            with gr.Group(visible=False) as semi_grp:
                                semi_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="半透明强度（越大越实）")
                                semi_mode     = gr.Radio(choices=["auto", "暗部优先", "透色优先"], value="auto", label="半透明模式")
                                semi_hint     = gr.Markdown("", elem_classes=["hint-box"])

                                def _semi_hint(val, mode):
                                    if val is None: val=0.5
                                    band = int(2 + val*10)
                                    mode_t = {"auto":"自动","暗部优先":"更保守","透色优先":"更通透"}.get(mode, "自动")
                                    return f"**强度**：{val:.2f}（近似边带≈{band}px） · **模式**：{mode_t}。建议：烟雾 0.6–0.8；薄纱 0.4–0.6；玻璃/水面 0.3–0.5。"
                                semi_strength.input(_semi_hint, inputs=[semi_strength, gr.State("auto")], outputs=semi_hint)
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

                def on_process(img, bg_img, model, res, semi_en, semi_str, semi_md, def_en, def_str, progress=gr.Progress()):
                    if img is None:
                        return None, None
                    short = model.split(" - ")[0].strip() if isinstance(model, str) and " - " in model else model
                    engine.load_model(short, (int(res), int(res)))
                    def cb(p, msg): progress(p, desc=msg)
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
                            resolution_b    = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="输入分辨率")
                            res_hint_b      = gr.Markdown("", elem_classes=["hint-box"])
                            resolution_b.input(lambda r: f"批量分辨率：{int(r)}×{int(r)}，显存占用和速度与单图相当。", inputs=resolution_b, outputs=res_hint_b)

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
                    short = model.split(" - ")[0].strip() if isinstance(model, str) and " - " in model else model
                    engine.load_model(short, (int(res), int(res)))
                    def cb(p, msg): progress(p, desc=msg)
                    zip_path, saved_paths, msg = process_batch_images(
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
                            resolution_v    = gr.Slider(minimum=256, maximum=1536, step=64, value=768, label="输入分辨率（视频建议≤1536）")
                            res_hint_v      = gr.Markdown("", elem_classes=["hint-box"])
                            resolution_v.input(lambda r: f"更高分辨率将显著降低速度。当前：{int(r)}×{int(r)}。", inputs=resolution_v, outputs=res_hint_v)

                        with gr.Accordion("🪄 边缘增强选项（点击展开）", open=False):
                            semi_enable_v   = gr.Checkbox(label="半透明边缘", value=False)
                            semi_strength_v = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="半透明强度")
                            semi_mode_v     = gr.Radio(choices=["auto", "暗部优先", "透色优先"], value="auto", label="半透明模式")
                            defringe_enable_v   = gr.Checkbox(label="去白边（防渗色）", value=False)
                            defringe_strength_v = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.7, label="去白边力度")

                        run_v = gr.Button("🎬 开始处理视频", variant="primary")

                    with gr.Column(scale=5):
                        vid_out= gr.Video(label="输出视频", height=280)

                def on_video(vpath, bg_img, model, res, semi_en, semi_str, semi_md, def_en, def_str, progress=gr.Progress()):
                    if not vpath:
                        return None
                    short = model.split(" - ")[0].strip() if isinstance(model, str) and " - " in model else model
                    engine.load_model(short, (int(res), int(res)))
                    def cb(p, msg): progress(p, desc=msg)
                    out_path, msg = process_single_video(
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

        btn_open_out.click(lambda: _open_output_dir(), outputs=tool_status)
        btn_open_model.click(lambda: _open_models_dir(), outputs=tool_status)
        btn_clean_temp.click(lambda: _clear_temp_cache(), outputs=tool_status)
        btn_clean_out.click(lambda: _clear_output_dir(), outputs=tool_status)

    return demo
