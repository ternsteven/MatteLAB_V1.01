# -*- coding: utf-8 -*-
"""
BiRefNet Gradio 5.4 UI (ROI-only, modular)
- 单图/批量/视频三个页签
- 模型切换提示 + 懒加载说明 + ⚡ 预加载按钮
- 分辨率拖动显存/速度提示
- 半透明边缘 & 去白边选项（点击展开）
- ✍️ ROI 画板（ImageEditor, layers=True）：可选先裁剪再分割，加速并减少误检
- 底部工具栏：打开输出/模型目录、清理临时/输出、安全清理/彻底清理
"""

import os
import shutil
import numpy as np
import gradio as gr

# ---- backends ----
from .engine import BiRefEngine, EngineConfig
from .config_models import model_descriptions
from .batch import process_batch_images
from .video import process_single_video
from .settings import PRED_OUTPUT_DIR, ensure_dirs

# ---- handlers: directory & cache helpers ----
from .ui.handlers import (
    open_dir as _open_dir,
    clear_dir as _clear_dir,
    clear_cache_safe,
    clear_cache_full,
)

# ---- ROI helpers ----
from .ops.roi_ops import make_editor_thumbnail, editor_layers_to_mask_fullres
from .ui.handlers import open_dir as _open_dir, clear_dir as _clear_dir, clear_cache_safe, clear_cache_full



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
    """将下拉框的 'Name - 描述' 解析为短名 'Name'；兼容纯 Name。"""
    if isinstance(model, str) and " - " in model:
        return model.split(" - ", 1)[0].strip()
    return str(model) if model is not None else ""


def _model_hint_text(model) -> str:
    """模型提示：描述 + 懒加载提示 + 建议。"""
    short = _parse_model_choice(model)
    desc = model_descriptions.get(short, "通用前景分割模型")
    lines = [
        f"**已选模型**：`{short}` — {desc}",
        "**提示**：首次使用该模型时会在“开始处理/预加载”阶段**自动加载/下载权重**（仅一次，之后复用，称为“懒加载”）。如网络受限，请先手动下载到 `models_local/`。",
        "**建议**：若频繁切换模型，可先用较小图片运行一次以完成缓存，再进行大图/批量/视频处理。",
    ]
    return "<br>".join(lines)


# -------------------------
# Sketch / ROI helpers for ImageEditor
# -------------------------
def _init_roi_editor(img, long_side: int):
    """基于输入图片生成 ImageEditor 的初值与 meta。"""
    if img is None:
        return gr.update(), None
    editor_value, meta = make_editor_thumbnail(img, int(long_side))
    # ImageEditor 接受 dict：{"background": ..., "layers": [], "composite": ...}
    return editor_value, meta


def _clear_roi_layers(editor_value):
    """清空图层，仅保留背景。"""
    try:
        bg = editor_value.get("background") if isinstance(editor_value, dict) else None
    except Exception:
        bg = None
    return {"background": bg, "layers": [], "composite": bg}


def _on_roi_toggle(enabled, img, long_side):
    """切换启用 ROI 时初始化/隐藏画板。"""
    if enabled and img is not None:
        ev, meta = _init_roi_editor(img, int(long_side))
        return gr.update(visible=True), ev, meta
    return gr.update(visible=False), None, None


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

    # CSS
    custom_css = """
    .gradio-container { max-width: 1250px !important; margin: auto !important; }
    .biref-header h2 { margin: 0; }
    .hint-box { font-size: 0.92rem; line-height: 1.35; background: rgba(245,247,250,.8); border-left: 4px solid #7c3aed; padding: 8px 10px; border-radius: 6px; }
    .footer-toolbar { border-top: 1px solid #e5e7eb; margin-top: 8px; padding-top: 8px; }
    .footer-toolbar .gr-button { min-width: 165px; }
    .tight .gr-form { gap: 8px !important; }
    """

    with gr.Blocks(title="背景移除工具（BiRefNet）", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Row(elem_classes=["biref-header"]):
            gr.Markdown("## 背景移除工具 BiRefNet Background Remover")

        with gr.Tabs():
            # ================== 单图 ==================
            with gr.Tab("🖼️ 单图"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["tight"]):
                        inp = gr.Image(type="pil", label="输入图片", height=360)
                        bg = gr.Image(type="pil", label="背景（可选，留空=>透明）", height=180)

                        # 基础参数
                        with gr.Accordion("⚙️ 模型与分辨率", open=False):
                            model_choices = [f"{k} - {v}" for k, v in model_descriptions.items()]
                            if not model_choices:
                                model_choices = ["General - 通用前景分割模型"]
                            model_choice = gr.Dropdown(choices=model_choices, value=model_choices[0], label="模型")
                            model_hint = gr.Markdown(_model_hint_text(model_choices[0]), elem_classes=["hint-box"])

                            resolution = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="输入分辨率")
                            res_hint = gr.Markdown("", elem_classes=["hint-box"])

                            def _res_hint(res):
                                try:
                                    import torch
                                    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                                except Exception:
                                    gpu = "CPU/未检测到"
                                base_mem = 2.3  # 粗略基准（@1024）
                                mem = base_mem * (int(res) / 1024) ** 2
                                speed = "🚀 很快" if res <= 512 else ("⚡ 推荐" if res <= 1024 else ("🐢 稍慢" if res <= 1536 else "🐌 慢"))
                                return f"**设备**：{gpu}<br>**分辨率**：{res}×{res} | **预估显存**≈{mem:.1f}GB | {speed}"

                            preload_btn = gr.Button("⚡ 预加载当前模型")
                            preload_btn.click(_preload_model, inputs=[model_choice, resolution], outputs=[model_hint])
                            resolution.input(_res_hint, inputs=resolution, outputs=res_hint)
                            model_choice.change(_model_hint_text, inputs=model_choice, outputs=model_hint)

                        # 边缘增强
                        with gr.Accordion("🪄 边缘增强选项（点击展开）", open=False):
                            with gr.Row():
                                semi_enable = gr.Checkbox(label="半透明边缘（发丝/薄纱）", value=False)
                                defringe_enable = gr.Checkbox(label="去白边（防渗色）", value=False)

                            with gr.Group(visible=False) as semi_grp:
                                semi_strength = gr.Slider(0.0, 1.0, step=0.05, value=0.5, label="半透明强度（越大越实）")
                                semi_mode = gr.Radio(choices=["auto", "暗部优先", "透色优先"], value="auto", label="半透明模式")
                                semi_hint = gr.Markdown("", elem_classes=["hint-box"])

                                def _semi_hint(val, mode):
                                    if val is None:
                                        val = 0.5
                                    band = int(2 + val * 10)
                                    mode_t = {"auto": "自动", "暗部优先": "更保守", "透色优先": "更通透"}.get(mode, "自动")
                                    return f"**强度**：{val:.2f}（近似边带≈{band}px） · **模式**：{mode_t}。建议：烟雾 0.6–0.8；薄纱 0.4–0.6；玻璃/水面 0.3–0.5。"

                                semi_strength.input(lambda v, m: _semi_hint(v, m), inputs=[semi_strength, semi_mode], outputs=semi_hint)
                                semi_mode.change(lambda m, v: _semi_hint(v, m), inputs=[semi_mode, semi_strength], outputs=semi_hint)

                            with gr.Group(visible=False) as defringe_grp:
                                defringe_strength = gr.Slider(0.0, 1.0, step=0.05, value=0.7, label="去白边力度")
                                gr.Markdown("白色毛边/渗色明显时提高该值（会做颜色回灌+轻微收边）。", elem_classes=["hint-box"])

                            semi_enable.change(lambda on: gr.update(visible=on), inputs=semi_enable, outputs=semi_grp)
                            defringe_enable.change(lambda on: gr.update(visible=on), inputs=defringe_enable, outputs=defringe_grp)

                        # ROI 画板（ImageEditor + layers）
                        with gr.Accordion("✍️ 画板涂抹（ROI 指定区域，可选）", open=False):
                            roi_enable = gr.Checkbox(
                                label="启用 ROI 指定区域（在进入模型前裁剪并回贴）",
                                value=False,
                                info="开启后仅对你涂抹的 ROI 区域进行分割，速度更快、误检更少。"
                            )

                            with gr.Group(visible=False) as roi_group:
                                with gr.Row():
                                    roi_thumb_side = gr.Slider(256, 1200, step=64, value=640, label="缩略图长边 (仅影响画板)")
                                    roi_pad_px = gr.Slider(0, 128, step=2, value=16, label="ROI 外扩 padding（安全边）")
                                    roi_crop_before = gr.Checkbox(value=True, label="先裁剪再分割（更快更准）")

                                roi_canvas = gr.ImageEditor(
                                    label="在缩略图上用画笔涂抹 ROI（半透明预览，不影响像素）",
                                    type="numpy", image_mode="RGBA", height=380, sources=None, layers=True,
                                )

                                with gr.Row():
                                    roi_clear = gr.Button("清空涂抹", variant="secondary")
                                    gr.Markdown("提示：用画笔大致圈定需要抠图的区域，无需涂满。")

                            roi_meta_state = gr.State(value=None)

                            # 事件：启用/关闭 ROI 时初始化或隐藏
                            roi_enable.change(
                                _on_roi_toggle,
                                inputs=[roi_enable, inp, roi_thumb_side],
                                outputs=[roi_group, roi_canvas, roi_meta_state],
                                show_progress=False
                            )
                            # 缩略图长边变化时刷新画板
                            roi_thumb_side.change(
                                lambda en, im, ls: _init_roi_editor(im, ls) if en and im is not None else (gr.update(), None),
                                inputs=[roi_enable, inp, roi_thumb_side],
                                outputs=[roi_canvas, roi_meta_state],
                                show_progress=False
                            )
                            # 输入图片变化时刷新画板
                            inp.change(
                                lambda en, im, ls: _init_roi_editor(im, ls) if en and im is not None else (gr.update(), None),
                                inputs=[roi_enable, inp, roi_thumb_side],
                                outputs=[roi_canvas, roi_meta_state],
                                show_progress=False
                            )
                            # 清空按钮
                            roi_clear.click(_clear_roi_layers, inputs=[roi_canvas], outputs=[roi_canvas])

                        run_btn = gr.Button("🚀 开始处理", variant="primary")

                    with gr.Column(scale=5):
                        out = gr.Image(label="合成结果 / 透明 PNG", height=360)
                        mask = gr.Image(label="Mask / Alpha 预览", height=180)

                # 单图处理回调（ROI-only）
                def on_process(
                    img, bg_img, model, res,
                    semi_en, semi_str, semi_md, def_en, def_str,
                    roi_en, roi_ev, roi_meta, roi_crop, roi_pad
                ):
                    if img is None:
                        return None, None
                    short = _parse_model_choice(model)
                    engine.load_model(short, (int(res), int(res)))

                    # 将 ImageEditor 的图层合成为全尺寸 ROI 掩码（0/255），失败则忽略
                    roi_mask_full = None
                    try:
                        if bool(roi_en) and roi_ev is not None and roi_meta is not None:
                            roi_mask_full = editor_layers_to_mask_fullres(roi_ev, roi_meta)
                    except Exception:
                        roi_mask_full = None

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

                        # —— ROI 三参数（compose.py 需支持）——
                        roi_mask_fullres=roi_mask_full,
                        roi_crop_before=bool(roi_crop),
                        roi_pad_px=int(roi_pad or 0),
                    )
                    return result, m

                run_btn.click(
                    on_process,
                    inputs=[
                        inp, bg, model_choice, resolution,
                        semi_enable, semi_strength, semi_mode,
                        defringe_enable, defringe_strength,
                        # ROI 相关（顺序要与函数参数一致）
                        roi_enable, roi_canvas, roi_meta_state, roi_crop_before, roi_pad_px,
                    ],
                    outputs=[out, mask],
                    queue=True,              # 启用队列
                    concurrency_limit=1,    # 每个会话/事件同时只跑1个，避免显存抢占
                )

            # ================== 批量 ==================
            with gr.Tab("📁 批量"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["tight"]):
                        files_b = gr.Files(label="选择多张图片", file_count="multiple", type="filepath")
                        bg_b = gr.Image(type="pil", label="背景（可选，留空=>透明）", height=180)

                        with gr.Accordion("⚙️ 模型与分辨率", open=True):
                            model_choices_b = [f"{k} - {v}" for k, v in model_descriptions.items()]
                            if not model_choices_b:
                                model_choices_b = ["General - 通用前景分割模型"]
                            model_choice_b = gr.Dropdown(choices=model_choices_b, value=model_choices_b[0], label="模型")
                            model_hint_b = gr.Markdown(_model_hint_text(model_choices_b[0]), elem_classes=["hint-box"])

                            resolution_b = gr.Slider(256, 2048, step=64, value=1024, label="输入分辨率")
                            res_hint_b = gr.Markdown("", elem_classes=["hint-box"])

                            preload_btn_b = gr.Button("⚡ 预加载当前模型")
                            preload_btn_b.click(_preload_model, inputs=[model_choice_b, resolution_b], outputs=[model_hint_b])

                            resolution_b.input(lambda r: f"批量分辨率：{int(r)}×{int(r)}，显存占用和速度与单图相当。", inputs=resolution_b, outputs=res_hint_b)
                            model_choice_b.change(_model_hint_text, inputs=model_choice_b, outputs=model_hint_b)

                        with gr.Accordion("⚙️ 边缘增强选项（点击展开）", open=False):
                            semi_enable_b = gr.Checkbox(label="半透明边缘", value=False)
                            semi_strength_b = gr.Slider(0.0, 1.0, step=0.05, value=0.5, label="半透明强度")
                            semi_mode_b = gr.Radio(choices=["auto", "暗部优先", "透色优先"], value="auto", label="半透明模式")
                            defringe_enable_b = gr.Checkbox(label="去白边（防渗色）", value=False)
                            defringe_strength_b = gr.Slider(0.0, 1.0, step=0.05, value=0.7, label="去白边力度")

                        run_b = gr.Button("📦 开始批量处理", variant="primary")

                    with gr.Column(scale=5):
                        zip_out = gr.File(label="批量结果 ZIP")
                        gallery = gr.Gallery(label="预览（保存图在 ZIP 内与输出目录）", columns=4, height=320, preview=True)

                def on_batch(files, bg_img, model, res, semi_en, semi_str, semi_md, def_en, def_str, progress=gr.Progress()):
                    if not files:
                        return None, []
                    short = _parse_model_choice(model)
                    engine.load_model(short, (int(res), int(res)))

                    def cb(p, msg):
                        progress(p, desc=msg)

                    zip_path, saved_paths, _msg = process_batch_images(
                        engine,
                        files,
                        background_image=bg_img,
                        input_size=(int(res), int(res)),
                        semi_transparent=bool(semi_en),
                        semi_strength=float(semi_str or 0.5),
                        semi_mode=str(semi_md),
                        remove_white_halo=bool(def_en),
                        defringe_strength=float(def_str or 0.7),
                        progress_cb=cb,
                    )
                    show = [p for p in saved_paths if isinstance(p, str) and not p.startswith("[ERROR]")]
                    return zip_path, show

                run_b.click(
                    on_batch,
                    inputs=[
                        files_b, bg_b, model_choice_b, resolution_b,
                        semi_enable_b, semi_strength_b, semi_mode_b,
                        defringe_enable_b, defringe_strength_b,
                    ],
                    outputs=[zip_out, gallery],
                    queue=True,
                    concurrency_limit=1,
                )

            # ================== 视频 ==================
            with gr.Tab("🎬 视频"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["tight"]):
                        vid_in = gr.Video(label="输入视频", height=280)
                        bg_v = gr.Image(type="pil", label="背景（可选，留空=>透明）", height=160)

                        with gr.Accordion("⚙️ 模型与分辨率", open=True):
                            model_choices_v = [f"{k} - {v}" for k, v in model_descriptions.items()]
                            if not model_choices_v:
                                model_choices_v = ["General - 通用前景分割模型"]
                            model_choice_v = gr.Dropdown(choices=model_choices_v, value=model_choices_v[0], label="模型")
                            model_hint_v = gr.Markdown(_model_hint_text(model_choices_v[0]), elem_classes=["hint-box"])

                            resolution_v = gr.Slider(256, 1536, step=64, value=768, label="输入分辨率（视频建议≤1536）")
                            res_hint_v = gr.Markdown("", elem_classes=["hint-box"])

                            preload_btn_v = gr.Button("⚡ 预加载当前模型")
                            preload_btn_v.click(_preload_model, inputs=[model_choice_v, resolution_v], outputs=[model_hint_v])

                            resolution_v.input(lambda r: f"更高分辨率将显著降低速度。当前：{int(r)}×{int(r)}。", inputs=resolution_v, outputs=res_hint_v)
                            model_choice_v.change(_model_hint_text, inputs=model_choice_v, outputs=model_hint_v)

                        with gr.Accordion("🪄 边缘增强选项（点击展开）", open=False):
                            semi_enable_v = gr.Checkbox(label="半透明边缘", value=False)
                            semi_strength_v = gr.Slider(0.0, 1.0, step=0.05, value=0.5, label="半透明强度")
                            semi_mode_v = gr.Radio(choices=["auto", "暗部优先", "透色优先"], value="auto", label="半透明模式")
                            defringe_enable_v = gr.Checkbox(label="去白边（防渗色）", value=False)
                            defringe_strength_v = gr.Slider(0.0, 1.0, step=0.05, value=0.7, label="去白边力度")

                        run_v = gr.Button("🎬 开始处理视频", variant="primary")

                    with gr.Column(scale=5):
                        vid_out = gr.Video(label="输出视频", height=280)

                def on_video(vpath, bg_img, model, res, semi_en, semi_str, semi_md, def_en, def_str, progress=gr.Progress()):
                    if not vpath:
                        return None
                    short = _parse_model_choice(model)
                    engine.load_model(short, (int(res), int(res)))

                    def cb(p, msg):
                        progress(p, desc=msg)

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
                        progress_cb=cb,
                    )
                    return out_path

                run_v.click(
                    on_video,
                    inputs=[
                        vid_in, bg_v, model_choice_v, resolution_v,
                        semi_enable_v, semi_strength_v, semi_mode_v,
                        defringe_enable_v, defringe_strength_v,
                    ],
                    outputs=[vid_out],
                    queue=True,
                    concurrency_limit=1,
                )

        # ------------- Footer toolbar (bottom) -------------
        with gr.Row(elem_classes=["footer-toolbar"]):
            btn_open_out = gr.Button("📂 打开输出目录")
            btn_open_model = gr.Button("📦 打开模型目录")
            btn_clean_temp = gr.Button("🧼 清理临时缓存")
            btn_clean_out = gr.Button("🧹 清理输出结果")
            btn_clear_safe = gr.Button("🛡️ 清理缓存（安全）")
            btn_clear_full = gr.Button("🧨 清空缓存（彻底）")
            tool_status = gr.Markdown("")

        # 统一使用 handlers 中的实现（打开/清空），与工程内 settings 路径协同
        btn_open_out.click(lambda: _open_dir(PRED_OUTPUT_DIR), outputs=[tool_status])
        btn_open_model.click(lambda: _open_dir(os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "models_local")), outputs=[tool_status])
        btn_clean_temp.click(_clear_temp_cache, outputs=[tool_status])
        btn_clean_out.click(lambda: _clear_dir(PRED_OUTPUT_DIR), outputs=[tool_status])
        btn_clear_safe.click(clear_cache_safe, outputs=[tool_status])
        btn_clear_full.click(clear_cache_full, outputs=[tool_status])

    return demo
