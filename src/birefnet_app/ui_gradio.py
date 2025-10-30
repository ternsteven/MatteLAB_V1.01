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
import os, numpy as np, gradio as gr
import inspect
from PIL import Image
from functools import partial

from .settings import ensure_dirs
from .logging_utils import get_logger
from .engine import BiRefEngine, EngineConfig
from .components.controls import build_semi_controls
from .ops.roi_ops import editor_layers_to_mask_fullres
from .ops.roi_ops import make_editor_thumbnail as _make_editor_thumbnail
from .ops.save_ops import save_result_and_mask
from .adapters.single_image import run_single_image
from .adapters.batch_images import run_batch_images
from .adapters.video import run_single_video, run_batch_videos

logger = get_logger("MatteLAB.UI")
model_descriptions = {"General":"通用版","Portrait":"人像优化","Product":"电商产品","lite-2k":"轻量 2K"}

def _build_engine(default_model="General", size=(1024, 1024), device=None):
    """
    兼容不同版本 EngineConfig / BiRefEngine 的构造签名：
    - 优先把 device 传给 EngineConfig（如果支持）
    - 否则尝试传给 BiRefEngine（如果支持）
    - 再否则用 set_device()/device 属性兜底
    """
    # 1) EngineConfig: 尝试带 device，不行就不带
    try:
        ec_sig = inspect.signature(EngineConfig)
        if device is not None and 'device' in ec_sig.parameters:
            cfg = EngineConfig(default_model, size, device=device)
        else:
            cfg = EngineConfig(default_model, size)
    except Exception:
        cfg = EngineConfig(default_model, size)

    # 2) BiRefEngine: 尝试带 device，不行就不带
    try:
        be_sig = inspect.signature(BiRefEngine)
        if device is not None and 'device' in be_sig.parameters:
            eng = BiRefEngine(cfg, device=device)
        else:
            eng = BiRefEngine(cfg)
    except Exception:
        eng = BiRefEngine(cfg)

    # 3) 兜底 setter（如果类里有）
    try:
        if device is not None:
            if hasattr(eng, "set_device"):
                eng.set_device(device)
            elif hasattr(eng, "device"):
                eng.device = device
    except Exception:
        pass

    return eng


def create_interface():
    ensure_dirs()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = _build_engine("General", (1024, 1024), device=device)

    with gr.Blocks(title="BiRefNet 背景移除工具", theme=gr.themes.Soft(), css=".gradio-container{max-width:1200px;margin:auto;}") as interface:
        gr.Markdown("# 🎯 BiRefNet 背景移除工具\n- 🖼️ 图片/视频/批量处理\n- ⚡ GPU/CPU 自适应")

        with gr.Accordion("⚙️ 模型与分辨率设置", open=True):
            model_choices = [f"{k} - {v}" for k,v in model_descriptions.items()]
            model_choice = gr.Dropdown(label="选择模型任务", choices=model_choices, value=model_choices[0])
            resolution = gr.Slider(label="输入分辨率", minimum=256, maximum=2048, step=64, value=1024)
            resolution_info = gr.Markdown(value="⚙️ 当前输入分辨率：1024×1024\n💨 推理速度：中等（推荐）\n🎯 预估精度：高")
            status_box = gr.Textbox(label="状态", interactive=False, value=f"🧠 设备: {device.upper()}")

            def on_resolution_change(res):
                res = int(res); base_res = 1024; base_mem_gb = 2.5
                est = base_mem_gb * (res/base_res)**2
                if res<=512: s,q,n="🚀 非常快","⚪ 精度较低","适合实时预览或低显存"
                elif res<=1024: s,q,n="⚡ 中等（推荐）","🟢 精度高","适合大多数任务"
                elif res<=1536: s,q,n="🐢 稍慢","🔵 精度更高","适合高质量抠图"
                else: s,q,n="🐌 较慢","🟣 极高精度","适合最高质量输出"
                logger.info(f"🎚️ 分辨率滑块调整为 {res}x{res}，预估显存 {est:.1f} GB")
                return f"⚙️ 当前输入分辨率：{res}×{res}\n{s} · {q}\n🧠 预估显存占用：约 {est:.1f} GB\n💡 {n}"

            def on_model_change(selected):
                short = selected.split(" - ")[0].strip()
                ok = engine.load_model(short, (1024, 1024))
                return f"✅ 模型已加载：{short}" if ok else f"❌ 模型加载失败：{short}"

            model_choice.change(on_model_change, inputs=[model_choice], outputs=[status_box])
            resolution.change(on_resolution_change, inputs=[resolution], outputs=[resolution_info])

            def update_resolution_limit(selected_model):
                if "lite-2k" in str(selected_model).lower():
                    return gr.update(minimum=1024, maximum=2048, value=1024, step=64, label="输入分辨率 (Lite 模型限制 ≥1024)")
                return gr.update(minimum=256, maximum=2048, value=1024, step=64, label="输入分辨率")
            model_choice.change(update_resolution_limit, inputs=model_choice, outputs=resolution)

        with gr.Tabs():
            with gr.Tab("🖼️ 单张图片处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(label="上传图片", type="pil", height=400)
                        background_image = gr.Image(label="背景图片（可选，默认透明背景）", type="pil", height=200)
                        semi_enable_img, semi_strength_img, semi_mode_img = build_semi_controls()
                        defringe_img = gr.Checkbox(label="去白边（自动）", value=False)
                        with gr.Group(visible=False) as defringe_opts_img:
                            defringe_strength_img = gr.Slider(label="去白边力度", minimum=0.0, maximum=1.0, step=0.05, value=0.7)
                        defringe_img.change(lambda on: gr.update(visible=on), inputs=defringe_img, outputs=defringe_opts_img)

                        roi_enable = gr.Checkbox(label="🎯 指定区域（前裁剪）", value=False)
                        with gr.Group(visible=False) as roi_group:
                            with gr.Accordion("高级选项", open=False):
                                with gr.Row():
                                    roi_thumb_side = gr.Slider(label="缩略图长边 (px)", minimum=256, maximum=1200, step=64, value=640)
                                    roi_pad_px = gr.Slider(label="ROI 外扩 padding (px)", minimum=0, maximum=128, step=2, value=16)
                                    roi_crop_before = gr.Checkbox(label="在模型前裁剪（更快/更准）", value=True)
                            roi_canvas = gr.ImageEditor(label="在缩略图上涂抹 ROI（半透明预览）", type="numpy", image_mode="RGBA", height=380, sources=None, layers=True)
                            with gr.Row():
                                roi_clear = gr.Button("清空涂抹", variant="secondary")
                                roi_tips = gr.Markdown("提示：在图上半透明涂抹需要保留的前景；适度涂抹 + padding 即可。")
                        roi_meta_state = gr.State(value=None)
                        def _init_roi_editor(img, long_side):
                            if img is None: return gr.update(), None
                            ev, meta = make_editor_thumbnail(img, int(long_side))
                            thumb = ev["background"].convert("RGBA") if hasattr(ev["background"], "convert") else ev["background"]
                            bg_np = np.array(thumb, dtype=np.uint8)
                            return {"background": bg_np, "layers": [], "composite": bg_np}, meta
                        def _clear_roi_layers(ev):
                            bg = ev.get("background") if isinstance(ev, dict) else None
                            return {"background": bg, "layers": [], "composite": bg}
                        def _on_roi_toggle(enabled, img, long_side):
                            if enabled and img is not None:
                                ev, meta = _init_roi_editor(img, int(long_side))
                                return gr.update(visible=True), ev, meta
                            return gr.update(visible=False), None, None
                        roi_enable.change(_on_roi_toggle, inputs=[roi_enable, input_image, roi_thumb_side], outputs=[roi_group, roi_canvas, roi_meta_state], show_progress=False)
                        def _maybe_refresh_editor(enabled, img, long_side):
                            if not enabled or img is None: return gr.update(), None
                            return _init_roi_editor(img, int(long_side))
                        roi_thumb_side.change(_maybe_refresh_editor, inputs=[roi_enable, input_image, roi_thumb_side], outputs=[roi_canvas, roi_meta_state], show_progress=False)
                        input_image.change(_maybe_refresh_editor, inputs=[roi_enable, input_image, roi_thumb_side], outputs=[roi_canvas, roi_meta_state], show_progress=False)
                        roi_clear.click(_clear_roi_layers, inputs=[roi_canvas], outputs=[roi_canvas])

                        process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="处理结果", height=400, format="png", image_mode="RGBA", show_download_button=True, interactive=False)
                        mask_preview = gr.Image(label="遮罩预览", height=200, format="png", image_mode="RGB", show_download_button=True, interactive=False)
                        status_text = gr.Textbox(label="处理状态", interactive=False)

                evt = process_btn.click(
                    fn=lambda *args: run_single_image(engine, *args),
                    inputs=[input_image, background_image, semi_enable_img, semi_strength_img, semi_mode_img,
                            defringe_img, defringe_strength_img, roi_enable, roi_canvas, roi_meta_state, roi_crop_before, roi_pad_px,
                            model_choice, resolution],
                    outputs=[output_image, mask_preview, status_text], queue=True
                )
                evt.then(save_result_and_mask, inputs=[output_image, mask_preview], outputs=[mask_preview, status_text])

            with gr.Tab("📁 批量图片处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_images = gr.File(label="上传多张图片", file_count="multiple", file_types=["image"])
                        batch_bg_image = gr.Image(label="背景图片（可选，默认透明背景）", type="pil", height=200)
                        semi_enable_bi, semi_strength_bi, semi_mode_bi = build_semi_controls()
                        defringe_bi = gr.Checkbox(label="去白边（自动）", value=False)
                        with gr.Group(visible=False) as defringe_opts_bi:
                            defringe_strength_bi = gr.Slider(label="去白边力度（批量）", minimum=0.0, maximum=1.0, step=0.05, value=0.65)
                        defringe_bi.change(lambda on: gr.update(visible=on), inputs=defringe_bi, outputs=defringe_opts_bi)
                        batch_process_btn = gr.Button("🚀 批量处理", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        batch_output = gr.File(label="下载处理结果（ZIP文件）")
                        batch_status = gr.Textbox(label="处理状态", interactive=False)
                batch_process_btn.click(
                    fn=lambda *args: run_batch_images(engine, *args),
                    inputs=[batch_images, batch_bg_image, semi_enable_bi, semi_strength_bi, semi_mode_bi,
                            defringe_bi, defringe_strength_bi, model_choice, resolution],
                    outputs=[batch_output, batch_status], queue=True
                )
            with gr.Tab("🎬 单个视频"):
                # 输入视频（用 File，拿到真实路径更稳）
                video_input = gr.File(
                    label="输入视频",
                    file_count="single",
                    file_types=[".mp4", ".mov", ".avi", ".mkv"]
                )

                # 只保留“背景颜色”，默认绿色；背景图固定传 None
                video_bg_color = gr.ColorPicker(label="背景颜色", value="#00FF00")
                video_bg_image_none = gr.State(value=None)

                # 半透明控件（默认不展开；勾选后显示）
                semi_enable_v, semi_strength_v, semi_mode_v = build_semi_controls()

                # 输出区 & 按钮
                video_output = gr.Video(label="输出视频")
                video_status = gr.Markdown()
                video_process_btn = gr.Button("开始处理", variant="primary")

                # 事件绑定（保持你原来引擎绑定方式，如果你已有 engine 变量）
                video_process_btn.click(
                    fn=partial(run_single_video, engine),  # 若你的项目不是用 partial，请保持你现有的绑定写法
                    inputs=[
                        video_input,
                        video_bg_image_none,  # 背景图→固定 None
                        video_bg_color,       # 背景颜色
                        semi_enable_v, semi_strength_v, semi_mode_v,
                        model_choice, resolution
                    ],
                    outputs=[video_output, video_status],
                    queue=True
                )
            # —— 批量视频 ——
            with gr.Tab("📦 批量视频"):
                batch_video_files = gr.File(
                    label="选择多个视频",
                    file_count="multiple",
                    file_types=[".mp4", ".mov", ".avi", ".mkv"]
                )

                batch_video_bg_color = gr.ColorPicker(label="背景颜色", value="#00FF00")
                batch_video_bg_image_none = gr.State(value=None)

                semi_enable_bv, semi_strength_bv, semi_mode_bv = build_semi_controls()

                batch_video_output = gr.File(label="输出（ZIP 或单个 MP4）")
                batch_video_status = gr.Markdown()
                batch_video_process_btn = gr.Button("批量处理", variant="primary")

                batch_video_process_btn.click(
                    fn=partial(run_batch_videos, engine),
                    inputs=[
                        batch_video_files,
                        batch_video_bg_image_none,  # 背景图→固定 None
                        batch_video_bg_color,       # 背景颜色
                        semi_enable_bv, semi_strength_bv, semi_mode_bv,
                        model_choice, resolution
                    ],
                    outputs=[batch_video_output, batch_video_status],
                    queue=True
                )
######################
            with gr.Accordion("📂 打开缓存与结果目录", open=False):
                gr.Markdown(
                    "你可以打开或清理缓存与输出文件夹。\n"
                    "💡 建议使用“安全清理”保留离线模型，避免断网后无法加载模型。"
                )

                open_preds = gr.Button("🖼️ 打开抠图结果目录 (preds-BiRefNet)")
                open_weights = gr.Button("🧱 打开离线模型目录 (models_local)")
                output_text = gr.Textbox(label="操作结果", interactive=False)

                def open_folder(path):
                    import subprocess, platform, os
                    abs_path = os.path.abspath(path)
                    os.makedirs(abs_path, exist_ok=True)
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

                open_preds.click(fn=lambda: open_folder("preds-BiRefNet"), outputs=[output_text])
                open_weights.click(fn=lambda: open_folder("models_local"), outputs=[output_text])

                gr.Markdown("### 🧹 缓存清理选项")
                clear_safe_btn = gr.Button("🧹 安全清理 (保留离线模型)", variant="secondary")
                clear_full_btn = gr.Button("🔥 完全清理 (包含模型缓存)", variant="stop")
                ###
                def clear_cache_safe():
                    """
                    安全清理：
                    - 删除项目内所有 __pycache__
                    - 清空 preds-BiRefNet 目录内容（保留目录本身）
                    不清理 models_local，也不删除任何模型权重。
                    """
                    import os, shutil

                    removed = []

                    # 1) 递归删除 __pycache__（跳过 models_local 子树）
                    root = os.path.abspath(".")
                    for dirpath, dirnames, filenames in os.walk(root):
                        # 不进入 models_local
                        if "models_local" in dirnames:
                            dirnames.remove("models_local")
                        if "__pycache__" in dirnames:
                            p = os.path.join(dirpath, "__pycache__")
                            try:
                                shutil.rmtree(p)
                                removed.append(p)
                            except Exception as e:
                                print(f"⚠️ 删除失败 {p}: {e}")

                    # 2) 清空 preds-BiRefNet 内容（保留目录）
                    try:
                        try:
                            from src.birefnet_app.settings import OUT_DIR as _OUT_DIR
                            out_dir = _OUT_DIR or "preds-BiRefNet"
                        except Exception:
                            out_dir = "preds-BiRefNet"

                        out_abs = os.path.abspath(out_dir)
                        if os.path.isdir(out_abs):
                            for name in os.listdir(out_abs):
                                path = os.path.join(out_abs, name)
                                try:
                                    if os.path.isdir(path):
                                        shutil.rmtree(path)
                                    else:
                                        os.remove(path)
                                    removed.append(path)
                                except Exception as e:
                                    print(f"⚠️ 删除失败 {path}: {e}")
                        else:
                            # 若不存在就创建（保持一致的结构）
                            os.makedirs(out_abs, exist_ok=True)

                    except Exception as e:
                        print(f"⚠️ 清空 {out_dir} 失败: {e}")

                    return "✅ 已清理以下路径：\n" + ("\n".join(removed) if removed else "（无可清理项）")

                def clear_cache_full():
                    import shutil, os
                    cleared = []
                    for path in ["weights", "preds-BiRefNet", "models_local", "__pycache__"]:
                        if os.path.exists(path):
                            try: shutil.rmtree(path); cleared.append(path)
                            except Exception as e: print(f"⚠️ 删除失败 {path}: {e}")
                    return "🧨 已彻底清理:\n" + "\n".join(cleared) if cleared else "ℹ️ 未发现可清理缓存。"

                clear_safe_btn.click(fn=clear_cache_safe, outputs=[output_text])
                clear_full_btn.click(fn=clear_cache_full, outputs=[output_text])
                            
######################
        return interface

