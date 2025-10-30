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

import os, time, traceback
import shutil
import numpy as np
import gradio as gr
from PIL import Image


from datetime import datetime
from PIL import ImageDraw
from src.birefnet_app.settings import ensure_dirs
from src.birefnet_app.settings import PRED_OUTPUT_DIR

# ---- backends ----
from .engine import BiRefEngine, EngineConfig
from .config_models import model_descriptions
from .batch import process_batch_images as _core_process_batch_images
from .video import process_single_video as _core_process_single_video
from .settings import PRED_OUTPUT_DIR, ensure_dirs
from .ops.image_io import save_image_safe, force_png_path
from .ops.roi_ops import make_editor_thumbnail as _make_editor_thumbnail, editor_layers_to_mask_fullres
from .ui.handlers import open_dir as _open_dir, clear_dir as _clear_dir, clear_cache_safe, clear_cache_full


#####定义ui组件####

# ==== 半透明/去白边控制组====
def build_semi_controls():
    with gr.Row():
        semi_enable = gr.Checkbox(
            label="半透明扣除", value=False,
            info="对玻璃/纱帘等半透明区域做透射估计"
        )
        semi_strength = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.05, value=0.5,
            label="半透明强度", info="越大越透明"
        )
        semi_mode = gr.Dropdown(
            label="处理模式",
            choices=["balanced", "aggressive", "conservative"],
            value="balanced",
            info="平衡/激进/保守"
        )
    return semi_enable, semi_strength, semi_mode

# ==== 统一图像类型（避免 ndarray.save 报错 & 确保下载按钮可下载）====
def _to_pil_rgba(x):
    if isinstance(x, Image.Image):
        return x if x.mode == "RGBA" else x.convert("RGBA")
    arr = np.asarray(x)
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    if arr.ndim == 3:
        im = Image.fromarray(arr.astype(np.uint8))
        return im if im.mode == "RGBA" else im.convert("RGBA")
    return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

def _to_pil_mask_rgb(x):
    if isinstance(x, Image.Image):
        return x if x.mode == "RGB" else x.convert("RGB")
    arr = np.asarray(x)
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if arr.ndim == 3:
        im = Image.fromarray(arr.astype(np.uint8))
        return im if im.mode == "RGB" else im.convert("RGB")
    return Image.new("RGB", (1, 1), (0, 0, 0))

# ==== 处理完成后保存前景与蒙版（与你的 evt.then 兼容：返回 [mask_img, status_md]）====
def _post_save_and_stamp(result_img, mask_img):
    result_img = _to_pil_rgba(result_img)
    mask_img   = _to_pil_mask_rgb(mask_img)
    os.makedirs("outputs/masks", exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path  = os.path.join("outputs",       f"cutout_{ts}.png")
    mask_path = os.path.join("outputs/masks", f"mask_{ts}.png")
    try:
        save_image_safe(result_img, out_path)     # RGBA → PNG
        save_image_safe(mask_img,   mask_path)    # RGB  → PNG
        md = f"💾 已保存：`{os.path.basename(out_path)}`；蒙版：`{os.path.basename(mask_path)}`"
    except Exception as e:
        md = f"⚠️ 保存失败：{e}"
    return mask_img, md
######
# ==== 单图处理：与你的 process_btn.click 绑定形参一一对应 ====
def process_image_with_settings(
    img, bg_img,
    semi_en, semi_str, semi_md,
    def_en, def_str,
    roi_en, roi_ev, roi_meta, roi_crop, roi_pad,
    selected_model, res_value,        # ← 我们会在绑定里把 model_choice、resolution 追加传入
    engine: BiRefEngine               # ← 由 create_interface 内部闭包注入
):
    # 保证三输出
    if img is None:
        return Image.new("RGBA", (1,1), (0,0,0,0)), Image.new("RGB",(1,1),(0,0,0)), "⚠️ 请先上传图片"
    try:
        short = str(selected_model).split(" - ")[0].strip() if selected_model else "General"
        res   = int(res_value)

        # 确保模型已就绪（若用户没点“加载模型”也能跑）
        engine.load_model(short, (res, res))

        # ROI：由画板缩略图 + meta 还原全尺寸 0/255 掩码
        roi_mask_full = None
        if bool(roi_en) and (roi_ev is not None) and (roi_meta is not None):
            try:
                roi_mask_full = editor_layers_to_mask_fullres(roi_ev, roi_meta)
            except Exception:
                roi_mask_full = None

        result, mask = engine.apply_background_replacement(
            image=img,
            background_image=bg_img,
            model_name=short,
            input_size=(res, res),
            semi_transparent=bool(semi_en),
            semi_strength=float(semi_str or 0.5),
            semi_mode=str(semi_md),
            remove_white_halo=bool(def_en),
            defringe_strength=float(def_str or 0.7),
            roi_mask_fullres=roi_mask_full,
            roi_crop_before=bool(roi_crop),
            roi_pad_px=int(roi_pad or 0),
        )

        result = _to_pil_rgba(result if result is not None else Image.new("RGBA", (1,1), (0,0,0,0)))
        mask   = _to_pil_mask_rgb(mask   if mask   is not None else Image.new("RGB",  (1,1), (0,0,0)))
        w, h = result.size
        return result, mask, f"✅ 完成（模型：{short}，输入：{res}×{res}，输出：{w}×{h}）"
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return Image.new("RGBA",(1,1),(0,0,0,0)), Image.new("RGB",(1,1),(0,0,0)), f"❌ 处理失败：{e}\n```\n{tb}\n```"

######
# ==== 批量图片：UI 适配器（把 engine 放在第 1 个参数）====
def process_batch_images_adapter(
    files, bg_image,
    semi_enable, semi_strength, semi_mode,
    defringe_enable, defringe_strength,
    selected_model, res_value,
    engine: BiRefEngine
):
    try:
        short = str(selected_model).split(" - ")[0].strip() if selected_model else "General"
        res   = int(res_value)
        engine.load_model(short, (res, res))

        zip_path, saved_list, status = _core_process_batch_images(
            engine, files,
            background_image=bg_image,
            input_size=(res, res),
            semi_transparent=bool(semi_enable),
            semi_strength=float(semi_strength or 0.5),
            semi_mode=str(semi_mode),
            remove_white_halo=bool(defringe_enable),
            defringe_strength=float(defringe_strength or 0.7),
        )
        return zip_path, status
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return None, f"❌ 批量图片处理失败：{e}\n```\n{tb}\n```"

# ==== 单/批视频：用你已有的 process_single_video 包一层 ====
def process_video_adapter(
    input_video, bg_image, bg_color,
    semi_enable_v, semi_strength_v, semi_mode_v,
    selected_model, res_value,
    engine: BiRefEngine
):
    if _core_process_single_video is None:
        return None, "❌ 未找到视频处理函数：video.process_single_video"
    try:
        short = str(selected_model).split(" - ")[0].strip() if selected_model else "General"
        res   = int(res_value)
        engine.load_model(short, (res, res))
        return _core_process_single_video(
            engine, input_video,
            background_image=bg_image,
            input_size=(res, res),
            semi_transparent=bool(semi_enable_v),
            semi_strength=float(semi_strength_v or 0.5),
            semi_mode=str(semi_mode_v),
        )
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return None, f"❌ 视频处理失败：{e}\n```\n{tb}\n```"
def process_batch_videos_adapter(
    files, bg_image, res_value, bg_color,
    semi_enable_bv, semi_strength_bv, semi_mode_bv,
    selected_model,
    engine: BiRefEngine
):
    if _core_process_single_video is None:
        return None, "❌ 未找到视频处理函数：video.process_single_video"
    try:
        short = str(selected_model).split(" - ")[0].strip() if selected_model else "General"
        res   = int(res_value)
        engine.load_model(short, (res, res))

        # 逐个视频处理并打包
        outs = []
        for f in (files or []):
            out_path, st = _core_process_single_video(
                engine, f,
                background_image=bg_image,
                input_size=(res, res),
                semi_transparent=bool(semi_enable_bv),
                semi_strength=float(semi_strength_bv or 0.5),
                semi_mode=str(semi_mode_bv),
            )
            if out_path:
                outs.append(out_path)

        if not outs:
            return None, "⚠️ 没有生成可下载的视频文件"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        zip_path = os.path.join("outputs", f"videos_{stamp}.zip")
        os.makedirs("outputs", exist_ok=True)
        import zipfile
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in outs:
                z.write(p, arcname=os.path.basename(p))
        return zip_path, f"✅ 批量视频处理完成：{len(outs)} 个文件"
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return None, f"❌ 批量视频处理失败：{e}\n```\n{tb}\n```"
        

def _make_editor_thumbnail(img: Image.Image, long_side: int):
    """按长边生成缩略图，返回 editor_value 所需的 background 及 meta"""
    w, h = img.size
    if max(w, h) == 0:
        thumb = img.copy()
        meta = {"full_size": (w, h), "thumb_size": thumb.size, "scale": 1.0}
        return {"background": thumb, "layers": [], "composite": thumb}, meta
    scale = float(long_side) / float(max(w, h))
    if scale >= 1.0:
        thumb = img.copy()
        scale = 1.0
    else:
        thumb = img.resize((int(round(w*scale)), int(round(h*scale))), Image.LANCZOS)
    meta = {"full_size": (w, h), "thumb_size": thumb.size, "scale": scale}
    return {"background": thumb, "layers": [], "composite": thumb}, meta


def _np_from_any(x):
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return np.array(x)
    return np.asarray(x)

def editor_layers_to_mask_fullres(editor_value, meta, thr: int = 5):
    """
    根据 editor_value 生成全分辨率二值掩码（0/255 的 numpy.uint8）
    1) 优先从 composite 与 background 做差分
    2) 没有 composite 时，尝试 layers 聚合
    """
    thumb_w, thumb_h = tuple(meta.get("thumb_size", (0, 0)))
    full_w, full_h = tuple(meta.get("full_size", (0, 0)))
    if not thumb_w or not thumb_h or not full_w or not full_h:
        return None

    # 兼容 dict / pydantic
    bg = getattr(editor_value, "background", None)
    if bg is None and isinstance(editor_value, dict):
        bg = editor_value.get("background")
    comp = getattr(editor_value, "composite", None)
    if comp is None and isinstance(editor_value, dict):
        comp = editor_value.get("composite")
    layers = getattr(editor_value, "layers", None)
    if layers is None and isinstance(editor_value, dict):
        layers = editor_value.get("layers", [])

    bg_np   = _np_from_any(bg)
    comp_np = _np_from_any(comp)

    mask_thumb = None
    if comp_np is not None and bg_np is not None:
        a = comp_np[..., :3].astype(np.int16)
        b = bg_np[..., :3].astype(np.int16)
        diff = np.abs(a - b).mean(axis=2)
        mask_thumb = (diff > thr).astype(np.uint8) * 255

    # 兜底：从图层生成
    if mask_thumb is None and isinstance(layers, (list, tuple)) and len(layers) > 0:
        acc = np.zeros((thumb_h, thumb_w), dtype=np.uint8)
        for L in layers:
            # 兼容 dict / pydantic
            m = getattr(L, "mask", None)
            if m is None and isinstance(L, dict):
                m = L.get("mask")
            if m is None:
                # 有的版本没有 mask，可尝试基于该层图像差分
                Limg = getattr(L, "image", None)
                if Limg is None and isinstance(L, dict):
                    Limg = L.get("image")
                if Limg is not None:
                    L_np = _np_from_any(Limg)[..., :3].astype(np.int16)
                    diffL = np.abs(L_np - bg_np[..., :3].astype(np.int16)).mean(axis=2)
                    m = (diffL > thr).astype(np.uint8) * 255
            if m is not None:
                m_np = _np_from_any(m)
                if m_np.ndim == 3:
                    m_np = m_np[..., 0]
                acc = np.maximum(acc, (m_np > 0).astype(np.uint8) * 255)
        mask_thumb = acc

    if mask_thumb is None:
        return None

    # 放大回全尺寸
    mask_img = Image.fromarray(mask_thumb, mode="L")
    mask_full = mask_img.resize((full_w, full_h), Image.NEAREST)
    return np.array(mask_full, dtype=np.uint8)

#####定义ui组件结束####
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

def _parse_model_choice(selected):
    # 兼容 “General - 通用版” 这类格式
    if not selected:
        return "General"
    return str(selected).split(" - ")[0].strip()


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

import os
import time


def create_interface():
    """创建Gradio界面"""
    engine = BiRefEngine(EngineConfig("General", (1024, 1024)))
    with gr.Blocks(
        title="BiRefNet 背景移除工具",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        .tab-nav button {
            color: white !important;
            font-weight: bold !important;
        }
        .tab-nav button.selected {
            background: rgba(255,255,255,0.2) !important;
        }
        """
    ) as interface:
        
        gr.Markdown(
            """
            # 🎯 BiRefNet 背景移除工具
            
            **功能特点：**
            - 🖼️ 支持单张图片和批量图片处理
            - 🎬 支持单个视频和批量视频处理
            - 🎨 支持自定义背景图片或默认透明背景
            - 📦 批量处理结果自动打包下载
            - ⚡ 高性能GPU加速推理
           
            
            **使用说明：** 上传图片或视频，可选择背景图片，系统将自动移除原背景并替换为指定背景（默认绿色）。
            """
        )
                # ===== 修改开始：新增模型与分辨率设置UI =====
        # ===== 简化后的模型与分辨率设置 =====
        with gr.Accordion("⚙️ 模型与分辨率设置", open=True):
            # === 模型下拉框：显示备注 ===
            # 构建带描述的可视化选项
            model_choices = [f"{key} - {desc}" for key, desc in model_descriptions.items()]

            model_choice = gr.Dropdown(
                label="选择模型任务",
                choices=model_choices,
                value=model_choices[0],
                info="选择适合任务的模型，系统会自动加载对应权重"
            )

            resolution = gr.Slider(
                label="输入分辨率",
                minimum=256,
                maximum=2048,
                step=64,
                value=1024,
                info="设置模型推理输入分辨率"
            )
            resolution_info = gr.Markdown(
                value="⚙️ 当前输入分辨率：1024×1024\n💨 推理速度：中等（推荐）\n🎯 预估精度：高",
                label="分辨率性能提示"
            )

            status_box = gr.Textbox(label="状态", interactive=False)

            def on_model_change(selected_model):
                print(f"🪄 用户选择了模型：{selected_model}")
                status = "正在加载模型，请稍候..."
                ok = load_model(selected_model, (1024, 1024))
                if ok:
                    status = f"✅ 模型已加载：{selected_model}"
                else:
                    status = f"❌ 模型加载失败：{selected_model}"
                return status
            def on_resolution_change(res):
                """根据滑块值动态提示性能、精度与显存预估"""
                res = int(res)
                # 估算显存消耗（经验值）
                base_res = 1024
                base_mem_gb = 2.5  # 在 RTX3090 上 1024×1024 大约占 2.5 GB
                estimated_mem = base_mem_gb * (res / base_res) ** 2

                # 设置性能描述
                if res <= 512:
                    speed = "🚀 非常快"
                    quality = "⚪ 精度较低"
                    note = "适合实时预览或低显存设备"
                elif res <= 1024:
                    speed = "⚡ 中等（推荐）"
                    quality = "🟢 精度高"
                    note = "适合大多数任务"
                elif res <= 1536:
                    speed = "🐢 稍慢"
                    quality = "🔵 精度更高"
                    note = "适合高质量抠图"
                else:
                    speed = "🐌 较慢"
                    quality = "🟣 极高精度"
                    note = "适合静态图片的最高质量输出"

                msg = (
                    f"⚙️ 当前输入分辨率：{res}×{res}\n"
                    f"{speed} · {quality}\n"
                    f"🧠 预估显存占用：约 {estimated_mem:.1f} GB\n"
                    f"💡 {note}"
                )

                print(f"🎚️ 分辨率滑块调整为 {res}x{res}，预估显存 {estimated_mem:.1f} GB")
                return msg
            ###
            def on_model_change(selected):
                short_name = selected.split(" - ")[0].strip()
                status = f"正在加载模型 {short_name} ..."
                ok = engine.load_model(short_name, (1024, 1024))   # ← 用 engine
                return f"✅ 模型加载成功：{short_name}" if ok else f"❌ 模型加载失败：{short_name}"


            model_choice.change(
                fn=on_model_change,
                inputs=[model_choice],
                outputs=[status_box]
            )
            resolution.change(
                fn=on_resolution_change,
                inputs=[resolution],
                outputs=[resolution_info]
            )
            def update_resolution_limit(selected_model):
                """
                根据选择的模型动态限制分辨率范围。
                Lite 模型在低于 1024 分辨率下表现不稳定。
                """
                min_res, max_res = 256, 2048
                default_value = 1024

                if "lite-2K" in str(selected_model):
                    min_res = 1024
                    print(f"⚠️ {selected_model} 模型仅支持分辨率 >=1024，已调整滑块下限")
                    return gr.update(
                        minimum=min_res,
                        maximum=max_res,
                        value=max(default_value, min_res),
                        step=64,
                        label="输入分辨率 (Lite 模型限制 ≥1024)"
                    )
                else:
                    return gr.update(
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=64,
                        label="输入分辨率"
                    )

            # 绑定模型选择变化时的滑块更新
            model_choice.change(
                fn=update_resolution_limit,
                inputs=model_choice,
                outputs=resolution
            )
        # ===== 修改结束 =====

        with gr.Tabs():
            # 单张图片处理标签页
            with gr.Tab("🖼️ 单张图片处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="上传图片",
                            type="pil",
                            height=400
                        )
                        
                        background_image = gr.Image(
                            label="背景图片（可选，默认透明背景）",
                            type="pil",
                            height=200
                        )
                        ##半透明切换按钮###
                        # —— 半透明扣除：开关/滑块/模式 + 折叠说明（复用一套） ——
                        semi_enable_img, semi_strength_img, semi_mode_img = build_semi_controls()

                        # 去白边开关（自动消除 1–2 px 白色毛边）
                        defringe_img = gr.Checkbox(
                            label="去白边（自动消除 1–2 px 白色毛边）",
                            value=False,
                            info="轻微收缩边缘并回灌前景色，减少白色毛边。"
                        )
                        # —— 去白边力度滑杆（默认隐藏；勾选后显示）——
                        with gr.Group(visible=False) as defringe_opts_img:
                            defringe_strength_img = gr.Slider(
                                label="去白边力度",
                                minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                                info="推荐：人像 0.6–0.85；白底可到 0.9–1.0（更强收边）。高分辨率下会自适应放大侵蚀核。"
                            )

                        # 勾选联动：显示/隐藏力度滑杆
                        defringe_img.change(
                            fn=lambda on: gr.update(visible=on),
                            inputs=defringe_img,
                            outputs=defringe_opts_img
                        )
#############################绘画涂抹#####################################
                        # === ROI 画板 UI（新版） ===
                        roi_enable = gr.Checkbox(
                            label="🎯 指定区域（在进入模型前裁剪并对齐回原图）",
                            value=False,
                            info="开启后只对你圈定/涂抹的区域做抠图，其他区域保持背景"
                        )

                        with gr.Group(visible=False) as roi_group:
                            # 默认收起的高级选项
                            with gr.Accordion("高级选项", open=False):
                                with gr.Row():
                                    roi_thumb_side = gr.Slider(
                                        label="缩略图长边 (px)",
                                        minimum=256, maximum=1200, step=64, value=640,
                                        info="只影响画板显示与交互，不影响最终分辨率"
                                    )
                                    roi_pad_px = gr.Slider(
                                        label="ROI 外扩 padding (px)",
                                        minimum=0, maximum=128, step=2, value=16,
                                        info="先裁剪再分割时的安全边，越大越保守、速度稍慢"
                                    )
                                    roi_crop_before = gr.Checkbox(
                                        label="在模型前裁剪（更快/更准）",
                                        value=True
                                    )

                            # 半透明画笔（默认 45% 不透明度），颜色固定为一组半透明色
                            roi_canvas = gr.ImageEditor(
                                label="在缩略图上用画笔涂抹 ROI（半透明预览，不影响结果）",
                                type="numpy", image_mode="RGBA", height=380, sources=None, layers=True,
                                brush=gr.Brush(
                                    default_size="auto",
                                    colors=["#ff9800", "#1e88e5", "#43a047", "#e53935", "#ffffff"],
                                    default_color="#ff9800",
                                    color_mode="fixed"
                                ),
                            )

                            with gr.Row():
                                roi_clear = gr.Button("清空涂抹", variant="secondary")
                                roi_tips = gr.Markdown(
                                    "提示：选择画笔后在图上**半透明**涂抹要保留的前景区域；无需涂满，适当涂抹 + padding 即可。"
                                )

                        roi_meta_state = gr.State(value=None)   # 记录缩略图/原图尺寸

                        ####
                        # === 工具：初始化画板（返回 numpy RGBA 背景，匹配 type="numpy"） ===
                        def _init_roi_editor(img: Image.Image | None, long_side: int, overlay_color=(255, 152, 0), overlay_alpha=0.45):
                            if img is None:
                                return gr.update(), None
                            ev, meta = _make_editor_thumbnail(img, int(long_side))
                            thumb = ev["background"].convert("RGBA") if hasattr(ev["background"], "convert") else ev["background"]
                            bg_np = np.array(thumb, dtype=np.uint8)

                            # 生成半透明预览（此时还没图层，先把 composite = 背景）
                            editor_value = {"background": bg_np, "layers": [], "composite": bg_np}
                            return editor_value, meta

                        # 清空：仅清图层，保留背景，避免变成白底看不到原图
                        def _clear_roi_layers(editor_value):
                            bg = editor_value.get("background") if isinstance(editor_value, dict) else None
                            return {"background": bg, "layers": [], "composite": bg}

                        # 开关勾选 → 自动显示/隐藏 + 自动初始化画板（相当于“默认点击启动”）
                        def _on_roi_toggle(enabled, img, long_side):
                            if enabled and img is not None:
                                ev, meta = _init_roi_editor(img, int(long_side))
                                return gr.update(visible=True), ev, meta
                            else:
                                # 关掉时隐藏并清空
                                return gr.update(visible=False), None, None

                        roi_enable.change(
                            _on_roi_toggle,
                            inputs=[roi_enable, input_image, roi_thumb_side],
                            outputs=[roi_group, roi_canvas, roi_meta_state],
                            show_progress=False
                        )

                        # 改缩略图长边 → 自动刷新（仅在已启用时）
                        def _maybe_refresh_editor(enabled, img, long_side):
                            if not enabled or img is None:
                                return gr.update(), None
                            return _init_roi_editor(img, int(long_side))

                        roi_thumb_side.change(
                            _maybe_refresh_editor,
                            inputs=[roi_enable, input_image, roi_thumb_side],
                            outputs=[roi_canvas, roi_meta_state],
                            show_progress=False
                        )

                        # 更换输入图 → 自动刷新（仅在已启用时）
                        input_image.change(
                            _maybe_refresh_editor,
                            inputs=[roi_enable, input_image, roi_thumb_side],
                            outputs=[roi_canvas, roi_meta_state],
                            show_progress=False
                        )

                        # 清空涂抹（保留背景）
                        roi_clear.click(_clear_roi_layers, inputs=[roi_canvas], outputs=[roi_canvas])

#################################绘画涂抹########################################
                        process_btn = gr.Button(
                            "🚀 开始处理",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="处理结果",
                            height=400,
                            format="png",
                            image_mode="RGBA",
                            show_download_button=True,   # ← 新增
                            interactive=False            # ← 新增
                        )
                        mask_preview = gr.Image(
                            label="遮罩预览",
                            height=200,
                            format="png",
                            image_mode="RGB",
                            show_download_button=True,   # ← 新增
                            interactive=False            # ← 新增
                        )
                        status_text = gr.Textbox(
                            label="处理状态",
                            interactive=False
                        )
                
                # 绑定处理函数
                evt = process_btn.click(
                    fn=lambda *args: process_image_with_settings(*args, engine=engine),
                    inputs=[
                        input_image, background_image,
                        semi_enable_img, semi_strength_img, semi_mode_img,
                        defringe_img, defringe_strength_img,
                        roi_enable, roi_canvas, roi_meta_state, roi_crop_before, roi_pad_px,
                        model_choice, resolution,                # ← 新增
                    ],
                    outputs=[output_image, mask_preview, status_text]
                )
                evt.then(_post_save_and_stamp, inputs=[output_image, mask_preview], outputs=[mask_preview, status_text])

            # 批量图片处理标签页
            with gr.Tab("📁 批量图片处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_images = gr.File(
                            label="上传多张图片",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        batch_bg_image = gr.Image(
                            label="背景图片（可选，默认透明背景）",
                            type="pil",
                            height=200
                        )
                        # ===== 批量图片处理 Tab =====
                        semi_enable_bi, semi_strength_bi, semi_mode_bi = build_semi_controls()

                        defringe_bi = gr.Checkbox(
                            label="去白边（自动）",
                            value=False,
                            info="批量图片去白边。"
                        )
                        with gr.Group(visible=False) as defringe_opts_bi:
                            defringe_strength_bi = gr.Slider(
                                label="去白边力度（批量）",
                                minimum=0.0, maximum=1.0, step=0.05, value=0.65,
                                info="推荐：0.55–0.8 兼顾速度与质量；>0.9 为激进模式（更强收边）。高分辨率自适应放大。"
                            )

                        defringe_bi.change(
                            fn=lambda on: gr.update(visible=on),
                            inputs=defringe_bi,
                            outputs=defringe_opts_bi
                        )
                        batch_process_btn = gr.Button(
                            "🚀 批量处理",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        batch_output = gr.File(
                            label="下载处理结果（ZIP文件）"
                        )
                        
                        batch_status = gr.Textbox(
                            label="处理状态",
                            interactive=False
                        )
                
                # 绑定批量处理函数
                batch_process_btn.click(
                    fn=lambda *args: process_batch_images_adapter(*args, engine=engine),
                    inputs=[
                        batch_images, batch_bg_image,
                        semi_enable_bi, semi_strength_bi, semi_mode_bi,
                        defringe_bi, defringe_strength_bi,
                        model_choice, resolution,                # ← 新增
                    ],
                    outputs=[batch_output, batch_status]
                )

            # 单个视频处理标签页
            with gr.Tab("🎬 单个视频处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_video = gr.Video(
                            label="上传视频",
                            height=300
                        )
                        
                        video_bg_image = gr.Image(
                            label="背景图片（可选，默认绿色背景）",
                            type="pil",
                            height=200
                        )
                        video_bg_color = gr.ColorPicker(
                            label="背景颜色（未上传图片时生效）",
                            value="#00FF00"
                        )
                        # ===== 单个视频处理 Tab =====

                        semi_enable_v, semi_strength_v, semi_mode_v = build_semi_controls()

                        video_process_btn = gr.Button(
                            "🚀 开始处理",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        output_video = gr.Video(
                            label="处理结果",
                            height=300
                        )
                        
                        video_status = gr.Textbox(
                            label="处理状态",
                            interactive=False
                        )
                
                # 绑定视频处理函数（✅ 多传两个新参数）
                video_process_btn.click(
                    fn=lambda *args: process_video_adapter(*args, engine=engine),
                    inputs=[input_video, video_bg_image, video_bg_color,
                            semi_enable_v, semi_strength_v, semi_mode_v,
                            model_choice, resolution],          # ← 新增
                    outputs=[output_video, video_status]
                )

            # 批量视频处理标签页
            with gr.Tab("📹 批量视频处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_videos = gr.File(
                            label="上传多个视频",
                            file_count="multiple",
                            file_types=["video"]
                        )
                        
                        batch_video_bg_image = gr.Image(
                            label="背景图片（可选，默认绿色背景）",
                            type="pil",
                            height=200
                        )

                        batch_video_bg_color = gr.ColorPicker(
                            label="背景颜色（未上传图片时生效）",
                            value="#00FF00"
                        )
                        # ===== 批量视频处理 Tab =====

                        semi_enable_bv, semi_strength_bv, semi_mode_bv = build_semi_controls()

                        batch_video_process_btn = gr.Button(
                            "🚀 批量处理",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        batch_video_output = gr.File(
                            label="下载处理结果（ZIP文件）"
                        )
                        
                        batch_video_status = gr.Textbox(
                            label="处理状态",
                            interactive=False
                        )
                
                # 绑定批量视频处理函数
                batch_video_process_btn.click(
                    fn=lambda *args: process_batch_videos_adapter(*args, engine=engine),
                    inputs=[batch_videos, batch_video_bg_image, resolution, batch_video_bg_color,
                            semi_enable_bv, semi_strength_bv, semi_mode_bv,
                            model_choice],                      # ← 新增
                    outputs=[batch_video_output, batch_video_status]
                )
            # （已移除：模型训练标签页）
            # （已移除：配置调整标签页）
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown(
                """
                ### 🔧 功能说明
                
                1. **单张图片处理**：上传一张图片，可选择背景图片，系统自动移除背景
                2. **批量图片处理**：同时上传多张图片进行批量处理，结果打包为ZIP文件
                3. **视频处理**：支持单个和批量视频处理，逐帧移除背景
                4. **背景选择**：可上传自定义背景图片，或使用默认背景
                
                
                
                ### ⚡ 性能优化
                
                - 使用GPU加速推理（如果可用）
                - 支持半精度计算提升速度
                - 批量处理自动优化内存使用
                
                ### 📝 注意事项
                
                - 支持常见图片格式：JPG, PNG, WEBP等
                - 支持常见视频格式：MP4, AVI, MOV等
                - 视频处理需要较长时间，请耐心等待
                - 批量处理结果会自动打包为ZIP文件供下载
                - 训练功能需要准备好的数据集
                - 配置修改会自动创建备份文件
                """
            )
        with gr.Accordion("📂 打开缓存与结果目录", open=False):
            gr.Markdown(
                "你可以打开或清理缓存与输出文件夹。"
                "\n💡 建议使用“安全清理”保留离线模型，避免断网后无法加载模型。"
            )

            # === 打开目录按钮 ===
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

            # === 清理缓存按钮 ===
            gr.Markdown("### 🧹 缓存清理选项")

            clear_safe_btn = gr.Button("🧹 安全清理 (保留离线模型)", variant="secondary")
            clear_full_btn = gr.Button("🔥 完全清理 (包含模型缓存)", variant="stop")

            def clear_cache_safe():
                """安全清理：保留离线模型，仅删除缓存和结果"""
                import shutil, os
                cleared = []

                # 1️⃣ 清理推理结果和临时缓存
                for path in ["weights", "preds-BiRefNet", "__pycache__"]:
                    if os.path.exists(path):
                        try:
                            shutil.rmtree(path)
                            cleared.append(path)
                        except Exception as e:
                            print(f"⚠️ 删除失败 {path}: {e}")

                # 2️⃣ 清理 HuggingFace 缓存目录但保留离线模型
                models_local = "models_local"
                if os.path.exists(models_local):
                    subdirs = os.listdir(models_local)
                    deletable = []
                    for d in subdirs:
                        full_path = os.path.join(models_local, d)
                        # 删除 huggingface 缓存目录（models-- 开头）
                        if d.startswith("models--"):
                            deletable.append(full_path)
                    for path in deletable:
                        try:
                            shutil.rmtree(path)
                            cleared.append(path)
                        except Exception as e:
                            print(f"⚠️ 删除失败 {path}: {e}")

                if cleared:
                    return "✅ 已清理以下目录（保留离线模型）:\n" + "\n".join(cleared)
                else:
                    return "ℹ️ 未发现可清理缓存。"

            def clear_cache_full():
                """完全清理：包括模型缓存"""
                import shutil, os
                cleared = []
                for path in ["weights", "preds-BiRefNet", "models_local", "__pycache__"]:
                    if os.path.exists(path):
                        try:
                            shutil.rmtree(path)
                            cleared.append(path)
                        except Exception as e:
                            print(f"⚠️ 删除失败 {path}: {e}")
                if cleared:
                    return "🧨 已彻底清理以下目录（模型缓存已删除）:\n" + "\n".join(cleared)
                else:
                    return "ℹ️ 未发现可清理缓存。"

            clear_safe_btn.click(fn=clear_cache_safe, outputs=[output_text])
            clear_full_btn.click(fn=clear_cache_full, outputs=[output_text])

    return interface

