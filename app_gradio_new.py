import warnings; warnings.filterwarnings("ignore", message="Importing from timm", category=FutureWarning)



# app_gradio_new.py
from src.birefnet_app.ui_gradio import create_interface
from src.birefnet_app.settings import ensure_dirs
from src.birefnet_app.ops.image_io import (
    load_image_safe,
    force_png_path as _force_png_path,
    save_image_safe as _save_image_safe,
    resize_keep_ratio,
    preprocess_image,
)
from src.birefnet_app.ops.bg_ops import (
    hex_to_rgb,
    resize_bg_keep_aspect as _resize_bg_keep_aspect,
    create_background,
    replace_background_with_mask,
    create_transparent_result,
)
from src.birefnet_app.ops.mask_ops import to_binary_mask
# 如果旧代码还在用 _to_binary_mask(...)，可加：
_to_binary_mask = to_binary_mask


from src.birefnet_app.ops.roi_ops import (
    to_single_channel_uint8 as _to_single_channel_uint8,
    bbox_from_mask as _bbox_from_mask,
)



if __name__ == "__main__":
    ensure_dirs()
    demo = create_interface()
    demo.launch(server_name="127.0.0.1", share=False)
