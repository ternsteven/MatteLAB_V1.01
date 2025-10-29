import os, cv2, numpy as np, torch, requests
from PIL import Image
import torchvision.transforms as T
from transformers import AutoModelForImageSegmentation

from .config_models import usage_to_weights_file
from .ops.image_io import preprocess_image
from .ops.mask_ops import to_binary_mask, estimate_soft_alpha_inside_mask, refine_alpha_with_channel
from .ops import bg_ops as bg

class EngineConfig:
    def __init__(self, model_name="General", input_size=(1024,1024)):
        self.model_name = model_name
        self.input_size = input_size

class BiRefEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_model_name = None

    def _online_ok(self):
        try:
            return requests.get("https://huggingface.co", timeout=3).status_code == 200
        except Exception:
            return False

    # 载入/切换模型
    def load_model(self, model_name:str, input_size):
        repo = usage_to_weights_file.get(model_name, model_name)
        hf_repo = f"zhengpeng7/{repo}"
        cache_dir = os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "models_local")
        os.makedirs(cache_dir, exist_ok=True)

        if self.model is not None and self.current_model_name==model_name:
            self.cfg.input_size = input_size; return

        if self._online_ok():
            self.model = AutoModelForImageSegmentation.from_pretrained(hf_repo, trust_remote_code=True, cache_dir=cache_dir)
        else:
            local = os.path.join(cache_dir, repo)
            self.model = AutoModelForImageSegmentation.from_pretrained(local, trust_remote_code=True)

        self.model.to(self.device).eval()
        self.current_model_name = model_name
        self.cfg.input_size = input_size
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 推理分割
    def segment(self, image, *, model_name=None, input_size=None):
        if model_name is None: model_name = self.cfg.model_name
        if input_size is None: input_size = self.cfg.input_size
        self.load_model(model_name, input_size)

        im = preprocess_image(image)
        orig = im.size
        im_r = im.resize(input_size, Image.BILINEAR)
        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        x = tfm(im_r).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            y = self.model(x)
            pred = y[0] if isinstance(y,(list,tuple)) else y
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        pred = cv2.resize(pred, orig, interpolation=cv2.INTER_LINEAR)
        return (pred*255).astype(np.uint8)

    # 背景替换（含半透明/去白边/透明导出）
    def apply_background_replacement(self, image, *, background_image=None,
                                     semi_transparent=False, semi_strength=0.5, semi_mode="auto",
                                     remove_white_halo=False, defringe_strength=0.7,
                                     model_name=None, input_size=None):
        img_arr = np.array(image) if isinstance(image, Image.Image) else np.asarray(image)
        raw_mask = self.segment(img_arr, model_name=model_name or self.cfg.model_name,
                                input_size=input_size or self.cfg.input_size)
        if raw_mask is None: raise RuntimeError("无法生成分割mask")

        if semi_transparent:
            hard = ((raw_mask.astype(np.uint8))>127).astype(np.uint8)*255
            soft_alpha = estimate_soft_alpha_inside_mask(img_arr, hard, strength=float(semi_strength), mode=semi_mode)
            refined = refine_alpha_with_channel(img_arr, soft_alpha, mode=semi_mode, strength=float(semi_strength))
            mask_u8 = refined
        else:
            mask_u8 = to_binary_mask(raw_mask, use_otsu=True)

        bg_img = bg.create_background('image' if background_image is not None else 'transparent',
                                      background_image, (img_arr.shape[1], img_arr.shape[0]))
        if bg_img is not None:
            out = bg.replace_background_with_mask(img_arr, bg_img, mask_u8,
                                                  remove_white_halo=remove_white_halo,
                                                  defringe_strength=float(defringe_strength))
        else:
            out = bg.create_transparent_result(img_arr, mask_u8,
                                               remove_white_halo=remove_white_halo,
                                               defringe_strength=float(defringe_strength))
        return out, Image.fromarray(mask_u8).convert("RGB")
