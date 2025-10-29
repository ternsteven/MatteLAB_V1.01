import os, cv2, numpy as np, torch, requests
from PIL import Image
import torchvision.transforms as T
from transformers import AutoModelForImageSegmentation

from .config_models import usage_to_weights_file
from .ops.image_io import preprocess_image
from .ops.mask_ops import to_binary_mask, estimate_soft_alpha_inside_mask, refine_alpha_with_channel
from .ops import bg_ops as bg
from .compose import apply_background_replacement as _compose_apply

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
    def apply_background_replacement(self, *args, **kwargs):
        return _compose_apply(self, *args, **kwargs)