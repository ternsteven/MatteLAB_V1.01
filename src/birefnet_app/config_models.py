# Model aliases and descriptions for UI display.
usage_to_weights_file = {
    'General': 'BiRefNet',
    'General-Lite': 'BiRefNet_lite',
    'General-Lite-2K': 'BiRefNet_lite-2K',
    'Matting': 'BiRefNet-matting',
    'Portrait': 'BiRefNet-portrait',
    'DIS': 'BiRefNet-DIS5K',
    'HRSOD': 'BiRefNet-HRSOD',
    'COD': 'BiRefNet-COD',
    'DIS-TR_TEs': 'BiRefNet-DIS5K-TR_TEs',
    'General-legacy': 'BiRefNet-legacy'
}

model_descriptions = {
    "General": "通用版（BiRefNet） - 适合大多数自然图像",
    "General-Lite": "轻量版（BiRefNet_lite） - 推理速度快，精度略低",
    "General-Lite-2K": "高分辨率版（BiRefNet_lite-2K） - 适合2K图像",
    "Matting": "抠图版（BiRefNet-matting） - 擅长发丝、透明边缘",
    "Portrait": "人像优化版（BiRefNet-portrait） - 擅长人像抠图",
    "DIS": "细节增强版（BiRefNet-DIS5K） - 细节表现更好",
    "HRSOD": "高分辨率分割版（BiRefNet-HRSOD） - 复杂背景效果更佳",
    "COD": "伪装检测版（BiRefNet-COD） - 擅长隐藏/伪装目标",
    "DIS-TR_TEs": "DIS5K训练增强版（BiRefNet-DIS5K-TR_TEs）",
    "General-legacy": "旧版通用模型（兼容性好，权重较旧）"
}
