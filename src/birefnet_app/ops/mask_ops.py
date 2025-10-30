

import numpy as np, cv2
from PIL import Image
from .bg_ops import _estimate_background_inpaint  # from bg_ops


def to_binary_mask(mask: np.ndarray, *, use_otsu: bool = True) -> np.ndarray:
    """
    将 0~255 的软 mask 变成真正的二值 0/255，并做一次轻量形态学清理，避免小孔/毛刺。
    """
    m = mask
    if m.dtype != np.uint8:
        m = (np.clip(m, 0, 1) * 255).astype(np.uint8)

    # 阈值：默认 Otsu，自适应不同图像；如需固定阈值可把 use_otsu=False 改成固定 128
    if use_otsu:
        _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, m = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)

    # 轻量清理：开运算去毛刺 + 闭运算补小孔
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=1)

    return m.astype(np.uint8)

def estimate_soft_alpha_inside_mask(
    image_or_array,
    base_mask: np.ndarray,
    *,
    strength: float = 0.5,        # 0~1：影响 inpaint 半径、融合比、平滑半径
    mode: str = "auto"            # "auto" / "暗部优先" / "透色优先"
) -> np.ndarray:
    """
    在基础二值掩码 ROI 内进行非二值 α 估计（不判类别，一视同仁）。
    关键：用 inpaint 背景估计 B(x)，以通道配重近似 α ≈ ||I-B|| / ||F_ref-B||，再引导滤波。
    返回 8bit α（0~255），掩码外部自动置 0。
    """
    # ---- 输入规整 ----
    if isinstance(image_or_array, Image.Image):
        I = np.array(image_or_array.convert("RGB"))
    else:
        I = image_or_array
        if I.ndim == 3 and I.shape[2] == 4:
            I = I[:, :, :3]
    H, W = I.shape[:2]

    # 归一化基础掩码 & 二值
    if base_mask.dtype != np.uint8:
        m8 = (np.clip(base_mask, 0, 1) * 255).astype(np.uint8)
    else:
        m8 = base_mask
    binm = (m8 >= 128).astype(np.uint8)

    # ---- 背景估计：对掩码区域做 inpaint 得到 B(x) ----
    inpaint_r = 2 + int(8 * float(strength))  # 力度越大，借色半径越大
    B = _estimate_background_inpaint(I, binm, radius=inpaint_r)

    # ---- 前景参考 F_ref：取掩码的“实心”前景颜色中位数（通道分别统计） ----
    er_r = max(1, int(2 + 6 * float(strength)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er_r + 1, 2 * er_r + 1))
    fg_core = cv2.erode(binm, ker, iterations=1)
    if fg_core.sum() < 10:  # 过薄时回退到整个掩码
        fg_core = binm.copy()

    I_f = I.astype(np.float32) / 255.0
    B_f = B.astype(np.float32) / 255.0

    F_ref = np.zeros(3, np.float32)
    for c in range(3):
        vals = I_f[:, :, c][fg_core > 0]
        F_ref[c] = np.median(vals) if vals.size > 0 else 0.5

    # ---- α 粗估计：通道配重的比值形式（避免分母过小）----
    #   α_raw(x) = Σ|I-B| / (Σ|F_ref - B| + eps)
    eps = 1e-5
    num = np.abs(I_f - B_f).sum(axis=2)                     # [H,W]
    den = (np.abs(F_ref.reshape(1, 1, 3) - B_f)).sum(axis=2) + eps
    alpha_raw = num / den
    alpha_raw = np.clip(alpha_raw, 0.0, 1.0)

    # ---- 模式整形（曲线/Gamma）：更保守 or 更开放 ----
    if mode in ("暗部优先", "dark"):
        # 更保守：整体偏实（α 稍增）
        gamma = max(0.4, 1.0 - 0.5 * float(strength))  # <1 提升 α
        alpha_raw = np.power(alpha_raw, gamma)
    elif mode in ("透色优先", "bleed"):
        # 更开放：整体更透（α 稍降）
        gamma = 1.2 + 0.8 * float(strength)            # >1 压低 α
        alpha_raw = np.power(alpha_raw, gamma)

    # ---- 引导滤波/双边滤波：避免条带，保持边缘 ----
    # ---- 引导滤波/双边滤波（优先 fastGuidedFilter）----
    try:
        import cv2.ximgproc as xip
        gf_r = max(3, int(3 + 8 * float(strength)))
        try:
            alpha_smooth = xip.fastGuidedFilter(
                guide=(I_f*255).astype(np.uint8), src=alpha_raw.astype(np.float32),
                radius=gf_r, eps=1e-4
            )
        except Exception:
            alpha_smooth = xip.guidedFilter(
                guide=I_f, src=alpha_raw.astype(np.float32), radius=gf_r, eps=1e-4
            )
    except Exception:
        d = 5 + 2 * int(3 + 8 * float(strength))
        alpha_smooth = cv2.bilateralFilter(alpha_raw.astype(np.float32), d=d, sigmaColor=0.08, sigmaSpace=max(5, d))

    # ---- 与基础掩码融合（ROI 内），掩码外置零 ----
    base_alpha = (m8.astype(np.float32) / 255.0)
    mix = 0.45 + 0.5 * float(strength)   # 力度越大越接近 α_smooth
    alpha_final = (1.0 - mix) * base_alpha + mix * alpha_smooth
    alpha_final = alpha_final * (binm > 0).astype(np.float32)  # 掩码外强制 0

    return (np.clip(alpha_final, 0.0, 1.0) * 255).astype(np.uint8)


def refine_alpha_with_channel(
    image_or_array,
    base_mask: np.ndarray,
    mode: str = "auto",        # "auto" / "暗部优先" / "透色优先"
    strength: float = 0.5      # 0.0~1.0，建议默认 0.5
) -> np.ndarray:
    """
    基于“PS 通道抠图”思想的 α 估计：在掩码边界环带内按 I=αF+(1-α)B 估 α，
    并与基础掩码做可控融合，输出 0~255 的 8bit α 通道。
    """
    # --- 输入整理 ---
    if isinstance(image_or_array, Image.Image):
        img = np.array(image_or_array.convert("RGB"))
    else:
        img = image_or_array
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
    H, W = img.shape[:2]

    base = base_mask
    if base.dtype != np.uint8:
        base = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    base_alpha = (base.astype(np.float32)) / 255.0

    # --- 形态学区域: 二值掩码/未知环带/实心区 ---
    binary = (base >= 128).astype(np.uint8)
    radius = max(1, int(2 + strength * 10))  # 力度→环带半径
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dil = cv2.dilate(binary, ker, iterations=1)
    ero = cv2.erode(binary, ker, iterations=1)
    unknown = cv2.subtract(dil, ero)  # 边界环带

    expand = max(0, int(strength * 6))  # 进一步外扩
    if expand > 0:
        ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand + 1, 2 * expand + 1))
        unknown = cv2.dilate(unknown, ker2, iterations=1)

    solid_r = max(1, radius * 2)  # 更强腐蚀得到“实心”采样区
    ker_solid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * solid_r + 1, 2 * solid_r + 1))
    fg_solid = cv2.erode(binary, ker_solid, iterations=1)
    bg_solid = cv2.erode((1 - binary), ker_solid, iterations=1)

    I = img.astype(np.float32) / 255.0
    F_mean = np.zeros(3, np.float32)
    B_mean = np.zeros(3, np.float32)

    # --- 颜色统计（中位数更稳健） ---
    for c in range(3):
        vals_f = I[:, :, c][fg_solid > 0]
        vals_b = I[:, :, c][bg_solid > 0]
        if vals_f.size == 0:
            vals_f = I[:, :, c][binary > 0]
        if vals_b.size == 0:
            vals_b = I[:, :, c][binary == 0]
        F_mean[c] = np.median(vals_f) if vals_f.size > 0 else 0.8
        B_mean[c] = np.median(vals_b) if vals_b.size > 0 else 0.2

    den = F_mean - B_mean
    weights = np.abs(den)
    sw = float(weights.sum()) + 1e-6
    if sw < 1e-6:
        return base  # 分离度太低，直接返回原掩码
    weights /= sw

    # --- 未知环带 α 估计 ---
    alpha_unknown = np.zeros((H, W), np.float32)
    eps = 1e-4
    for c in range(3):
        if weights[c] <= 1e-6:
            continue
        ac = (I[:, :, c] - B_mean[c]) / (den[c] + eps)
        alpha_unknown += ac * weights[c]
    alpha_unknown = np.clip(alpha_unknown, 0.0, 1.0)

    # --- 模式 → γ 形状控制 ---
    # 暗部优先：更保守（加深），透色优先：更开放（抬高）
    if mode in ("暗部优先", "dark"):
        gamma = 1.2 + 0.8 * (1 - strength)
        alpha_unknown = np.power(alpha_unknown, gamma)
    elif mode in ("透色优先", "bleed"):
        gamma = max(0.5, 1.0 - 0.5 * strength)
        alpha_unknown = np.power(alpha_unknown, gamma)
    # 其它/auto 不做额外曲线

    # --- 和基础掩码融合，仅在未知环带影响 ---
    mixing = 0.35 + 0.55 * strength  # 力度越大越依赖α估计
    mask_unknown = (unknown > 0).astype(np.float32)
    final = base_alpha * (1.0 - mask_unknown) + \
            ((1 - mixing) * base_alpha + mixing * alpha_unknown) * mask_unknown

    # --- 边缘保持平滑（优先 guided filter，退化为双边滤波） ---
    try:
        import cv2.ximgproc as xip
        final = xip.guidedFilter(
            guide=I, src=final.astype(np.float32), radius=radius * 2 + 1, eps=1e-4
        )
    except Exception:
        d = 5 + 2 * radius
        final = cv2.bilateralFilter(final.astype(np.float32), d=d, sigmaColor=0.1, sigmaSpace=radius * 2 + 1)

    final = np.clip(final, 0.0, 1.0)
    return (final * 255).astype(np.uint8)
