# -*- coding: utf-8 -*-
import numpy as np, cv2

def to_single_channel_uint8(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 2:
        out = a
    elif a.ndim == 3:
        out = a[..., 0]
    else:
        raise ValueError("mask 维度不支持")
    if out.dtype != np.uint8:
        out = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
    return out

def bbox_from_mask(mask, expand_ratio: float = 0.06, min_size: int = 12):
    m = to_single_channel_uint8(mask)
    ys, xs = np.where(m >= 128)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)  # 空框

    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x2 - x1 + 1, y2 - y1 + 1
    cx, cy = x1 + w / 2, y1 + h / 2

    # 四周按比例扩展
    rx, ry = int(w * expand_ratio), int(h * expand_ratio)
    x1, y1 = max(0, x1 - rx), max(0, y1 - ry)
    x2, y2 = x2 + rx, y2 + ry

    # 保证最小尺寸
    if (x2 - x1 + 1) < min_size:
        pad = (min_size - (x2 - x1 + 1)) // 2 + 1
        x1, x2 = x1 - pad, x2 + pad
    if (y2 - y1 + 1) < min_size:
        pad = (min_size - (y2 - y1 + 1)) // 2 + 1
        y1, y2 = y1 - pad, y2 + pad

    return (int(x1), int(y1), int(x2), int(y2))
