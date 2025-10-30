

# -*- coding: utf-8 -*-
import os
OUT_DIR = os.getenv("BIRE_OUT_DIR", "preds-BiRefNet")
MASK_SUBDIR = os.getenv("BIRE_MASK_SUBDIR", "masks")
LOG_DIR = os.getenv("BIRE_LOG_DIR", "logs")
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, MASK_SUBDIR), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

