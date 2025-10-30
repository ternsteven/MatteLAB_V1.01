
# -*- coding: utf-8 -*-
import logging, os
from .settings import LOG_DIR, ensure_dirs
def get_logger(name: str = "MatteLAB.UI") -> logging.Logger:
    ensure_dirs()
    lg = logging.getLogger(name)
    if lg.handlers: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt); lg.addHandler(ch)
    os.makedirs(LOG_DIR, exist_ok=True)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "mattelab.log"), encoding="utf-8")
    fh.setLevel(logging.INFO); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg

