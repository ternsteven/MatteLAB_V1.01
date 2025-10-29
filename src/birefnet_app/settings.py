import os

# Output directory; can be overridden via env var
PRED_OUTPUT_DIR = os.environ.get("BIRE_OUTPUT_DIR") or os.path.join(os.getcwd(), "preds-BiRefNet")

def ensure_dirs():
    os.makedirs(PRED_OUTPUT_DIR, exist_ok=True)
