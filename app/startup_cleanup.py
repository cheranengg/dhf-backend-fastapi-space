# app/startup_cleanup.py
import os, shutil

def _safe_rm(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass

def clean_hf_cache():
    base = os.getenv("HF_HOME", "/tmp/hf")
    # clear the whole cache dir (safe for Spaces)
    _safe_rm(base)
    os.makedirs(base, exist_ok=True)

# run on import
try:
    clean_hf_cache()
    print(f"[startup] HF cache cleaned under: {os.getenv('HF_HOME','/tmp/hf')}")
except Exception as e:
    print(f"[startup] HF cache cleanup skipped: {e}")
