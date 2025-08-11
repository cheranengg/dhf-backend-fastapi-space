# app/utils/warmup.py
import os

def _try_load_hf(env_key: str):
    path = os.getenv(env_key, "")
    if not path or not os.path.exists(path):
        return
    print(f"[warmup] Checking {env_key} at {path}")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _ = AutoModelForCausalLM.from_pretrained(path)
        print(f"[warmup] Loaded OK: {path}")
    except Exception as e:
        print(f"[warmup] Skipped {path}: {e}")

def run():
    # Touch all three if present (no-op if dirs missing)
    _try_load_hf("HA_MODEL_MERGED_DIR")
    _try_load_hf("DVP_MODEL_DIR")
    _try_load_hf("TM_MODEL_DIR")

if __name__ == "__main__":
    run()
