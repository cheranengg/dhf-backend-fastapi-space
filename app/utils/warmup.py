import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def _prefetch(repo):
    if not repo:
        return
    kw = {}
    tok = os.getenv("HF_TOKEN")
    if tok:
        kw["token"] = tok
    cache_dir = os.getenv("HF_HOME")
    if cache_dir:
        kw["cache_dir"] = cache_dir
    print(f"[warmup] prefetch {repo}")
    t = AutoTokenizer.from_pretrained(repo, **kw)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    _ = AutoModelForCausalLM.from_pretrained(repo, **kw)

def run():
    _prefetch(os.getenv("HA_MODEL_MERGED_DIR"))
    _prefetch(os.getenv("DVP_MODEL_DIR"))
    _prefetch(os.getenv("TM_MODEL_DIR"))

if __name__ == "__main__":
    run()
