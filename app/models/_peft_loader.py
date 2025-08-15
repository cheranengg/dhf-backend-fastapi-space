# app/models/_peft_loader.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any

HF_TOKEN = (
    os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN") or None
)
CACHE_DIR = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf"

def _tk_kwargs() -> Dict[str, Any]:
    kw = {"cache_dir": CACHE_DIR}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return kw

def load_base_plus_adapter(
    base_repo: str,
    adapter_repo: str,
    load_4bit: bool = (os.getenv("LOAD_4BIT", "1") == "1"),
    force_cpu: bool = (os.getenv("FORCE_CPU", "0") == "1"),
):
    """
    Returns (tokenizer, peft_model, device_str)
    Loads base Mistral + applies PEFT adapter from HF repo.
    Uses slow tokenizer (SentencePiece) to avoid protobuf.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    from peft import PeftModel

    # tokenizer: slow SP to avoid protobuf requirement
    tok = AutoTokenizer.from_pretrained(base_repo, use_fast=False, **_tk_kwargs())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    want_cuda = torch.cuda.is_available() and not force_cpu
    device = "cuda" if want_cuda else "cpu"

    bnb_cfg = None
    dtype = torch.float16 if want_cuda and not load_4bit else None

    if load_4bit and want_cuda:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    base = AutoModelForCausalLM.from_pretrained(
        base_repo,
        device_map="auto" if want_cuda else None,
        torch_dtype=dtype,
        quantization_config=bnb_cfg,
        low_cpu_mem_usage=True,
        **_tk_kwargs()
    )

    model = PeftModel.from_pretrained(base, adapter_repo, **_tk_kwargs())

    if not want_cuda:
        model.to("cpu")

    # one more guard
    try:
        model.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass

    return tok, model, device
