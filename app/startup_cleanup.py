# app/startup_cleanup.py
# Purpose:
# - Ensure persistent cache/offload dirs exist under /data
# - Optionally clean caches via env flags (disabled by default)
# - Never auto-wipe /data caches on every restart

from __future__ import annotations

import os
import shutil
from typing import Iterable

# ----------------------------
# Env flags (safe defaults)
# ----------------------------
# If 1, allow cleaning HF_HOME at startup (ONLY if not under /data)
CLEAN_HF_CACHE = os.getenv("CLEAN_HF_CACHE", "0") == "1"

# If 1, clean any transient /tmp caches (safe)
CLEAN_TMP_CACHE = os.getenv("CLEAN_TMP_CACHE", "1") == "1"

# Paths (honor your Space variables)
HF_HOME = os.getenv("HF_HOME", "/tmp/hf")
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "transformers"))
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/tmp/offload")

# Persistent base (Hugging Face Spaces persistent volume)
PERSIST_BASE = "/data"  # existence is managed by Spaces when persistent storage is enabled


# ----------------------------
# Helpers
# ----------------------------
def _safe_rm(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)
    except Exception:
        # best-effort remove; keep startup resilient
        pass


def _ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        try:
            os.makedirs(p, exist_ok=True)
        except Exception:
            # If /data isn't mounted (no persistent storage), this will fail silently
            pass


def _is_under(path: str, parent: str) -> bool:
    try:
        return os.path.realpath(path).startswith(os.path.realpath(parent) + os.sep)
    except Exception:
        return False


# ----------------------------
# One-time startup actions
# ----------------------------
# 1) Ensure persistent dirs exist (ok if they already do)
_ensure_dirs([
    os.path.join(PERSIST_BASE, "hf", "hub"),
    os.path.join(PERSIST_BASE, "hf", "transformers"),
    os.path.join(PERSIST_BASE, "offload"),
])

# 2) Optionally clean transient caches
if CLEAN_TMP_CACHE:
    # Safe to clear /tmp caches on every boot
    for tmp_path in ["/tmp/hf", "/tmp/offload"]:
        _safe_rm(tmp_path)
        _ensure_dirs([tmp_path])
    print(f"[startup] Cleaned transient caches under /tmp")

# 3) Optionally clean HF_HOME cache (ONLY if not in /data)
if CLEAN_HF_CACHE:
    if not _is_under(HF_HOME, PERSIST_BASE):
        _safe_rm(HF_HOME)
        _ensure_dirs([HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE])
        print(f"[startup] CLEAN_HF_CACHE=1 → cleaned HF_HOME at: {HF_HOME}")
    else:
        print(f"[startup] CLEAN_HF_CACHE=1 ignored: HF_HOME is under {PERSIST_BASE} (persistent).")

# 4) Final ensure (honor env-configured paths)
_ensure_dirs([HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE, OFFLOAD_DIR])

print(
    "[startup] Ready. "
    f"HF_HOME={HF_HOME} | HF_HUB_CACHE={HF_HUB_CACHE} | "
    f"TRANSFORMERS_CACHE={TRANSFORMERS_CACHE} | OFFLOAD_DIR={OFFLOAD_DIR}"
)
