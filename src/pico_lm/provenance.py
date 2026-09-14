"""Content fingerprints and safe, portable experiment manifests."""

import hashlib
import importlib.metadata
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def tokenizer_hash(tok):
    return hashlib.sha256(tok.tk.to_str().encode("utf-8")).hexdigest()


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def manifest(paths=(), **settings):
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root))
    except (OSError, subprocess.CalledProcessError):
        commit, diff, dirty = None, b"", None
    return {
        "schema": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "dirty": dirty,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_files": [fingerprint(p) for p in sorted(Path(__file__).parent.glob("*.py"))],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "tokenizers", "pico-lm")
        },
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "files": [fingerprint(p) for p in paths if p and Path(p).is_file()],
        "settings": settings,
    }


def capture_rng():
    ns = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
    }


def restore_rng(state):
    if state is None:
        print("warning: legacy checkpoint has no RNG state; exact continuation unavailable")
        return
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
