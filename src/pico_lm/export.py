"""Export: dynamic int8 quantization for fast CPU inference."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .provenance import manifest
from .train import load_model


def quantize_int8(ckpt_path: str | Path, out_path: str | Path) -> Path:
    """Dynamic int8 quantization of all Linear layers; roughly 4x smaller, CPU-only."""
    model, ckpt = load_model(ckpt_path, device="cpu")
    model.eval()
    qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "dynamic-int8-state-v1",
            "model": qmodel.state_dict(),
            "model_cfg": ckpt["model_cfg"],
            "tokenizer_path": ckpt.get("tokenizer_path"),
            "provenance": manifest([ckpt_path], method="dynamic-int8-state-v1"),
            "source_provenance": ckpt.get("provenance"),
        },
        out_path,
    )
    return out_path


def load_quantized(path: str | Path) -> tuple[torch.nn.Module, str | None]:
    model, ckpt = load_model(path, device="cpu")
    return model, ckpt.get("tokenizer_path")
