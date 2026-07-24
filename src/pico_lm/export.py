"""Export: dynamic int8 quantization for fast CPU inference."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

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
            "quantized_model": qmodel,  # pickled module: load with weights_only=False
            "model_cfg": ckpt["model_cfg"],
            "tokenizer_path": ckpt.get("tokenizer_path"),
        },
        out_path,
    )
    return out_path


def load_quantized(path: str | Path) -> tuple[torch.nn.Module, str | None]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    return ckpt["quantized_model"], ckpt.get("tokenizer_path")
