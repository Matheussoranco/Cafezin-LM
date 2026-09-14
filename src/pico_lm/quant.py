"""Post-training weight-only quantization (int8 / int4) for Pico-LM checkpoints.

Two independent paths:

* ``weight-only`` (this module's default) — symmetric absmax quantization of the
  weight matrices, dequantized back to the compute dtype inside ``forward``.
  Shrinks the checkpoint on disk and runs on **any** device, at the cost of a
  dequant per matmul. It does not make inference faster; it makes it smaller.
* ``dynamic`` — :func:`torch.ao.quantization.quantize_dynamic`, which really does
  int8 matmuls but only on CPU (see :mod:`pico_lm.export`).

Granularity: int8 is per-output-channel, int4 is grouped along the input
dimension (``group_size`` weights share a scale). Grouping matters a lot at 4
bits; see ``docs/posttraining.md`` for the measured numbers.

The embedding is left in fp by default. In these presets it is *most* of the
model (nano: 2.1M of 2.9M parameters), so `--quantize-embedding` is where the
size win actually comes from — and also where the quality loss comes from.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import Transformer

SUPPORTED_BITS = (4, 8)


# --------------------------------------------------------------------------- #
# tensor-level quantization
# --------------------------------------------------------------------------- #
def _effective_group(in_features: int, group_size: int) -> int:
    """Largest usable group <= ``group_size`` that divides ``in_features``.

    ``0`` (or anything >= in_features) means one group per output channel.
    """
    if group_size is None or group_size <= 0 or group_size >= in_features:
        return in_features
    if in_features % group_size == 0:
        return group_size
    return in_features


def quantize_weight(
    w: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Symmetric absmax quantize ``(out, in)`` -> (int8 codes, fp16 scales, group).

    Codes are stored one-per-int8 even at 4 bits; :func:`pack_int4` squeezes
    them to nibbles for storage.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    out_f, in_f = w.shape
    g = _effective_group(in_f, group_size)
    wg = w.detach().reshape(out_f, in_f // g, g).float()
    qmax = 2 ** (bits - 1) - 1  # 127 / 7
    scale = (wg.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-8)
    q = torch.round(wg / scale).clamp(-qmax - 1, qmax).to(torch.int8)
    return q.reshape(out_f, in_f), scale.reshape(out_f, in_f // g).to(torch.float16), g


def dequantize_weight(q: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    out_f, in_f = q.shape
    wg = q.reshape(out_f, in_f // group, group).float()
    return (wg * scale.float().unsqueeze(-1)).reshape(out_f, in_f)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack int8 codes in [-8, 7] into two nibbles per byte along the last dim."""
    if q.size(-1) % 2 != 0:
        q = F.pad(q, (0, 1))
    u = (q.to(torch.int16) + 8).to(torch.uint8)
    return (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()


def unpack_int4(p: torch.Tensor, last_dim: int) -> torch.Tensor:
    lo = (p & 0x0F).to(torch.int16) - 8
    hi = ((p >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.stack((lo, hi), dim=-1).flatten(-2).to(torch.int8)
    return out[..., :last_dim]


# --------------------------------------------------------------------------- #
# quantized modules
# --------------------------------------------------------------------------- #
class QuantLinear(nn.Module):
    """Weight-only quantized ``nn.Linear`` (bias-free, as every Linear here is).

    Weights live as int8 codes (or packed nibbles at 4 bits) plus per-group fp16
    scales, and are dequantized to the activation dtype on every forward.
    """

    def __init__(self, in_features: int, out_features: int, bits: int, group_size: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group = _effective_group(in_features, group_size)
        n_groups = in_features // self.group
        if bits == 8:
            codes = torch.zeros(out_features, in_features, dtype=torch.int8)
        else:
            codes = torch.zeros(out_features, (in_features + 1) // 2, dtype=torch.uint8)
        self.register_buffer("qweight", codes)
        self.register_buffer("scales", torch.zeros(out_features, n_groups, dtype=torch.float16))

    @classmethod
    def from_linear(cls, lin: nn.Linear, bits: int, group_size: int) -> QuantLinear:
        mod = cls(lin.in_features, lin.out_features, bits, group_size)
        mod.load_weight(lin.weight.data)
        return mod

    def load_weight(self, w: torch.Tensor) -> None:
        q, scale, group = quantize_weight(w, self.bits, self.group)
        self.group = group
        self.qweight.copy_(pack_int4(q) if self.bits == 4 else q)
        self.scales.copy_(scale)

    def dequantized(self) -> torch.Tensor:
        q = unpack_int4(self.qweight, self.in_features) if self.bits == 4 else self.qweight
        return dequantize_weight(q, self.scales, self.group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantized().to(x.dtype))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, bits={self.bits}, group={self.group}"
        )


class QuantEmbedding(nn.Module):
    """Weight-only quantized embedding table.

    Only the looked-up rows are dequantized, so this is cheap on the input side.
    When weights are tied, the *same* buffers back the lm_head projection.
    """

    def __init__(self, num_embeddings: int, dim: int, bits: int, group_size: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.bits = bits
        self.group = _effective_group(dim, group_size)
        n_groups = dim // self.group
        if bits == 8:
            self.register_buffer("qweight", torch.zeros(num_embeddings, dim, dtype=torch.int8))
        else:
            self.register_buffer(
                "qweight", torch.zeros(num_embeddings, (dim + 1) // 2, dtype=torch.uint8)
            )
        self.register_buffer("scales", torch.zeros(num_embeddings, n_groups, dtype=torch.float16))

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, bits: int, group_size: int) -> QuantEmbedding:
        mod = cls(emb.num_embeddings, emb.embedding_dim, bits, group_size)
        q, scale, group = quantize_weight(emb.weight.data, bits, mod.group)
        mod.group = group
        mod.qweight.copy_(pack_int4(q) if bits == 4 else q)
        mod.scales.copy_(scale)
        return mod

    def dequantized(self) -> torch.Tensor:
        q = unpack_int4(self.qweight, self.dim) if self.bits == 4 else self.qweight
        return dequantize_weight(q, self.scales, self.group)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        rows = self.qweight[idx]  # (..., in) int8 / (..., in/2) uint8
        scales = self.scales[idx]
        flat_q = rows.reshape(-1, rows.size(-1))
        flat_s = scales.reshape(-1, scales.size(-1))
        if self.bits == 4:
            flat_q = unpack_int4(flat_q, self.dim)
        out = dequantize_weight(flat_q, flat_s, self.group)
        return out.reshape(*idx.shape, self.dim)


class TiedQuantHead(nn.Module):
    """lm_head that reuses a :class:`QuantEmbedding`'s codes (weight tying)."""

    def __init__(self, embedding: QuantEmbedding) -> None:
        super().__init__()
        self.embedding = [embedding]  # hidden from state_dict: the buffers live on the embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding[0].dequantized().to(x.dtype))


# --------------------------------------------------------------------------- #
# model-level quantization
# --------------------------------------------------------------------------- #
ATTN_PROJECTIONS = ("wq", "wk", "wv", "wo")
MLP_PROJECTIONS = ("w1", "w2", "w3")


def _linear_targets(model: Transformer) -> list[tuple[nn.Module, str, nn.Linear]]:
    targets = []
    for block in model.blocks:
        for parent, names in (
            (block.attn, ATTN_PROJECTIONS),
            (block.mlp, MLP_PROJECTIONS),
        ):
            targets.extend((parent, name, getattr(parent, name)) for name in names)
    return targets


def quantize_model(
    model: Transformer,
    bits: int = 8,
    group_size: int = 64,
    quantize_embedding: bool = False,
) -> Transformer:
    """Swap the transformer's Linear (and optionally embedding) layers in place."""
    model = model.to("cpu").eval()
    for parent, name, lin in _linear_targets(model):
        setattr(parent, name, QuantLinear.from_linear(lin, bits, group_size))

    if quantize_embedding:
        qemb = QuantEmbedding.from_embedding(model.tok_emb, bits, group_size)
        tied = model.cfg.tie_weights
        model.tok_emb = qemb
        model.lm_head = (
            TiedQuantHead(qemb)
            if tied
            else QuantLinear.from_linear(model.lm_head, bits, group_size)
        )
    return model


def build_quantized_shell(cfg: ModelConfig, meta: dict) -> Transformer:
    """An un-populated quantized model with the right module structure to load into."""
    model = Transformer(cfg)
    bits, group = meta["bits"], meta["group_size"]
    for parent, name, lin in _linear_targets(model):
        setattr(parent, name, QuantLinear(lin.in_features, lin.out_features, bits, group))
    if meta.get("quantize_embedding"):
        qemb = QuantEmbedding(cfg.vocab_size, cfg.dim, bits, group)
        model.tok_emb = qemb
        model.lm_head = (
            TiedQuantHead(qemb)
            if cfg.tie_weights
            else QuantLinear(cfg.dim, cfg.vocab_size, bits, group)
        )
    return model


def save_quantized(model: Transformer, meta: dict, ckpt: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": ckpt["model_cfg"],
            "quant": meta,
            "step": ckpt.get("step"),
            "best_val": ckpt.get("best_val"),
            "tokenizer_path": ckpt.get("tokenizer_path"),
            "source_provenance": ckpt.get("provenance"),
        },
        out_path,
    )
    return out_path


def is_quantized_ckpt(ckpt: dict) -> bool:
    return isinstance(ckpt.get("quant"), dict)


def load_quantized(ckpt_path: str | Path, device: str = "cpu") -> tuple[Transformer, dict]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    model = build_quantized_shell(cfg, ckpt["quant"])
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt


def quantize_checkpoint(
    ckpt_path: str | Path,
    out_path: str | Path,
    bits: int = 8,
    group_size: int = 64,
    quantize_embedding: bool = False,
) -> tuple[Path, dict]:
    """Quantize a trained checkpoint and write a self-contained quantized one."""
    from .train import load_model

    model, ckpt = load_model(ckpt_path, device="cpu")
    meta = {
        "bits": bits,
        "group_size": group_size,
        "quantize_embedding": quantize_embedding,
        "method": "weight-only-symmetric",
    }
    from .provenance import manifest

    meta["provenance"] = manifest(
        [ckpt_path], bits=bits, group_size=group_size, quantize_embedding=quantize_embedding
    )
    quantize_model(model, bits, group_size, quantize_embedding)
    out = save_quantized(model, meta, ckpt, out_path)
    return out, meta
