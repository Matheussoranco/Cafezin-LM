"""Decoder-only transformer: RMSNorm, RoPE, grouped-query attention, SwiGLU.

The same architecture family as Llama/Mistral, scaled down. Supports an
incremental KV cache for O(1)-per-token generation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(
    head_dim: int, max_seq_len: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D/2)
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2).type_as(x)


class KVCache:
    """Preallocated per-layer key/value cache for incremental decoding."""

    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        shape = (cfg.n_layers, batch_size, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.size(2)
        self.k[layer, :, :, self.pos : self.pos + t] = k
        self.v[layer, :, :, self.pos : self.pos + t] = v
        end = self.pos + t
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        layer: int = 0,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        past = cache.pos if cache is not None else 0
        if cache is not None:
            k, v = cache.update(layer, k, v)

        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        drop = self.dropout if self.training else 0.0
        if past == 0:
            # full-sequence forward (training or prefill): standard causal mask
            y = F.scaled_dot_product_attention(q, k, v, is_causal=t > 1, dropout_p=drop)
        elif t == 1:
            # single-token decode: attend to everything cached
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # multi-token continuation over an existing cache: offset causal mask
            kv_len = past + t
            mask = torch.ones(t, kv_len, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.dim * 8 / 3)
        hidden = cfg.mlp_multiple_of * ((hidden + cfg.mlp_multiple_of - 1) // cfg.mlp_multiple_of)
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None, layer=0):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, cache, layer))
        x = x + self.drop(self.mlp(self.mlp_norm(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scale residual-stream output projections (GPT-2 style)
        scale = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=scale)
            nn.init.normal_(block.mlp.w2.weight, mean=0.0, std=scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_weights:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        t = idx.size(1)
        pos = cache.pos if cache is not None else 0
        if pos + t > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {pos + t} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        cos = self.rope_cos[pos : pos + t]
        sin = self.rope_sin[pos : pos + t]

        x = self.drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache, i)
        x = self.norm(x)
        if cache is not None:
            cache.pos += t

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            return logits, loss
        return self.lm_head(x), None

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = None,
        eos_id: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        device = idx.device
        # leave room for generation inside the positional window
        keep = max(1, self.cfg.max_seq_len - max_new_tokens)
        if idx.size(1) > keep:
            idx = idx[:, -keep:]
        max_new_tokens = min(max_new_tokens, self.cfg.max_seq_len - idx.size(1))

        cache = KVCache(self.cfg, idx.size(0), device, self.tok_emb.weight.dtype)
        logits, _ = self(idx, cache=cache)
        out = idx
        next_input = None
        for _ in range(max_new_tokens):
            if next_input is not None:
                logits, _ = self(next_input, cache=cache)
            step_logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                for b in range(out.size(0)):
                    seen = out[b].unique()
                    penalized = step_logits[b, seen]
                    step_logits[b, seen] = torch.where(
                        penalized > 0,
                        penalized / repetition_penalty,
                        penalized * repetition_penalty,
                    )

            if temperature <= 0:
                next_tok = step_logits.argmax(dim=-1, keepdim=True)
            else:
                step_logits = step_logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, step_logits.size(-1))
                    kth = torch.topk(step_logits, k, dim=-1).values[:, -1:]
                    step_logits = step_logits.masked_fill(step_logits < kth, float("-inf"))
                if top_p is not None and 0 < top_p < 1:
                    sorted_logits, sorted_idx = torch.sort(step_logits, descending=True, dim=-1)
                    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    # keep tokens until cumulative prob exceeds top_p (always keep the first)
                    remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    step_logits = torch.full_like(step_logits, float("-inf")).scatter(
                        -1, sorted_idx, sorted_logits
                    )
                probs = torch.softmax(step_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, next_tok], dim=1)
            next_input = next_tok
            if eos_id is not None and (next_tok == eos_id).all():
                break
        return out
