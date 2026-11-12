"""Cafezin LM: a hybrid linear/full-attention multimodal decoder, in pure PyTorch.

A faithful port of the Qwen3.5-VL architecture family. Every module attribute
name is chosen so that ``state_dict()`` keys match the reference checkpoint
*exactly* — ``model.language_model.layers.3.self_attn.q_proj.weight`` and
friends — which is what makes the weights interchangeable rather than merely
similar. `cafezin_lm.checkpoint` proves that claim against the real files.

The one component with no public reference is the MTP head: the reference
loader ignores ``^mtp.*``, so its wiring is inferred from the tensor shapes
(``fc`` mapping 2 x hidden -> hidden, one full-attention decoder layer, shared
embeddings) and the standard multi-token-prediction formulation. It is
documented as inferred, not copied.

Precision note: the chunked gated-delta-rule path computes in float32
internally, exactly like the reference. The linear-attention layers are the
numerically delicate part of this architecture; do not "optimise" that away.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig, TextConfig, VisionConfig

CHUNK_SIZE = 64


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


class RMSNormZeroCentered(nn.Module):
    """RMSNorm whose weight is a *delta* around 1: `out = norm(x) * (1 + w)`.

    Initialised to zeros so a fresh model behaves like an unweighted RMSNorm.
    This is not a stylistic choice — the reference checkpoint is stored this way
    (see docs/architecture.md); loading the weights with a plain `w` scale would
    be wrong by a per-channel offset.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * (1.0 + self.weight.float())).type_as(x)


class RMSNorm(nn.Module):
    """Plain RMSNorm with a direct per-channel scale (weight initialised to 1)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * self.weight.float()).type_as(x)


class RMSNormGated(nn.Module):
    """RMSNorm then a SiLU gate — the head-wise norm closing a linear-attention block.

    Note the order: normalise, *then* gate. The gate is applied in float32 and
    the result cast back, which is what the reference does.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        states = hidden_states.float()
        variance = states.pow(2).mean(-1, keepdim=True)
        states = states * torch.rsqrt(variance + self.variance_epsilon)
        states = self.weight * states.to(input_dtype)
        states = states * F.silu(gate.float())
        return states.to(input_dtype)


class LayerNorm(nn.Module):
    """LayerNorm for the vision tower, with an optional zero-centred scale.

    ``zero_centered=True`` computes ``(1 + weight)`` like the rest of this model;
    the default follows the reference port, which uses the stored weight as-is.
    See ``VisionConfig.zero_centered_norm``.
    """

    def __init__(self, dim: int, eps: float = 1e-6, zero_centered: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (1.0 + self.weight) if self.zero_centered else self.weight
        return F.layer_norm(x, (x.shape[-1],), weight, self.bias, self.eps)


# ---------------------------------------------------------------------------
# rotary embeddings
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the leading `cos.shape[-1]` channels; pass the rest through.

    Partial rotary: only 64 of each 256-wide head is rotated, so the tail is
    concatenated back untouched.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


class TextRotaryEmbedding(nn.Module):
    """Partial-rotary mRoPE for the text decoder.

    Three position grids (temporal, height, width) are packed into one rotary
    table: grid 0 fills the whole table, then grids H and W overwrite interleaved
    slices. For text-only input the three grids are identical and this degenerates
    to ordinary RoPE — which is why a text-only model still needs the 3-row form.
    """

    def __init__(self, cfg: TextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_seq_len_cached = cfg.max_position_embeddings
        rope_dim = cfg.rotary_dim
        inv_freq = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim)
        )
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(cfg.mrope_section)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (3, B, T)
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.shape[0] == 1:
            position_ids = position_ids.expand(3, -1, -1)
        if position_ids.shape[0] != 3:
            raise ValueError(
                f"text rotary expects 3 position grids (t, h, w); got {position_ids.shape[0]}"
            )
        freqs = position_ids[..., None].float() * self.inv_freq.to(x.device)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        sin = self.recomposition_frequencies(sin)
        cos = self.recomposition_frequencies(cos)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def recomposition_frequencies(self, freq: torch.Tensor) -> torch.Tensor:
        freqs_thw = freq[0].clone()  # start from the temporal grid
        for grid, offset in enumerate((1, 2), start=1):  # overwrite H, then W slots
            length = self.mrope_section[grid] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[grid, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)


class VisionRotaryEmbedding(nn.Module):
    """Axial 2-D RoPE: half the head for the height axis, half for the width."""

    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        spatial_dim = cfg.head_dim // 2
        inv_freq = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
        )
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (2, N) — row 0 = h, row 1 = w
        freqs = position_ids[..., None].float() * self.inv_freq.to(x.device)
        cos = self.recomposition_frequencies(freqs.cos() * self.attention_scaling)
        sin = self.recomposition_frequencies(freqs.sin() * self.attention_scaling)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def recomposition_frequencies(self, freq: torch.Tensor) -> torch.Tensor:
        freq_hw = torch.cat([freq[0], freq[1]], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)


# ---------------------------------------------------------------------------
# linear attention: the gated delta rule
# ---------------------------------------------------------------------------


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked gated delta rule (the training-time linear-attention kernel).

    The recurrence this evaluates, per token t and head h, is

        S_t = S_{t-1} * exp(g_t) * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
        o_t = S_t^T q_t

    evaluated in O(T * chunk) rather than O(T^2) attention. Rather than looping
    over tokens, it splits the sequence into chunks and solves each chunk's
    "UT transform" (a unit lower-triangular system) so the intra-chunk part is a
    few matmuls, then scans the per-chunk states.

    Shapes in: q/k (B, T, Hk, Dk), v (B, T, Hv, Dv), g/beta (B, T, Hv).
    Shapes out: o (B, T, Hv, Dv) and optionally the final state (B, Hv, Dk, Dv).
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, num_k_heads, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    decay = g

    # Grouped value attention: each q/k head serves (Hv // Hk) consecutive values.
    if num_k_heads != num_v_heads:
        repeats = num_v_heads // num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query * (query.shape[-1] ** -0.5)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total = sequence_length + pad_size
    num_chunks = total // chunk_size

    # beta is the state's per-token learning rate; apply it to K and V up front.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    )
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    strictly_upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(
        1
    )

    # Decay is in log space, so a cumsum is a running product of gates.
    cum_decay = decay.cumsum(dim=3)
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper, float("-inf")).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # Solve the unit-lower-triangular UT system: this converts a sequence of
    # delta-rule updates into two triangular solves.
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(
        ut_system, decayed_k_beta, upper=False, unitriangular=True
    )

    if initial_state is None:
        last_recurrent_state = torch.zeros(
            recurrent_state_shape, dtype=new_values.dtype, device=new_values.device
        )
    else:
        last_recurrent_state = initial_state.to(new_values)

    core_attn_out = torch.zeros_like(new_values)

    # Apply the within-chunk decay once instead of per chunk.
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    for i in range(num_chunks):
        # Only the part of the chunk's values the old state cannot already
        # predict is written into the state — the delta rule's correction term.
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        last_recurrent_state = (
            last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
        )

    final_state = last_recurrent_state if output_final_state else None
    core_attn_out = core_attn_out.reshape(batch_size, num_v_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format
    )
    return core_attn_out, final_state


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Token-by-token form of the same rule — the decode path.

    Kept because it is the definition the chunked kernel must agree with, and
    `tests/test_model.py` asserts they do.
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, num_k_heads, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g

    if num_k_heads != num_v_heads:
        repeats = num_v_heads // num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query / (query.shape[-1] ** 0.5)

    if initial_state is None:
        last_recurrent_state = torch.zeros(
            (batch_size, num_v_heads, k_head_dim, v_head_dim),
            dtype=value.dtype,
            device=value.device,
        )
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        last_recurrent_state = last_recurrent_state * decay[:, :, i].exp()[..., None, None]
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    final_state = last_recurrent_state if output_final_state else None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, final_state


class GatedDeltaNet(nn.Module):
    """Linear-attention token mixer: depthwise causal conv -> gated delta rule -> gated norm."""

    def __init__(self, cfg: TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_v_heads = cfg.linear_num_value_heads
        self.num_k_heads = cfg.linear_num_key_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.layer_norm_epsilon = cfg.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Discretisation step (never negative: softplus) and per-head decay.
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: GenerationCache | None = None,
        delta_rule: Callable | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

        batch_size, seq_len, _ = hidden_states.shape
        delta_rule = delta_rule or chunk_gated_delta_rule

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        conv_state = cache.conv_state(self.layer_idx, self.conv_dim) if cache is not None else None
        if conv_state is not None:
            # Depthwise causal conv over the window carried from the previous step.
            window = torch.cat([conv_state, mixed_qkv], dim=-1).to(self.conv1d.weight.dtype)
            keep = self.conv_kernel_size - 1
            cache.set_conv_state(self.layer_idx, window[:, :, -keep:] if keep else window[:, :, :0])
            conv_out = F.conv1d(window, self.conv1d.weight, None, padding=0, groups=self.conv_dim)
            mixed_qkv = conv_out[:, :, -seq_len:]
        else:
            mixed_qkv = F.conv1d(
                mixed_qkv.to(self.conv1d.weight.dtype),
                self.conv1d.weight,
                None,
                padding=self.conv_kernel_size - 1,
                groups=self.conv_dim,
            )[:, :, :seq_len]
        mixed_qkv = F.silu(mixed_qkv).to(hidden_states.dtype)

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # g <= 0 always: exp(g) is the state's decay factor. float32 keeps
        # A_log.exp() finite when the model is loaded in half precision.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        recurrent_state = (
            cache.recurrent_state(self.layer_idx, batch_size) if cache is not None else None
        )
        core_attn_out, last_state = delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache is not None,
            use_qk_l2norm_in_kernel=True,
        )
        if cache is not None and last_state is not None:
            cache.set_recurrent_state(self.layer_idx, last_state)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# full attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Full attention with GQA and a sigmoid gate on the output (attn_output_gate)."""

    def __init__(self, cfg: TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_key_value_heads = cfg.num_key_value_heads
        self.num_key_value_groups = cfg.num_key_value_groups
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = cfg.attention_dropout
        self.attn_output_gate = cfg.attn_output_gate

        q_out = self.num_heads * self.head_dim * (2 if self.attn_output_gate else 1)
        self.q_proj = nn.Linear(cfg.hidden_size, q_out, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(
            cfg.hidden_size, self.num_key_value_heads * self.head_dim, bias=cfg.attention_bias
        )
        self.v_proj = nn.Linear(
            cfg.hidden_size, self.num_key_value_heads * self.head_dim, bias=cfg.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias
        )
        self.q_norm = RMSNormZeroCentered(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNormZeroCentered(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: GenerationCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        head_shape = (batch_size, seq_len, -1, self.head_dim)

        if self.attn_output_gate:
            # q_proj emits [query, gate] interleaved per head.
            q_proj, gate = torch.chunk(
                self.q_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim * 2),
                2,
                dim=-1,
            )
            gate = gate.reshape(batch_size, seq_len, -1)
        else:
            q_proj, gate = self.q_proj(hidden_states), None

        query = self.q_norm(q_proj.view(head_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)

        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if cache is not None:
            key, value = cache.append_kv(self.layer_idx, key, value)

        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_mask = self._build_mask(query, key, seq_len, attention_mask, cache)
        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            # a mask already encodes causality; only the mask-free path may use the flag
            is_causal=attn_mask is None and seq_len > 1,
            scale=self.scaling,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn = attn.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        if gate is not None:
            attn = attn * torch.sigmoid(gate)
        return self.o_proj(attn)

    def _build_mask(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        seq_len: int,
        attention_mask: torch.Tensor | None,
        cache: GenerationCache | None,
    ) -> torch.Tensor | None:
        """Boolean keep-mask over (query, key) for both prefill and decode.

        Returns None when a plain causal flag is enough (the common training
        path). Query position ``p`` may read key ``k`` iff ``k <= p``, which is
        what makes a cached decode step attend over everything already written
        while a multi-token continuation stays causal. Padding keys are removed
        as well; a row left with nothing to attend to is opened up instead of
        left all-masked, because SDPA turns an all-masked row into NaN and those
        rows are padding anyway (excluded from the loss).
        """
        past = cache.prefill_length if cache is not None else 0
        if past == 0 and attention_mask is None:
            return None

        batch = query.size(0)
        total = key.size(2)
        q_pos = torch.arange(past, past + seq_len, device=query.device)
        k_pos = torch.arange(total, device=query.device)
        mask = (k_pos[None, :] <= q_pos[:, None]).expand(batch, seq_len, total).clone()
        if attention_mask is not None and attention_mask.shape[-1] == total:
            mask &= attention_mask.bool().reshape(batch, 1, total)
        mask |= ~mask.any(dim=-1, keepdim=True)
        return mask.unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, cfg: TextConfig, intermediate_size: int | None = None) -> None:
        super().__init__()
        intermediate = intermediate_size or cfg.intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """One text block: pre-norm token mixer, then pre-norm SwiGLU MLP."""

    def __init__(self, cfg: TextConfig, layer_idx: int, block_type: str | None = None) -> None:
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.block_type = block_type or cfg.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(cfg, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Attention(cfg, layer_idx)
        else:
            raise ValueError(f"unknown block type {self.block_type!r}")
        self.mlp = MLP(cfg, cfg.intermediate_size)
        self.input_layernorm = RMSNormZeroCentered(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNormZeroCentered(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: GenerationCache | None = None,
        delta_rule: Callable | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden = self.linear_attn(
                hidden, attention_mask=attention_mask, cache=cache, delta_rule=delta_rule
            )
        else:
            hidden = self.self_attn(
                hidden, cos=cos, sin=sin, attention_mask=attention_mask, cache=cache
            )
        hidden_states = residual + hidden

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# multi-token prediction
# ---------------------------------------------------------------------------


class MTP(nn.Module):
    """One-layer multi-token-prediction head (inferred wiring — see module docstring).

    Given the trunk's hidden state at position i and the embedding of token
    i+1, it fuses the two and runs a single full-attention block to predict
    token i+2. Weight names follow the checkpoint (``mtp.fc``,
    ``mtp.pre_fc_norm_*``, ``mtp.layers.*``, ``mtp.norm``); embeddings and the
    output projection are shared with the trunk, matching
    ``mtp_use_dedicated_embeddings: false`` and the tied ``lm_head``.
    """

    def __init__(self, cfg: TextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_size
        self.pre_fc_norm_hidden = RMSNormZeroCentered(hidden, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_embedding = RMSNormZeroCentered(hidden, eps=cfg.rms_norm_eps)
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.layers = nn.ModuleList(
            DecoderLayer(cfg, 0, block_type="full_attention")
            for _ in range(cfg.mtp_num_hidden_layers)
        )
        self.norm = RMSNormZeroCentered(hidden, eps=cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        next_embeds: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.fc(
            torch.cat(
                [self.pre_fc_norm_embedding(next_embeds), self.pre_fc_norm_hidden(hidden_states)],
                dim=-1,
            )
        )
        for layer in self.layers:
            fused = layer(fused, cos, sin)
        return self.norm(fused)


# ---------------------------------------------------------------------------
# vision tower
# ---------------------------------------------------------------------------


class VisionPatchEmbed(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.patch_size = cfg.patch_size
        self.temporal_patch_size = cfg.temporal_patch_size
        self.in_channels = cfg.in_channels
        self.embed_dim = cfg.hidden_size
        kernel = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel, stride=kernel, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(dtype=target)).view(-1, self.embed_dim)


class VisionPatchMerger(nn.Module):
    """Fuse each 2x2 patch group into one token and project to the text width."""

    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.hidden_size = cfg.hidden_size * cfg.spatial_merge_unit
        self.norm = LayerNorm(cfg.hidden_size, eps=1e-6, zero_centered=cfg.zero_centered_norm)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, cfg.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.reshape(-1, self.norm.weight.numel())).reshape(-1, self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    original = (q.dtype, k.dtype)
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = ((q * cos) + (rotate_half(q) * sin)).to(original[0])
    k_embed = ((k * cos) + (rotate_half(k) * sin)).to(original[1])
    return q_embed, k_embed


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.dim = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scaling = self.head_dim**-0.5
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)

        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)

        # Images are packed into one sequence; attention never crosses an image
        # boundary, so each image is attended separately.
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(query, lengths, dim=2)
        k_splits = torch.split(key, lengths, dim=2)
        v_splits = torch.split(value, lengths, dim=2)
        outputs = [
            F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scaling)
            for q, k, v in zip(q_splits, k_splits, v_splits, strict=True)
        ]
        attn_output = torch.cat(outputs, dim=2)

        attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionMLP(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.act_fn = (
            nn.GELU(approximate="tanh") if cfg.hidden_act == "gelu_pytorch_tanh" else nn.GELU()
        )
        self.linear_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class VisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.norm1 = LayerNorm(cfg.hidden_size, eps=1e-6, zero_centered=cfg.zero_centered_norm)
        self.norm2 = LayerNorm(cfg.hidden_size, eps=1e-6, zero_centered=cfg.zero_centered_norm)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, cos, sin)
        return hidden_states + self.mlp(self.norm2(hidden_states))


def bilinear_grid_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear resampling of the learned square position grid onto each image grid.

    ``align_corners=True`` semantics, matching the reference: a grid of size 1
    maps to position 0, and the corners of an NxN grid map exactly onto the
    learned grid's corners. Returns (indices (N, 4), weights (N, 4)) so the
    caller can do ``(pos_embed(indices) * weights[..., None]).sum(1)`` in one
    gather.
    """
    side = num_grid_per_side
    indices = []
    weights = []
    for t, h, w in grid_thw.tolist():
        for _ in range(t):
            hs = (
                torch.linspace(0, side - 1, h, device=device)
                if h > 1
                else torch.zeros(1, device=device)
            )
            ws = (
                torch.linspace(0, side - 1, w, device=device)
                if w > 1
                else torch.zeros(1, device=device)
            )
            h0 = hs.floor().long().clamp(0, side - 1)
            w0 = ws.floor().long().clamp(0, side - 1)
            h1 = (h0 + 1).clamp(0, side - 1)
            w1 = (w0 + 1).clamp(0, side - 1)
            wh = (hs - h0.float()).clamp(0.0, 1.0)
            ww = (ws - w0.float()).clamp(0.0, 1.0)
            hh, wwi = torch.meshgrid(h0, w0, indexing="ij")
            hh1, ww1 = torch.meshgrid(h1, w1, indexing="ij")
            whh, www = torch.meshgrid(wh, ww, indexing="ij")
            idx = torch.stack(
                [hh * side + wwi, hh * side + ww1, hh1 * side + wwi, hh1 * side + ww1], dim=-1
            ).reshape(-1, 4)
            wgt = torch.stack(
                [
                    (1 - whh) * (1 - www),
                    (1 - whh) * www,
                    whh * (1 - www),
                    whh * www,
                ],
                dim=-1,
            ).reshape(-1, 4)
            indices.append(idx)
            weights.append(wgt)
    return torch.cat(indices, dim=0), torch.cat(weights, dim=0)


def vision_position_ids(grid_thw: torch.Tensor) -> torch.Tensor:
    """Height and width coordinates for raster-ordered patches in each frame."""
    rows = []
    for t, h, w in grid_thw.tolist():
        height, width = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        coords = torch.stack((height.flatten(), width.flatten()))
        rows.append(coords.repeat(1, t))
    return torch.cat(rows, dim=1)


def vision_attention_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    lengths = (h * w for t, h, w in grid_thw.tolist() for _ in range(t))
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32)


class VisionModel(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.spatial_merge_unit = cfg.spatial_merge_unit
        self.patch_size = cfg.patch_size
        self.patch_embed = VisionPatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.num_grid_per_side = cfg.num_grid_per_side
        self.rotary_pos_emb = VisionRotaryEmbedding(cfg)
        self.blocks = nn.ModuleList(VisionBlock(cfg) for _ in range(cfg.depth))
        self.merger = VisionPatchMerger(cfg)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """pixel_values: (sum(t*h*w), C*temporal*patch*patch). Returns (tokens, out_hidden)."""
        index, weight = bilinear_grid_indices_and_weights(
            grid_thw, self.num_grid_per_side, device=pixel_values.device
        )
        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = (self.pos_embed(index.to(pixel_values.device)) * weight[:, :, None]).sum(1)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        pos_ids = vision_position_ids(grid_thw).to(pixel_values.device)
        cos, sin = self.rotary_pos_emb(hidden_states, pos_ids)
        cu_seqlens = vision_attention_seqlens(grid_thw).to(pixel_values.device)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens=cu_seqlens, cos=cos, sin=sin)

        hidden_states = self._merge(hidden_states, grid_thw)
        return self.merger(hidden_states)

    def _merge(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Reorder the flattened (t, h, w) tokens into 2x2 spatial groups."""
        merge = self.spatial_merge_size
        hidden_size = self.cfg.hidden_size
        chunks = []
        offset = 0
        for t, h, w in grid_thw.tolist():
            count = t * h * w
            block = hidden_states[offset : offset + count]
            offset += count
            block = block.view(t, h // merge, merge, w // merge, merge, hidden_size)
            block = block.permute(0, 1, 3, 2, 4, 5).reshape(-1, merge * merge * hidden_size)
            chunks.append(block)
        return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# top-level model
# ---------------------------------------------------------------------------


class GenerationCache:
    """Incremental-decode state: conv windows and recurrent states for the linear
    layers, growing K/V for the full-attention layers.

    The two halves of the layer mix need different kinds of memory, which is the
    point of the architecture: the linear layers keep a fixed-size per-head state
    no matter how long the context gets, while the full-attention layers keep a
    KV cache. ``prefill_length`` is how far the KV tensors have already been filled.
    """

    def __init__(self, cfg: ModelConfig, batch_size: int, device: torch.device | str) -> None:
        self.cfg = cfg
        self.batch_size = batch_size
        self.device = device
        self._conv: dict[int, torch.Tensor] = {}
        self._recurrent: dict[int, torch.Tensor] = {}
        self._kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.prefill_length = 0

    # linear attention -------------------------------------------------
    def conv_state(self, layer_idx: int, conv_dim: int) -> torch.Tensor:
        state = self._conv.get(layer_idx)
        if state is None:
            keep = self.cfg.text.linear_conv_kernel_dim - 1
            state = torch.zeros(self.batch_size, conv_dim, keep, device=self.device)
            self._conv[layer_idx] = state
        return state

    def set_conv_state(self, layer_idx: int, state: torch.Tensor) -> None:
        self._conv[layer_idx] = state

    def recurrent_state(self, layer_idx: int, batch_size: int) -> torch.Tensor | None:
        state = self._recurrent.get(layer_idx)
        if state is None:
            state = torch.zeros(
                batch_size,
                self.cfg.text.linear_num_value_heads,
                self.cfg.text.linear_key_head_dim,
                self.cfg.text.linear_value_head_dim,
                dtype=torch.float32,
                device=self.device,
            )
            self._recurrent[layer_idx] = state
        return state

    def set_recurrent_state(self, layer_idx: int, state: torch.Tensor) -> None:
        self._recurrent[layer_idx] = state

    # full attention ---------------------------------------------------
    def append_kv(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._kv.get(layer_idx)
        if cached is None:
            self._kv[layer_idx] = (key, value)
            return key, value
        past_key, past_value = cached
        key = torch.cat([past_key, key], dim=2)
        value = torch.cat([past_value, value], dim=2)
        self._kv[layer_idx] = (key, value)
        return key, value


class TextModel(nn.Module):
    def __init__(self, cfg: TextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNormZeroCentered(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = TextRotaryEmbedding(cfg)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: GenerationCache | None = None,
        delta_rule: Callable | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        seq_len = inputs_embeds.shape[1]
        if position_ids is None:
            past = cache.prefill_length if cache is not None else 0
            base = torch.arange(seq_len, device=inputs_embeds.device) + past
            position_ids = base.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, *position_ids.shape)

        rope_position_ids = position_ids[-3:]

        hidden_states = inputs_embeds
        cos, sin = self.rotary_emb(hidden_states, rope_position_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training and cache is None:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    cos,
                    sin,
                    attention_mask,
                    None,
                    delta_rule,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    cos,
                    sin,
                    attention_mask=attention_mask,
                    cache=cache,
                    delta_rule=delta_rule,
                )

        if cache is not None:
            cache.prefill_length += seq_len
        return self.norm(hidden_states)


class Model(nn.Module):
    """The multimodal trunk: vision tower + text decoder, as named in the checkpoint."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.visual = VisionModel(cfg.vision)
        self.language_model = TextModel(cfg.text)

    def get_image_features(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        dtype = self.visual.patch_embed.proj.weight.dtype
        return self.visual(pixel_values.type(dtype), grid_thw)

    def merge_image_features(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        mask = input_ids == self.cfg.image_token_id
        expected = int(mask.sum())
        if expected != image_embeds.shape[0]:
            raise ValueError(
                f"{expected} image placeholder tokens but {image_embeds.shape[0]} image features"
            )
        return inputs_embeds.masked_scatter(
            mask.unsqueeze(-1), image_embeds.to(inputs_embeds.dtype)
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        cache: GenerationCache | None = None,
        delta_rule: Callable | None = None,
    ) -> torch.Tensor:
        inputs_embeds = None
        if input_ids is not None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            if grid_thw is None:
                raise ValueError("pixel_values requires grid_thw")
            image_embeds = self.get_image_features(pixel_values, grid_thw)
            if input_ids is None:
                raise ValueError("image features require input_ids so placeholders can be located")
            inputs_embeds = self.merge_image_features(inputs_embeds, input_ids, image_embeds)
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            delta_rule=delta_rule,
        )


class CafezinLM(nn.Module):
    """The whole model: loss-bearing head over `Model` plus the MTP module.

    ``lm_head`` shares the token embedding (the checkpoint stores no
    ``lm_head.weight``), so a trained model's file layout is exactly the
    reference's.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.model = Model(self.cfg)
        self.mtp = MTP(self.cfg.text)
        self.lm_head = nn.Linear(self.cfg.text.hidden_size, self.cfg.text.vocab_size, bias=False)
        if self.cfg.lm_head_tied:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        self._init_gated_delta()

    # -- initialisation --------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.text.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNormZeroCentered):
            # zero => effective scale of exactly 1
            nn.init.zeros_(module.weight)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _init_gated_delta(self) -> None:
        """Per-block overrides the reference applies after the generic pass."""
        for module in self.modules():
            if isinstance(module, GatedDeltaNet):
                nn.init.ones_(module.dt_bias)
                # log(A) with A ~ U(0.01, 16): the lower bound stays off zero so
                # exp(A_log) never collapses the memory to nothing.
                noise = torch.empty(module.num_v_heads).uniform_(0.01, 16).log_()
                with torch.no_grad():
                    module.A_log.copy_(noise.to(module.A_log.device))
            elif isinstance(module, RMSNormGated):
                nn.init.ones_(module.weight)

    # -- helpers ---------------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled
        self.model.language_model.gradient_checkpointing = enabled

    def compute_dtype(self) -> torch.dtype:
        for p in self.parameters():
            if p.is_floating_point():
                return p.dtype
        return torch.float32

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        cache: GenerationCache | None = None,
        mtp_weight: float = 0.0,
        delta_rule: Callable | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            cache=cache,
            delta_rule=delta_rule,
        )
        logits = self.lm_head(hidden_states)
        if targets is None:
            return logits, None
        if targets.shape != input_ids.shape:
            raise ValueError("targets must contain one next-token label per input token")

        # TokenDataset supplies next-token targets aligned with the input positions.
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            targets.flatten(),
            ignore_index=-1,
        )
        if mtp_weight > 0 and self.cfg.mtp_num_hidden_layers > 0 and input_ids.size(1) > 1:
            mtp_logits = self.mtp_forward(
                hidden_states, input_ids, position_ids=position_ids, cache=cache
            )
            mtp_loss = F.cross_entropy(
                mtp_logits.flatten(0, 1),
                targets[:, 1:].reshape(-1),
                ignore_index=-1,
            )
            loss = loss + mtp_weight * mtp_loss
        return logits, loss

    def mtp_forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cache: GenerationCache | None = None,
    ) -> torch.Tensor:
        """Logits for the token *two* positions ahead, from trunk hidden states."""
        trunk = hidden_states[:, : hidden_states.size(1) - 1]
        next_ids = input_ids[:, 1 : hidden_states.size(1)]
        next_embeds = self.model.language_model.embed_tokens(next_ids)

        if position_ids is None:
            base = torch.arange(trunk.size(1), device=trunk.device)
            position_ids = base.view(1, 1, -1).expand(4, trunk.size(0), -1)
        if position_ids.ndim == 3:
            position_ids = position_ids[-3:]
        cos, sin = self.model.language_model.rotary_emb(trunk, position_ids[..., : trunk.size(1)])

        fused = self.mtp(trunk, next_embeds, cos, sin)
        return self.lm_head(fused)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = 20,
        top_p: float | None = 0.95,
        eos_token_id: int | None = None,
        vocab_limit: int | None = None,
    ) -> torch.Tensor:
        """Incremental decode using the linear-attention state cache.

        Prefill runs the chunked kernel; each new token then runs the recurrent
        one against the carried state, which is the behaviour the architecture
        exists for — constant memory per linear layer regardless of context.
        """
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must contain at least one token per prompt")
        cache = GenerationCache(self.cfg, input_ids.size(0), device)
        eos_token_id = eos_token_id if eos_token_id is not None else self.cfg.text.eos_token_id

        logits, _ = self.forward(input_ids, cache=cache, delta_rule=chunk_gated_delta_rule)
        out = input_ids
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            step_logits = logits[:, -1, :vocab_limit].float()
            if temperature <= 0:
                next_token = step_logits.argmax(dim=-1, keepdim=True)
            else:
                step_logits = step_logits / temperature
                if top_k:
                    kth = torch.topk(step_logits, min(top_k, step_logits.size(-1)), dim=-1).values[
                        :, -1:
                    ]
                    step_logits = step_logits.masked_fill(step_logits < kth, float("-inf"))
                if top_p and 0 < top_p < 1:
                    sorted_logits, sorted_idx = torch.sort(step_logits, descending=True, dim=-1)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    remove = probs.cumsum(dim=-1) - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    step_logits = torch.full_like(step_logits, float("-inf")).scatter(
                        -1, sorted_idx, sorted_logits
                    )
                next_token = torch.multinomial(torch.softmax(step_logits, dim=-1), num_samples=1)

            next_token = torch.where(
                finished[:, None], torch.full_like(next_token, eos_token_id), next_token
            )
            finished |= next_token.squeeze(1).eq(eos_token_id)
            out = torch.cat([out, next_token], dim=1)
            if finished.all():
                break
            logits, _ = self.forward(next_token, cache=cache, delta_rule=recurrent_gated_delta_rule)
        return out


def build_model(cfg: ModelConfig | None = None, seed: int | None = None) -> CafezinLM:
    """Instantiate the model; `seed` makes the initialisation reproducible."""
    if seed is not None:
        torch.manual_seed(seed)
    return CafezinLM(cfg)


def parameter_count_breakdown(model: CafezinLM) -> dict[str, int]:
    totals = {"language_model": 0, "visual": 0, "mtp": 0, "embedding": 0}
    seen: set[int] = set()
    for name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in seen:  # skip a tied lm_head
            continue
        seen.add(id(param))
        if name.startswith("model.visual."):
            totals["visual"] += param.numel()
        elif name.startswith("mtp."):
            totals["mtp"] += param.numel()
        elif name.endswith("embed_tokens.weight"):
            totals["embedding"] += param.numel()
            totals["language_model"] += param.numel()
        else:
            totals["language_model"] += param.numel()
    return totals


def parameter_count_estimate(text: TextConfig) -> int:
    """Text-decoder parameter count without instantiating anything (for budgeting).

    Counts the tied embedding once, since no separate ``lm_head`` is stored.
    """
    h = text.hidden_size
    emb = text.vocab_size * h
    mlp = 3 * h * text.intermediate_size

    linear = (
        text.linear_conv_dim * h  # in_proj_qkv
        + text.linear_value_dim * h  # in_proj_z
        + 2 * text.linear_num_value_heads * h  # in_proj_b, in_proj_a
        + h * text.linear_value_dim  # out_proj
        + text.linear_conv_dim * text.linear_conv_kernel_dim  # depthwise conv
        + 2 * text.linear_num_value_heads  # dt_bias, A_log
        + text.linear_value_head_dim  # gated norm (per-head width, not per-head count)
    )
    full = (
        text.num_attention_heads * text.head_dim * 2 * h  # q_proj (query + gate)
        + 2 * text.num_key_value_heads * text.head_dim * h  # k_proj, v_proj
        + text.num_attention_heads * text.head_dim * h  # o_proj
        + 2 * text.head_dim  # q_norm, k_norm
    )
    norms = 2 * h  # input_layernorm, post_attention_layernorm
    return (
        emb
        + text.num_linear_attention_layers * (mlp + linear + norms)
        + text.num_full_attention_layers * (mlp + full + norms)
        + h  # final norm
    )


def parameter_count_estimate_full(cfg: ModelConfig) -> int:
    """Whole model: text decoder + MTP (shares embeddings) + vision tower."""
    text, vision = cfg.text, cfg.vision
    h, inter = text.hidden_size, text.intermediate_size
    mtp = (
        2 * h * h  # fc: 2h -> h
        + 3 * h * inter  # one full decoder layer's MLP
        + text.num_attention_heads * text.head_dim * 2 * h
        + 2 * text.num_key_value_heads * text.head_dim * h
        + text.num_attention_heads * text.head_dim * h
        + 2 * text.head_dim
        + 2 * h  # that layer's two norms
        + 3 * h  # pre_fc_norm_hidden, pre_fc_norm_embedding, final norm
    ) * text.mtp_num_hidden_layers
    vh, vinter = vision.hidden_size, vision.intermediate_size
    patch = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2 * vh + vh
    block = (
        4 * vh * vh  # qkv (3x) + proj
        + 4 * vh  # their biases
        + 2 * vh * vinter
        + vh
        + vinter  # mlp fc1/fc2 with biases
        + 4 * vh  # two LayerNorms
    )
    merge = vision.spatial_merge_unit * vh
    merger = (
        merge * merge + merge + vision.out_hidden_size * merge + vision.out_hidden_size + 2 * vh
    )
    vision_total = patch + vision.num_position_embeddings * vh + vision.depth * block + merger
    return parameter_count_estimate(text) + mtp + vision_total
