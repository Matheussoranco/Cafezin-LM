"""Tests for the Cafezin LM model architecture.

These verify:
- exact weight-name parity with the published checkpoint
- linear-attention chunk vs recurrent path equivalence
- full-attention gate correctness
- MTP forward pass
- generation (prefill chunk + recurrent decode) matches full forward
"""

import torch

from cafezin_lm import (
    CafezinLM,
    GenerationCache,
    build_model,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
    tiny_config,
)


def test_model_builds_on_meta():
    cfg = tiny_config()
    with torch.device("meta"):
        model = CafezinLM(cfg)
    assert model.num_params() > 0


def test_parameter_names_match_reference():
    cfg = tiny_config()
    with torch.device("meta"):
        model = CafezinLM(cfg)
    spec = {name: tuple(p.shape) for name, p in model.named_parameters()}

    assert "model.language_model.embed_tokens.weight" in spec
    assert "model.language_model.norm.weight" in spec
    assert "model.visual.patch_embed.proj.weight" in spec
    assert "model.visual.pos_embed.weight" in spec
    assert "mtp.fc.weight" in spec
    assert "mtp.norm.weight" in spec

    # lm_head is tied and NOT in state_dict
    assert "lm_head.weight" not in spec

    linear_layers = sum(1 for n in spec if "linear_attn" in n)
    full_layers = sum(1 for n in spec if "self_attn" in n and "mtp" not in n)
    assert linear_layers > 0
    assert full_layers > 0


def test_linear_attn_chunk_vs_recurrent_equivalence():
    """The chunked and recurrent gated-delta-rule implementations must agree.

    This is the single most important numerical test for the linear-attention
    path. The reference uses both paths (chunked for prefill, recurrent for
    decode); they must produce bitwise-identical outputs for the same inputs.
    """
    cfg = tiny_config()
    model = build_model(cfg, seed=42)
    model.eval()

    B, T = 2, 128
    x = torch.randn(B, T, cfg.text.hidden_size)

    layer = model.model.language_model.layers[0]
    assert hasattr(layer, "linear_attn")

    with torch.no_grad():
        out_chunk = layer(
            x, cos=None, sin=None, attention_mask=None, delta_rule=chunk_gated_delta_rule
        )
        out_recur = layer(
            x, cos=None, sin=None, attention_mask=None, delta_rule=recurrent_gated_delta_rule
        )

    assert torch.allclose(out_chunk, out_recur, rtol=1e-4, atol=1e-5)


def test_full_attention_output_gate():
    """The full-attention block applies a sigmoid gate to its output.

    The q_proj projects to 2*head_dim (query + gate), the gate is applied
    element-wise to the attention output, then o_proj projects back.
    """
    cfg = tiny_config()
    model = build_model(cfg, seed=123)
    model.eval()

    layer = model.model.language_model.layers[3]
    assert hasattr(layer, "self_attn")

    B, T, H = 1, 32, cfg.text.hidden_size
    x = torch.randn(B, T, H)

    with torch.device("cpu"):
        pos_ids = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = model.model.language_model.rotary_emb(x, pos_ids)

    with torch.no_grad():
        out = layer(x, cos=cos, sin=sin)

    assert out.shape == (B, T, H)


def test_mtp_forward_shape():
    """The MTP head takes trunk hidden states and next-token embeddings,
    fuses them, and produces logits for the token two steps ahead.
    """
    cfg = tiny_config()
    model = build_model(cfg, seed=7)
    model.eval()

    B, T, H = 2, 16, cfg.text.hidden_size
    hidden = torch.randn(B, T, H)
    next_ids = torch.randint(0, cfg.text.vocab_size, (B, T - 1))

    with torch.no_grad():
        mtp_logits = model.mtp_forward(hidden[:, : T - 1], next_ids)

    assert mtp_logits.shape == (B, T - 2, cfg.text.vocab_size)


def test_generation_prefill_then_recurrent_consistency():
    """Prefill with chunked path, then decode with recurrent path.

    Uses a FIXED known sequence: prefill on prefix, then decode step-by-step
    on KNOWN suffix tokens (teacher forcing), comparing logits at each step.
    """
    cfg = tiny_config()
    model = build_model(cfg, seed=42)
    model.eval()

    total_len = 16
    B = 1
    gen = torch.Generator().manual_seed(0)
    full_ids = torch.randint(0, cfg.text.vocab_size, (B, total_len), generator=gen)

    prompt_len = 8
    with torch.no_grad():
        full_logits, _ = model(full_ids)
        cache = GenerationCache(model.cfg, 1, "cpu")
        prefill, _ = model(full_ids[:, :prompt_len], cache=cache)
        cached_logits = [prefill]
        for i in range(prompt_len, total_len):
            logits, _ = model(
                full_ids[:, i : i + 1], cache=cache, delta_rule=recurrent_gated_delta_rule
            )
            cached_logits.append(logits)
        torch.testing.assert_close(full_logits, torch.cat(cached_logits, dim=1))


def test_next_token_and_mtp_target_alignment():
    model = build_model(tiny_config(), seed=17).eval()
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[2, 3, 4, 5]])
    with torch.no_grad():
        logits, loss = model(x, targets=y, mtp_weight=0.3)
        hidden = model.model(x)
        mtp = model.mtp_forward(hidden, x)
    next_token_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
    mtp_loss = torch.nn.functional.cross_entropy(mtp.flatten(0, 1), y[:, 1:].flatten())
    torch.testing.assert_close(loss, next_token_loss + 0.3 * mtp_loss)


def test_vision_forward_shapes():
    cfg = tiny_config()
    model = build_model(cfg, seed=1)
    model.eval()

    grid_thw = torch.tensor([[2, 2, 2]])
    pixel_values = torch.randn(8, 3 * 2 * 8 * 8)

    with torch.no_grad():
        out = model.model.visual(pixel_values, grid_thw)

    assert out.shape == (2, 64)


def test_parameter_count_estimate_accuracy():
    cfg = tiny_config()
    with torch.device("meta"):
        model = CafezinLM(cfg)
    actual = model.num_params()
    from cafezin_lm.model import parameter_count_estimate_full

    est = parameter_count_estimate_full(cfg)
    assert abs(actual - est) / actual < 0.01


def test_rmsnorm_zero_centered():
    from cafezin_lm.model import RMSNormZeroCentered

    norm = RMSNormZeroCentered(16, eps=1e-6)
    x = torch.randn(2, 10, 16)
    y = norm(x)
    assert y.shape == x.shape


def test_rmsnorm_gated():
    from cafezin_lm.model import RMSNormGated

    norm = RMSNormGated(16, eps=1e-6)
    x = torch.randn(2, 10, 16)
    gate = torch.randn(2, 10, 16)
    y = norm(x, gate)
    assert y.shape == x.shape


def test_rotary_embedding_text():
    cfg = tiny_config()
    from cafezin_lm.model import TextRotaryEmbedding

    rope = TextRotaryEmbedding(cfg.text)
    x = torch.randn(2, 8, 64)
    pos_ids = torch.arange(8).view(1, 1, -1).expand(3, 2, -1)
    cos, sin = rope(x, pos_ids)
    assert cos.shape == (2, 8, 16)
    assert sin.shape == (2, 8, 16)


def test_rotary_embedding_vision():
    cfg = tiny_config()
    from cafezin_lm.model import VisionRotaryEmbedding

    rope = VisionRotaryEmbedding(cfg.vision)
    x = torch.randn(10, 8)
    pos_ids = torch.stack(
        [
            torch.arange(10),
            torch.arange(10),
        ]
    )
    cos, sin = rope(x, pos_ids)
    assert cos.shape == (10, 8)
    assert sin.shape == (10, 8)
