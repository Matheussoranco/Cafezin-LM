from types import MethodType

import pytest
import torch

from pico_lm.config import ModelConfig
from pico_lm.model import KVCache, Transformer


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    m = Transformer(cfg)
    m.eval()
    return m


def test_forward_shapes_and_loss(model, cfg):
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_kv_cache_matches_full_forward(model, cfg):
    """Incremental decoding with the cache must reproduce full-sequence logits."""
    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        full_logits, _ = model(x)
        cache = KVCache(cfg, batch_size=2, device="cpu", dtype=torch.float32)
        prefill_logits, _ = model(x[:, :8], cache=cache)
        step_logits = []
        for i in range(8, 12):
            lg, _ = model(x[:, i : i + 1], cache=cache)
            step_logits.append(lg)
    torch.testing.assert_close(prefill_logits, full_logits[:, :8], atol=1e-4, rtol=1e-4)
    for i, lg in enumerate(step_logits):
        torch.testing.assert_close(lg, full_logits[:, 8 + i : 9 + i], atol=1e-4, rtol=1e-4)


def test_kv_cache_multi_token_continuation(model, cfg):
    """Feeding several tokens at once over an existing cache uses the offset mask."""
    torch.manual_seed(2)
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        full_logits, _ = model(x)
        cache = KVCache(cfg, batch_size=1, device="cpu", dtype=torch.float32)
        model(x[:, :4], cache=cache)
        cont_logits, _ = model(x[:, 4:], cache=cache)
    torch.testing.assert_close(cont_logits, full_logits[:, 4:], atol=1e-4, rtol=1e-4)


def test_generate_respects_eos_and_length(model, cfg):
    x = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model.generate(x, max_new_tokens=10, temperature=0.0)
    assert out.shape[0] == 1
    assert 5 < out.shape[1] <= 15
    assert out.shape[1] <= cfg.max_seq_len


def test_generate_never_exceeds_context(model, cfg):
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    out = model.generate(x, max_new_tokens=20, temperature=0.0)
    assert out.shape[1] <= cfg.max_seq_len


def test_top_p_near_zero_reduces_to_greedy(model, cfg):
    """Nucleus sampling with top_p -> 0 must keep exactly the top token.

    Regression guard for the classic off-by-one in the cumulative-sum boundary:
    the code excludes token i when the *exclusive* prefix sum before it
    (cum - probs) exceeds top_p, which always keeps the top token itself. An
    inclusive-sum version (`cum > top_p`) would wrongly drop the top token too
    at this threshold, leaving every logit at -inf and breaking sampling.
    """
    torch.manual_seed(7)
    x = torch.randint(0, cfg.vocab_size, (1, 5))
    greedy = model.generate(x, max_new_tokens=6, temperature=0.0)
    torch.manual_seed(7)
    nucleus = model.generate(x, max_new_tokens=6, temperature=1.0, top_k=None, top_p=1e-6)
    assert torch.equal(greedy, nucleus)


def test_loss_decreases_when_overfitting(cfg):
    torch.manual_seed(3)
    model = Transformer(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.roll(x, -1, dims=1)
    first = None
    for _ in range(40):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 0.5, f"loss did not drop: {first:.3f} -> {loss.item():.3f}"


def test_weight_tying(model):
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_gradient_checkpointing_preserves_loss_and_backprop(cfg):
    torch.manual_seed(11)
    plain = Transformer(cfg).train()
    checked = Transformer(cfg).train()
    checked.load_state_dict(plain.state_dict())
    checked.set_gradient_checkpointing(True)
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    y = torch.randint(0, cfg.vocab_size, (2, 24))

    _, plain_loss = plain(x, y)
    _, checked_loss = checked(x, y)
    torch.testing.assert_close(checked_loss, plain_loss)
    checked_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in checked.parameters())


def test_batched_generation_keeps_finished_rows_at_eos(model, cfg):
    eos_id, ordinary_id = 2, 3

    def fake_forward(self, idx, targets=None, cache=None):
        del targets
        b, t = idx.shape
        old_pos = cache.pos
        logits = torch.full((b, t, cfg.vocab_size), -100.0)
        logits[0, -1, eos_id] = 100.0
        logits[1, -1, ordinary_id if old_pos == 0 else eos_id] = 100.0
        cache.pos += t
        return logits, None

    original = model.forward
    model.forward = MethodType(fake_forward, model)
    try:
        out = model.generate(
            torch.ones((2, 2), dtype=torch.long),
            max_new_tokens=4,
            temperature=0.0,
            eos_id=eos_id,
        )
    finally:
        model.forward = original

    assert out[:, 2:].tolist() == [[eos_id, eos_id], [ordinary_id, eos_id]]


def test_model_config_rejects_invalid_rope_geometry():
    with pytest.raises(ValueError, match="even for RoPE"):
        ModelConfig(dim=60, n_heads=4, n_kv_heads=2)
