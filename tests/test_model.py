import torch
import pytest

from slm.config import ModelConfig
from slm.model import KVCache, Transformer


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
