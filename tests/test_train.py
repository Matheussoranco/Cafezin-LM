"""End-to-end smoke test: tokenizer -> prepare -> train a few steps -> generate."""
import pytest
import torch

from slm.config import ModelConfig, TrainConfig
from slm.data import iter_texts, prepare
from slm.generate import complete
from slm.tokenizer import SLMTokenizer
from slm.train import load_model, train

TEXT = "\n\n".join(f"The fox number {i} jumps over the dog number {i}." for i in range(300))


@pytest.mark.slow
def test_end_to_end(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    tok_path = tmp_path / "data" / "tokenizer.json"
    tok = SLMTokenizer.train(iter_texts(inputs=[str(corpus)]), vocab_size=512, save_path=tok_path)
    prepare(tok, tmp_path / "data", val_fraction=0.1, inputs=[str(corpus)])

    model_cfg = ModelConfig(vocab_size=tok.vocab_size, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    train_cfg = TrainConfig(
        batch_size=4, grad_accum_steps=1, seq_len=64, max_steps=6,
        warmup_steps=2, eval_interval=3, eval_iters=2, log_interval=2,
    )
    best = train(model_cfg, train_cfg, tmp_path / "data", tmp_path / "run", tokenizer_path=str(tok_path))
    assert best < float("inf")
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "last.pt").exists()

    model, ckpt = load_model(tmp_path / "run" / "best.pt", device="cpu")
    assert ckpt["tokenizer_path"] == str(tok_path)
    text = complete(model, tok, "The fox", "cpu", max_new_tokens=10, temperature=0.0)
    assert isinstance(text, str)

    # resume picks up from the saved step
    train_cfg.max_steps = 8
    train(model_cfg, train_cfg, tmp_path / "data", tmp_path / "run", resume=True, tokenizer_path=str(tok_path))
