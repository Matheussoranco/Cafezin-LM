"""Library-level training smoke test: tokenizer -> prepare -> train -> resume -> generate."""
from pathlib import Path

import pytest

from pico_lm.config import ModelConfig, TrainConfig
from pico_lm.data import iter_texts, prepare
from pico_lm.generate import complete, load_for_inference
from pico_lm.tokenizer import PicoTokenizer
from pico_lm.train import cosine_lr, load_model, train

TEXT = "\n\n".join(f"The fox number {i} jumps over the dog number {i}." for i in range(300))


def _tiny_cfgs(vocab_size: int, max_steps: int = 6) -> tuple[ModelConfig, TrainConfig]:
    model_cfg = ModelConfig(
        vocab_size=vocab_size, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64
    )
    train_cfg = TrainConfig(
        batch_size=4, grad_accum_steps=1, seq_len=64, max_steps=max_steps,
        warmup_steps=2, eval_interval=3, eval_iters=2, log_interval=2,
    )
    return model_cfg, train_cfg


@pytest.mark.slow
def test_end_to_end(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    tok_path = tmp_path / "data" / "tokenizer.json"
    tok = PicoTokenizer.train(iter_texts(inputs=[str(corpus)]), vocab_size=512, save_path=tok_path)
    prepare(tok, tmp_path / "data", val_fraction=0.1, inputs=[str(corpus)])

    model_cfg, train_cfg = _tiny_cfgs(tok.vocab_size)
    run = tmp_path / "run"
    best = train(model_cfg, train_cfg, tmp_path / "data", run, tokenizer_path=str(tok_path))
    assert best < float("inf")
    assert (run / "best.pt").exists()
    assert (run / "last.pt").exists()
    # training stages the tokenizer beside the checkpoints
    assert (run / "tokenizer.json").exists()

    model, ckpt = load_model(run / "best.pt", device="cpu")
    assert Path(ckpt["tokenizer_path"]) == (run / "tokenizer.json").resolve()
    text = complete(model, tok, "The fox", "cpu", max_new_tokens=10, temperature=0.0)
    assert isinstance(text, str)

    # a checkpoint alone is enough to load a working generator
    _, tok2, _ = load_for_inference(run / "best.pt", device="cpu")
    assert tok2.encode("The fox") == tok.encode("The fox")

    # resume picks up from the saved step
    train_cfg.max_steps = 8
    train(model_cfg, train_cfg, tmp_path / "data", run, resume=True, tokenizer_path=str(tok_path))
    _, ckpt2 = load_model(run / "last.pt", device="cpu")
    assert ckpt2["step"] == 8


@pytest.mark.slow
def test_resume_keeps_best_val(tmp_path):
    """last.pt must record the best val seen so far, not the value from before the eval."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    tok_path = tmp_path / "data" / "tokenizer.json"
    tok = PicoTokenizer.train(iter_texts(inputs=[str(corpus)]), vocab_size=512, save_path=tok_path)
    prepare(tok, tmp_path / "data", val_fraction=0.1, inputs=[str(corpus)])

    model_cfg, train_cfg = _tiny_cfgs(tok.vocab_size, max_steps=3)
    run = tmp_path / "run"
    best = train(model_cfg, train_cfg, tmp_path / "data", run, tokenizer_path=str(tok_path))
    _, last = load_model(run / "last.pt", device="cpu")
    _, best_ckpt = load_model(run / "best.pt", device="cpu")
    assert last["best_val"] == pytest.approx(best)
    assert best_ckpt["best_val"] == pytest.approx(best)


def test_resume_without_checkpoint_starts_fresh(tmp_path, capsys):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    tok_path = tmp_path / "data" / "tokenizer.json"
    tok = PicoTokenizer.train(iter_texts(inputs=[str(corpus)]), vocab_size=512, save_path=tok_path)
    prepare(tok, tmp_path / "data", val_fraction=0.1, inputs=[str(corpus)])

    model_cfg, train_cfg = _tiny_cfgs(tok.vocab_size, max_steps=2)
    train_cfg.eval_interval = 2
    train(model_cfg, train_cfg, tmp_path / "data", tmp_path / "run", resume=True,
          tokenizer_path=str(tok_path))
    assert "starting from scratch" in capsys.readouterr().out


def test_cosine_lr_schedule():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    assert cosine_lr(0, cfg) == pytest.approx(cfg.lr / 10)
    assert cosine_lr(9, cfg) == pytest.approx(cfg.lr)
    assert cosine_lr(10, cfg) == pytest.approx(cfg.lr)
    mid = cosine_lr(55, cfg)
    assert cfg.min_lr < mid < cfg.lr
    assert cosine_lr(99, cfg) == pytest.approx(cfg.min_lr, abs=1e-5)
    assert cosine_lr(100, cfg) == pytest.approx(cfg.min_lr)
    # no warmup must not divide by zero
    assert cosine_lr(0, TrainConfig(warmup_steps=0, max_steps=10)) > 0
