"""The README quickstart, end to end, through the real CLI on the `nano` preset.

Runs the documented sequence — tokenizer -> prepare -> train -> resume -> info ->
generate -> quantize — against a small synthetic corpus and a handful of steps.
Marked `slow`; run it with `pytest -m slow`.
"""

import json
import random
from pathlib import Path

import pytest

from pico_lm.cli import main

NAMES = ["Ana", "Bruno", "Clara", "Diego", "Elena", "Felipe"]
ANIMALS = ["fox", "dog", "cat", "owl", "bear", "frog"]
PLACES = ["the garden", "the forest", "the river", "the hill"]


def _write_corpus(path: Path, n_docs: int = 1500) -> None:
    rng = random.Random(0)
    docs = []
    for _ in range(n_docs):
        name, animal, place = rng.choice(NAMES), rng.choice(ANIMALS), rng.choice(PLACES)
        docs.append(
            f"{name} went to {place} one morning. There {name} met a small {animal}. "
            f"The {animal} was hungry, so {name} shared some bread. "
            f"They played until the sun went down, and then {name} walked home happy."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(docs), encoding="utf-8")


@pytest.mark.slow
def test_quickstart_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_corpus(tmp_path / "corpus" / "stories.txt")

    # 1. tokenizer
    main(
        [
            "tokenizer",
            "--input",
            "corpus/*.txt",
            "--vocab-size",
            "1024",
            "--out",
            "data/tokenizer.json",
        ]
    )
    assert (tmp_path / "data" / "tokenizer.json").exists()

    # 2. prepare
    main(
        [
            "prepare",
            "--input",
            "corpus/*.txt",
            "--tokenizer",
            "data/tokenizer.json",
            "--out",
            "data",
            "--val-fraction",
            "0.05",
        ]
    )
    meta = json.loads((tmp_path / "data" / "meta.json").read_text(encoding="utf-8"))
    assert meta["train_tokens"] > 0
    assert meta["val_tokens"] > 0, "prepare must always produce a usable validation split"

    # 3. train a handful of steps
    train_args = [
        "train",
        "--preset",
        "nano",
        "--data",
        "data",
        "--out",
        "runs/nano",
        "--batch-size",
        "8",
        "--grad-accum-steps",
        "1",
        "--seq-len",
        "128",
        "--warmup-steps",
        "2",
        "--eval-interval",
        "5",
        "--eval-iters",
        "2",
        "--log-interval",
        "5",
    ]
    main([*train_args, "--max-steps", "10"])
    run = tmp_path / "runs" / "nano"
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    assert (run / "tokenizer.json").exists()
    assert (run / "log.csv").read_text(encoding="utf-8").count("\n") > 1

    # 4. resume and train further
    main([*train_args, "--max-steps", "20", "--resume"])
    out = capsys.readouterr().out
    assert "resumed from step 10" in out

    # 5. inspect + generate + quantize
    main(["info", "--ckpt", "runs/nano/best.pt"])
    main(
        [
            "generate",
            "--ckpt",
            "runs/nano/best.pt",
            "--prompt",
            "Ana went to",
            "--tokens",
            "16",
            "--temperature",
            "0.0",
        ]
    )
    main(["quantize", "--ckpt", "runs/nano/best.pt", "--out", "runs/nano-int8.pt"])
    assert (tmp_path / "runs" / "nano-int8.pt").exists()

    out = capsys.readouterr().out
    assert "params:" in out and "Ana went to" in out


@pytest.mark.slow
def test_generate_works_after_data_dir_is_gone(tmp_path, monkeypatch):
    """A run directory must be self-sufficient for inference."""
    import shutil

    from pico_lm.generate import complete, load_for_inference

    monkeypatch.chdir(tmp_path)
    _write_corpus(tmp_path / "corpus" / "stories.txt", n_docs=400)
    main(
        [
            "tokenizer",
            "--input",
            "corpus/*.txt",
            "--vocab-size",
            "512",
            "--out",
            "data/tokenizer.json",
        ]
    )
    main(
        [
            "prepare",
            "--input",
            "corpus/*.txt",
            "--tokenizer",
            "data/tokenizer.json",
            "--out",
            "data",
            "--val-fraction",
            "0.1",
        ]
    )
    main(
        [
            "train",
            "--preset",
            "nano",
            "--data",
            "data",
            "--out",
            "runs/nano",
            "--batch-size",
            "4",
            "--grad-accum-steps",
            "1",
            "--seq-len",
            "64",
            "--max-steps",
            "4",
            "--warmup-steps",
            "1",
            "--eval-interval",
            "4",
            "--eval-iters",
            "2",
            "--log-interval",
            "4",
        ]
    )

    shutil.rmtree(tmp_path / "data")
    model, tok, device = load_for_inference(tmp_path / "runs" / "nano" / "best.pt", device="cpu")
    text = complete(model, tok, "Ana went to", device, max_new_tokens=8, temperature=0.0)
    assert isinstance(text, str)
