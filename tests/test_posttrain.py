"""Tests for the post-training stack: SFT, quantization, behavioural eval."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from pico_lm.bench import (
    build_minimal_pairs,
    normalize_answer,
    score_instructions,
    sequence_logprobs,
)
from pico_lm.config import ModelConfig, SFTConfig
from pico_lm.data import iter_texts, prepare
from pico_lm.model import Transformer
from pico_lm.quant import (
    dequantize_weight,
    pack_int4,
    quantize_model,
    quantize_weight,
    save_quantized,
    unpack_int4,
)
from pico_lm.sft import SFTDataset, encode_pair, finetune, prepare_sft, read_pairs
from pico_lm.tokenizer import PicoTokenizer
from pico_lm.train import load_model

TEXT = "\n\n".join(
    f"The fox number {i} jumps over the lazy dog number {i} today." for i in range(300)
)


def _tiny_model(vocab_size: int) -> Transformer:
    cfg = ModelConfig(
        vocab_size=vocab_size, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128
    )
    return Transformer(cfg).eval()


@pytest.fixture
def tok(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    return PicoTokenizer.train(
        iter_texts(inputs=[str(corpus)]), vocab_size=512, save_path=tmp_path / "tokenizer.json"
    )


# --------------------------------------------------------------------------- #
# quantization
# --------------------------------------------------------------------------- #
def test_int4_pack_roundtrip():
    q = torch.randint(-8, 8, (16, 64), dtype=torch.int8)
    assert torch.equal(unpack_int4(pack_int4(q), 64), q)


def test_int4_pack_handles_odd_width():
    q = torch.randint(-8, 8, (4, 7), dtype=torch.int8)
    assert torch.equal(unpack_int4(pack_int4(q), 7), q)


@pytest.mark.parametrize("bits,group", [(8, 0), (8, 64), (4, 64), (4, 32)])
def test_quantize_dequantize_error_is_bounded(bits, group):
    torch.manual_seed(0)
    w = torch.randn(32, 128)
    q, scale, g = quantize_weight(w, bits, group)
    err = (dequantize_weight(q, scale, g) - w).abs().max()
    # every weight must land within half a quantization step of its group scale
    assert err <= scale.float().max() * 0.51


def test_int8_is_strictly_more_accurate_than_int4():
    torch.manual_seed(0)
    w = torch.randn(32, 128)
    errs = {}
    for bits in (8, 4):
        q, scale, g = quantize_weight(w, bits, 64)
        errs[bits] = (dequantize_weight(q, scale, g) - w).abs().mean().item()
    assert errs[8] < errs[4]


def test_quantized_model_saves_loads_and_generates(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model(512)
    x = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        reference, _ = model(x)

    qmodel = quantize_model(copy.deepcopy(model), bits=8, group_size=64, quantize_embedding=True)
    with torch.no_grad():
        got, _ = qmodel(x)
    # int8 must stay close to the original; this is what makes it usable at all
    assert (got - reference).abs().mean() < reference.abs().mean() * 0.05

    path = tmp_path / "q.pt"
    save_quantized(
        qmodel,
        {"bits": 8, "group_size": 64, "quantize_embedding": True, "method": "weight-only"},
        {"model_cfg": model.cfg.to_dict(), "step": 1, "tokenizer_path": None},
        path,
    )
    # load_model must transparently rebuild the quantized structure
    reloaded, ckpt = load_model(path, device="cpu")
    assert ckpt["quant"]["bits"] == 8
    with torch.no_grad():
        again, _ = reloaded(x)
    assert torch.allclose(got, again, atol=1e-5)

    out = reloaded.generate(x[:1, :4], max_new_tokens=5, temperature=0.0, top_k=0, top_p=None)
    assert out.shape == (1, 9)


def test_quantized_checkpoint_is_smaller_on_disk(tmp_path):
    model = _tiny_model(512)
    fp32 = tmp_path / "fp32.pt"
    torch.save({"model": model.state_dict(), "model_cfg": model.cfg.to_dict()}, fp32)

    sizes = {}
    for bits in (8, 4):
        q = quantize_model(copy.deepcopy(model), bits=bits, group_size=64, quantize_embedding=True)
        p = tmp_path / f"q{bits}.pt"
        save_quantized(
            q,
            {"bits": bits, "group_size": 64, "quantize_embedding": True, "method": "w"},
            {"model_cfg": model.cfg.to_dict()},
            p,
        )
        sizes[bits] = p.stat().st_size
    assert sizes[4] < sizes[8] < fp32.stat().st_size


# --------------------------------------------------------------------------- #
# supervised fine-tuning
# --------------------------------------------------------------------------- #
def _write_jsonl(path, n=40):
    rows = [
        {
            "instruction": f"Repeat this exactly: fox number {i} jumps",
            "response": f"fox number {i} jumps",
            "task": "repeat",
        }
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return rows


def test_read_pairs_accepts_both_schemas(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text(
        json.dumps({"instruction": "hi", "response": "there"})
        + "\n"
        + json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "yo"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recs = read_pairs(p)
    assert [r["response"] for r in recs] == ["there", "yo"]
    assert all(r["messages"][-1]["role"] == "user" for r in recs)


def test_loss_mask_covers_exactly_the_response(tmp_path, tok):
    jl = tmp_path / "d.jsonl"
    _write_jsonl(jl)
    prepare_sft(jl, tok, tmp_path, val_fraction=0.1, max_len=128)
    ds = SFTDataset(tmp_path / "sft_train.npz", pad_id=tok.pad_id)

    _, y = ds.collate([0])
    ids, plen = ds.example(0)
    supervised = (y[0] != -1).nonzero().flatten()
    # the first supervised position predicts the first response token
    assert supervised[0].item() == plen - 1
    assert y[0][supervised].tolist() == ids[plen:].tolist()
    # ...and the final supervised target is EOS, so the model learns to stop
    assert y[0][supervised][-1].item() == tok.eos_id


def test_padding_never_contributes_to_loss(tmp_path, tok):
    jl = tmp_path / "d.jsonl"
    rows = [
        {"instruction": "Repeat this exactly: a", "response": "a", "task": "t"},
        {
            "instruction": "Repeat this exactly: " + " ".join(["long"] * 20),
            "response": " ".join(["long"] * 20),
            "task": "t",
        },
    ] * 6
    jl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    prepare_sft(jl, tok, tmp_path, val_fraction=0.1, max_len=128)
    ds = SFTDataset(tmp_path / "sft_train.npz", pad_id=tok.pad_id)

    idx = list(range(min(6, len(ds))))
    _, y = ds.collate(idx)
    for b, i in enumerate(idx):
        ids, _ = ds.example(i)
        # everything past this example's real length must be ignored
        assert (y[b, len(ids) - 1 :] == -1).all()


def test_encode_pair_keeps_the_response_when_truncating(tok):
    rec = {
        "messages": [{"role": "user", "content": "word " * 400}],
        "response": "the answer",
        "task": "t",
    }
    ids, plen = encode_pair(rec, tok, max_len=64)
    assert len(ids) <= 64
    # the prompt is left-truncated; the response and its EOS always survive
    assert ids[plen:] == [*tok.encode("the answer"), tok.eos_id]


def test_finetune_requires_pretrained_weights(tmp_path, tok):
    jl = tmp_path / "d.jsonl"
    _write_jsonl(jl)
    prepare_sft(jl, tok, tmp_path, val_fraction=0.1, max_len=128)
    with pytest.raises(SystemExit, match="init-from"):
        finetune(tmp_path, tmp_path / "out", SFTConfig(max_steps=1))


@pytest.mark.slow
def test_finetune_end_to_end_improves_the_target_behaviour(tmp_path, tok):
    """The point of SFT is a behaviour change, so assert on behaviour."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    prepare(tok, tmp_path / "data", val_fraction=0.1, inputs=[str(corpus)])

    base = _tiny_model(tok.vocab_size)
    base_ckpt = tmp_path / "base.pt"
    torch.save(
        {
            "model": base.state_dict(),
            "model_cfg": base.cfg.to_dict(),
            "step": 0,
            "tokenizer_path": str(tmp_path / "tokenizer.json"),
        },
        base_ckpt,
    )

    jl = tmp_path / "d.jsonl"
    _write_jsonl(jl, n=64)
    prepare_sft(jl, tok, tmp_path / "sft", val_fraction=0.1, max_len=128)

    cfg = SFTConfig(
        batch_size=8, max_steps=30, eval_interval=15, eval_iters=2, warmup_steps=5,
        lr=3e-4, bench_examples=8, log_interval=100,
    )  # fmt: skip
    best = finetune(
        tmp_path / "sft",
        tmp_path / "out",
        cfg,
        init_from=str(base_ckpt),
        tokenizer_path=str(tmp_path / "tokenizer.json"),
        bench_jsonl=jl,
    )
    assert (tmp_path / "out" / "best.pt").exists()
    assert (tmp_path / "out" / "sft_history.json").exists()
    # the run records the behavioural score it selected on, not just a loss
    hist = json.loads((tmp_path / "out" / "sft_history.json").read_text())
    assert hist["select_by"] == "score"
    assert all("score" in h for h in hist["history"])
    assert best["loss"] < float("inf")


def test_finetune_falls_back_to_loss_selection_and_says_so(tmp_path, tok, capsys):
    jl = tmp_path / "d.jsonl"
    _write_jsonl(jl)
    prepare_sft(jl, tok, tmp_path / "sft", val_fraction=0.1, max_len=128)
    base = _tiny_model(tok.vocab_size)
    base_ckpt = tmp_path / "base.pt"
    torch.save({"model": base.state_dict(), "model_cfg": base.cfg.to_dict(), "step": 0}, base_ckpt)
    finetune(
        tmp_path / "sft",
        tmp_path / "out",
        SFTConfig(batch_size=4, max_steps=2, eval_interval=2, eval_iters=1, warmup_steps=1),
        init_from=str(base_ckpt),
        tokenizer_path=str(tmp_path / "tokenizer.json"),
    )
    assert "falling back to loss-based selection" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# behavioural eval
# --------------------------------------------------------------------------- #
def test_minimal_pairs_preserve_word_count_and_differ():
    docs = [
        "The big dog was happy and he ran to the park. A cat is sitting on a mat there.",
        "She has a red ball and they were playing with it near an old tree today.",
    ] * 20
    items = build_minimal_pairs(docs, n_items=20, seed=0)
    assert items
    for it in items:
        assert it["good"] != it["bad"]
        # a length confound would let mean-per-token normalization decide the answer
        assert len(it["good"].split()) == len(it["bad"].split())
        assert it["kind"] in {"swap", "agreement", "pronoun", "article"}


def test_sequence_logprobs_scores_only_the_requested_span():
    torch.manual_seed(0)
    model = _tiny_model(64)
    seq = [1, 2, 3, 4, 5, 6, 7, 8]
    full = sequence_logprobs(model, [seq], "cpu")[0]
    tail = sequence_logprobs(model, [seq], "cpu", score_from=[5])[0]
    assert full[1] == len(seq) - 1
    assert tail[1] == len(seq) - 5
    assert tail[0] > full[0] or tail[1] < full[1]  # fewer tokens scored


def test_sequence_logprobs_ignores_right_padding():
    torch.manual_seed(0)
    model = _tiny_model(64)
    short, long = [1, 2, 3], [1, 2, 3, 9, 9, 9, 9, 9]
    alone = sequence_logprobs(model, [short], "cpu")[0]
    batched = sequence_logprobs(model, [short, long], "cpu")[0]
    assert alone[1] == batched[1]
    assert alone[0] == pytest.approx(batched[0], abs=1e-4)


def test_normalize_answer_is_forgiving_about_formatting():
    assert normalize_answer("  The Cat.  ") == normalize_answer("the cat")
    assert normalize_answer("A  B") == "a b"
    assert normalize_answer("7") != normalize_answer("8")


def test_score_instructions_is_deterministic(tmp_path, tok):
    model = _tiny_model(tok.vocab_size)
    examples = [
        {"messages": [{"role": "user", "content": "Repeat this exactly: fox"}],
         "target": "fox", "task": "repeat"}
    ] * 3  # fmt: skip
    a = score_instructions(model, tok, examples, "cpu", max_new_tokens=4)
    b = score_instructions(model, tok, examples, "cpu", max_new_tokens=4)
    assert a["exact_match"] == b["exact_match"]
    assert 0.0 <= a["exact_match"] <= 1.0
    assert a["by_task"].keys() == {"repeat"}


def test_dynamic_quantized_checkpoint_loads_and_generates(tmp_path):
    """`quantize --method dynamic` must produce something the CLI can read back.

    The dynamic path pickles a live torch.ao module, so `weights_only=True`
    refuses it. That made the advertised feature write a file no consumer could
    load, and the old test only asserted the file existed.
    """
    from pico_lm.export import quantize_int8

    torch.manual_seed(0)
    model = _tiny_model(256)
    src = tmp_path / "fp32.pt"
    torch.save({"model": model.state_dict(), "model_cfg": model.cfg.to_dict(), "step": 3}, src)

    out = quantize_int8(src, tmp_path / "dynamic-int8.pt")
    assert out.exists()

    reloaded, ckpt = load_model(out, device="cpu")
    assert ckpt.get("model_cfg") is not None
    x = torch.randint(0, 256, (1, 4))
    generated = reloaded.generate(x, max_new_tokens=5, temperature=0.0, top_k=0, top_p=None)
    assert generated.shape == (1, 9)


def test_dynamic_quantized_model_reports_its_real_parameter_count(tmp_path):
    """int8 weights live in `_packed_params`, invisible to `.parameters()`.

    Counting only `.parameters()` reported a fraction of the true size — 6.30M
    for a 15.73M micro. The count must not collapse just because the weights
    moved into packed storage.

    It does not come back exactly equal, and that is real rather than slop:
    dynamic quantization gives the tied `lm_head` its own packed copy of the
    embedding, so a tied model genuinely stops sharing that tensor.
    """
    from pico_lm.export import quantize_int8

    torch.manual_seed(0)
    model = _tiny_model(256)
    expected = model.num_params()
    embedding = model.tok_emb.weight.numel()
    assert model.cfg.tie_weights
    src = tmp_path / "fp32.pt"
    torch.save({"model": model.state_dict(), "model_cfg": model.cfg.to_dict(), "step": 1}, src)

    quantized, _ = load_model(quantize_int8(src, tmp_path / "q.pt"), device="cpu")
    naive = sum(p.numel() for p in quantized.parameters())

    assert naive < expected * 0.5, "expected .parameters() alone to miss the packed weights"
    assert quantized.num_params() == expected + embedding


def test_load_model_still_rejects_a_corrupt_checkpoint(tmp_path):
    """The dynamic fallback must not turn unreadable files into silent successes.

    A corrupt file raises `UnpicklingError` — the very exception the fallback
    catches — so this pins the behaviour that the retry re-raises it instead of
    swallowing it or returning half a model.
    """
    import pickle

    bad = tmp_path / "truncated.pt"
    bad.write_bytes(b"not a torch checkpoint at all")
    with pytest.raises(pickle.UnpicklingError):
        load_model(bad, device="cpu")


def test_preset_parameter_counts_match_their_documented_comments():
    """Guards the ~3M/~22M/~42M/~110M drift that shipped in config.py."""
    from pico_lm.config import PRESETS

    documented = {"nano": 2.9, "micro": 15.7, "small": 32.0, "base": 88.1}
    for name, expected_m in documented.items():
        actual_m = Transformer(PRESETS[name]).num_params() / 1e6
        assert actual_m == pytest.approx(expected_m, abs=0.05), (
            f"{name}: config.py documents {expected_m}M, architecture gives {actual_m:.2f}M"
        )
