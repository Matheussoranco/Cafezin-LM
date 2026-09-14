"""Behavioural evaluation — task-level scores, not training loss.

Why this module exists: a fine-tune can drive loss to near zero and still be a
*worse* model than the checkpoint it started from. Loss is measured on the
distribution you trained on; capability is not. Everything here scores what the
model actually *does* on held-out data, with an explicit chance baseline so a
broken model is obvious rather than merely "slightly higher loss".

Four probes:

``minimal_pairs``
    BLiMP-style grammaticality preference. Each item is a real held-out sentence
    and a minimally corrupted copy (adjacent-word swap, subject-verb agreement,
    pronoun case, a/an). The model must give the real one a higher mean token
    log-probability. **Chance = 50%.**

``cloze``
    N-way continuation choice: a real context plus its true continuation against
    N-1 continuations lifted from elsewhere in the held-out split.
    **Chance = 100/N %.**

``genhealth``
    Free-generation degeneracy statistics — 4-gram repetition, distinct-2, and
    how often generation terminates. Catches the models that score fine on
    likelihood probes but loop forever when asked to actually produce text.

``instruct``
    Exact-match instruction following on held-out prompts, greedy decoded. This
    is the probe that catches an SFT run whose loss looked excellent.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import Transformer
from .tokenizer import ASSISTANT, PicoTokenizer

# --------------------------------------------------------------------------- #
# corpus helpers
# --------------------------------------------------------------------------- #


def val_documents(
    bin_path: str | Path, tokenizer: PicoTokenizer, max_docs: int = 2000, skip: int = 0
) -> list[str]:
    """Recover documents from a token memmap by splitting on the EOS token."""
    data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
    ends = np.flatnonzero(np.asarray(data) == tokenizer.eos_id)
    docs, start = [], 0
    for i, end in enumerate(ends):
        if i >= skip + max_docs:
            break
        if i >= skip:
            docs.append(tokenizer.decode(data[start:end].tolist()))
        start = int(end) + 1
    return [d.strip() for d in docs if d.strip()]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- #
# probe 1: minimal pairs (grammaticality preference, chance 50%)
# --------------------------------------------------------------------------- #
AGREEMENT = {
    "was": "were", "were": "was", "is": "are", "are": "is",
    "has": "have", "have": "has", "does": "do", "do": "does",
}  # fmt: skip
PRONOUN_CASE = {
    "he": "him", "him": "he", "she": "her", "her": "she",
    "they": "them", "them": "they", "we": "us", "us": "we",
}  # fmt: skip
ARTICLES = {"a": "an", "an": "a"}


def _sub_word(words: list[str], table: dict[str, str]) -> list[str] | None:
    """Replace one randomly chosen word that has an entry in ``table``."""
    hits = [i for i, w in enumerate(words) if w.lower() in table and i > 0]
    if not hits:
        return None
    i = random.choice(hits)
    out = list(words)
    out[i] = table[words[i].lower()]
    return out


def _swap_adjacent(words: list[str]) -> list[str] | None:
    """Swap two adjacent lowercase words (skipping the first, to keep casing)."""
    hits = [
        i
        for i in range(1, len(words) - 2)
        if words[i].isalpha() and words[i + 1].isalpha() and words[i] != words[i + 1]
    ]
    if not hits:
        return None
    i = random.choice(hits)
    out = list(words)
    out[i], out[i + 1] = out[i + 1], out[i]
    return out


CORRUPTIONS = {
    "swap": _swap_adjacent,
    "agreement": lambda w: _sub_word(w, AGREEMENT),
    "pronoun": lambda w: _sub_word(w, PRONOUN_CASE),
    "article": lambda w: _sub_word(w, ARTICLES),
}


def build_minimal_pairs(docs: list[str], n_items: int, seed: int = 0) -> list[dict]:
    """Real sentence vs. minimally corrupted copy. Word count is preserved, so
    mean-per-token normalization is not fighting a length confound."""
    random.seed(seed)
    sentences = [s for d in docs for s in split_sentences(d) if 6 <= len(s.split()) <= 30]
    random.shuffle(sentences)
    items, kinds = [], list(CORRUPTIONS)
    for idx, sent in enumerate(sentences):
        if len(items) >= n_items:
            break
        kind = kinds[idx % len(kinds)]
        words = sent.split()
        bad_words = CORRUPTIONS[kind](words)
        if bad_words is None:
            continue
        bad = " ".join(bad_words)
        if bad == sent:
            continue
        items.append({"kind": kind, "good": sent, "bad": bad})
    return items


# --------------------------------------------------------------------------- #
# scoring primitives
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sequence_logprobs(
    model: Transformer,
    sequences: list[list[int]],
    device: str,
    pad_id: int = 0,
    batch_size: int = 16,
    score_from: list[int] | None = None,
) -> list[tuple[float, int]]:
    """(sum log P, tokens scored) for each sequence.

    ``score_from[i]`` is the first *position* whose prediction counts; use it to
    score only a continuation given a context. Defaults to 1 (score everything
    after the first token).
    """
    model.eval()
    limit = model.cfg.max_seq_len
    seqs = [s[:limit] for s in sequences]
    starts = list(score_from) if score_from is not None else [1] * len(seqs)
    out: list[tuple[float, int]] = []
    for b0 in range(0, len(seqs), batch_size):
        chunk = seqs[b0 : b0 + batch_size]
        chunk_starts = starts[b0 : b0 + batch_size]
        width = max(len(s) for s in chunk)
        x = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        for i, s in enumerate(chunk):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        x = x.to(device)
        logits, _ = model(x)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        # position t predicts token t+1
        tgt = x[:, 1:]
        gathered = logprobs[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for i, s in enumerate(chunk):
            lo = max(1, chunk_starts[i]) - 1  # index into the shifted array
            hi = len(s) - 1
            if hi <= lo:
                out.append((0.0, 0))
                continue
            total = gathered[i, lo:hi].sum().item()
            out.append((total, hi - lo))
    return out


def score_minimal_pairs(
    model: Transformer, tok: PicoTokenizer, items: list[dict], device: str
) -> dict:
    seqs, meta = [], []
    for it in items:
        seqs.append(tok.encode(it["good"], bos=True))
        seqs.append(tok.encode(it["bad"], bos=True))
        meta.append(it["kind"])
    scored = sequence_logprobs(model, seqs, device, pad_id=tok.pad_id)
    per_kind: dict[str, list[int]] = {}
    correct = 0
    for i, kind in enumerate(meta):
        g_sum, g_n = scored[2 * i]
        b_sum, b_n = scored[2 * i + 1]
        g = g_sum / max(1, g_n)
        b = b_sum / max(1, b_n)
        hit = int(g > b)
        correct += hit
        per_kind.setdefault(kind, []).append(hit)
    n = max(1, len(meta))
    return {
        "accuracy": correct / n,
        "n": len(meta),
        "chance": 0.5,
        "by_kind": {k: sum(v) / len(v) for k, v in sorted(per_kind.items())},
        "items": items,
    }


def build_cloze(
    docs: list[str],
    tok: PicoTokenizer,
    n_items: int,
    ctx_tokens: int = 48,
    cont_tokens: int = 24,
    n_choices: int = 4,
    seed: int = 0,
) -> list[dict]:
    if n_choices < 2 or ctx_tokens < 1 or cont_tokens < 1 or n_items < 0:
        raise ValueError("invalid cloze dimensions")
    rng = random.Random(seed)
    encoded = [tok.encode(d) for d in docs]
    usable = [e for e in encoded if len(e) >= ctx_tokens + cont_tokens]
    rng.shuffle(usable)
    pool = [e[ctx_tokens : ctx_tokens + cont_tokens] for e in usable]
    items = []
    for i, e in enumerate(usable[:n_items]):
        ctx = e[:ctx_tokens]
        true = e[ctx_tokens : ctx_tokens + cont_tokens]
        alternatives = list(dict.fromkeys(tuple(c) for c in pool if c != true))
        if len(alternatives) < n_choices - 1:
            raise ValueError("not enough distinct continuations for cloze choices")
        choices = [true, *[list(c) for c in rng.sample(alternatives, n_choices - 1)]]
        order = list(range(n_choices))
        rng.shuffle(order)
        items.append(
            {
                "ctx": ctx,
                "choices": [choices[j] for j in order],
                "answer": order.index(0),
                "id": i,
            }
        )
    return items


def score_cloze(model: Transformer, tok: PicoTokenizer, items: list[dict], device: str) -> dict:
    seqs, starts = [], []
    for it in items:
        for ch in it["choices"]:
            seqs.append(it["ctx"] + ch)
            starts.append(len(it["ctx"]))
    scored = sequence_logprobs(model, seqs, device, pad_id=tok.pad_id, score_from=starts)
    n_choices = len(items[0]["choices"]) if items else 4
    correct = 0
    for i, it in enumerate(items):
        block = scored[i * n_choices : (i + 1) * n_choices]
        means = [s / max(1, n) for s, n in block]
        correct += int(int(np.argmax(means)) == it["answer"])
    return {
        "accuracy": correct / max(1, len(items)),
        "n": len(items),
        "chance": 1.0 / n_choices,
        "n_choices": n_choices,
        "items": items,
    }


# --------------------------------------------------------------------------- #
# probe 3: generation health
# --------------------------------------------------------------------------- #
def _ngram_stats(ids: list[int], n: int) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


@torch.no_grad()
def generation_health(
    model: Transformer,
    tok: PicoTokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int = 96,
    temperature: float = 0.8,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    model.eval()
    distinct2, unique4, lengths, finished, samples = [], [], [], 0, []
    for p in prompts:
        ids = tok.encode(p, bos=True)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            eos_id=tok.eos_id,
            repetition_penalty=1.0,  # no penalty: we are *measuring* repetition
        )
        gen = y[0, len(ids) :].tolist()
        if tok.eos_id in gen:
            finished += 1
            gen = gen[: gen.index(tok.eos_id)]
        if not gen:
            continue
        distinct2.append(_ngram_stats(gen, 2))
        if len(gen) >= 4:
            unique4.append(_ngram_stats(gen, 4))
        lengths.append(len(gen))
        if len(samples) < 4:
            samples.append({"prompt": p, "completion": tok.decode(gen)})
    return {
        "distinct_2": float(np.mean(distinct2)) if distinct2 else 0.0,
        "unique_4gram": float(np.mean(unique4)) if unique4 else 0.0,
        "repetition_4gram": 1.0 - float(np.mean(unique4)) if unique4 else 0.0,
        "repetition_eligible_samples": len(unique4),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "eos_rate": finished / max(1, len(prompts)),
        "samples": samples,
    }


# --------------------------------------------------------------------------- #
# probe 4: instruction following (exact match)
# --------------------------------------------------------------------------- #
def load_instruction_examples(path: str | Path) -> list[dict]:
    """Read JSONL of {"messages": [...]} or {"instruction", "response"} records."""
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "messages" in rec:
                msgs = rec["messages"]
                last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
                prompt_msgs = msgs[:last]
                target = msgs[last]["content"]
            else:
                prompt_msgs = [{"role": "user", "content": rec["instruction"]}]
                target = rec["response"]
                if rec.get("system"):
                    prompt_msgs.insert(0, {"role": "system", "content": rec["system"]})
            out.append({"messages": prompt_msgs, "target": target, "task": rec.get("task", "all")})
    return out


def normalize_answer(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip(".!?,"))


@torch.no_grad()
def score_instructions(
    model: Transformer,
    tok: PicoTokenizer,
    examples: list[dict],
    device: str,
    max_new_tokens: int = 32,
    n_samples: int = 4,
) -> dict:
    """Greedy-decode each prompt and exact-match against the reference."""
    model.eval()
    per_task: dict[str, list[int]] = {}
    samples = []
    for ex in examples:
        prompt = tok.format_chat(ex["messages"], add_generation_prompt=True)
        ids = tok.encode(prompt, bos=True)
        limit = model.cfg.max_seq_len - max_new_tokens
        if len(ids) > limit:
            ids = ids[-limit:]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # greedy: the eval must be deterministic
            top_k=0,
            top_p=None,
            eos_id=tok.eos_id,
            repetition_penalty=1.0,
        )
        gen = y[0, len(ids) :].tolist()
        if tok.eos_id in gen:
            gen = gen[: gen.index(tok.eos_id)]
        pred = tok.decode(gen)
        # a model that never learned the template will run on into the next turn
        pred = pred.split(ASSISTANT)[0]
        hit = int(normalize_answer(pred) == normalize_answer(ex["target"]))
        per_task.setdefault(ex["task"], []).append(hit)
        if len(samples) < n_samples:
            samples.append(
                {
                    "instruction": ex["messages"][-1]["content"],
                    "expected": ex["target"],
                    "got": pred.strip()[:160],
                    "correct": bool(hit),
                }
            )
    flat = [h for v in per_task.values() for h in v]
    return {
        "exact_match": sum(flat) / max(1, len(flat)),
        "n": len(flat),
        "by_task": {k: sum(v) / len(v) for k, v in sorted(per_task.items())},
        "samples": samples,
    }


# --------------------------------------------------------------------------- #
# throughput
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_speed(
    model: Transformer,
    device: str,
    prompt_tokens: int = 32,
    new_tokens: int = 128,
    repeats: int = 3,
) -> dict:
    if prompt_tokens < 1 or new_tokens < 1 or repeats < 1 or model.cfg.max_seq_len < 2:
        raise ValueError("speed benchmark requires positive dimensions and context >= 2")
    new_tokens = min(new_tokens, model.cfg.max_seq_len - 1)
    prompt_tokens = min(prompt_tokens, model.cfg.max_seq_len - new_tokens)
    model.eval()
    vocab = model.cfg.vocab_size
    x = torch.randint(0, vocab, (1, prompt_tokens), device=device)
    # warm up (CUDA kernel autotune / lazy init would otherwise land in run 1)
    model.generate(x, max_new_tokens=8, temperature=0.0, top_k=0, top_p=None)
    times = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(x, max_new_tokens=new_tokens, temperature=0.0, top_k=0, top_p=None)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    best = float(np.median(times))
    return {
        "decode_tok_per_s": new_tokens / best,
        "seconds": best,
        "new_tokens": new_tokens,
        "device": device,
        "timings": times,
        "aggregation": "median; elapsed time includes prompt prefill",
    }


# --------------------------------------------------------------------------- #
# validation loss (for reference only — never as the headline number)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def val_loss(model: Transformer, bin_path: str | Path, device: str, iters: int = 40) -> dict:
    from .data import TokenDataset

    ds = TokenDataset(bin_path)
    seq = min(256, model.cfg.max_seq_len)
    torch.manual_seed(1234)
    losses = []
    model.eval()
    for _ in range(iters):
        x, y = ds.get_batch(8, seq, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    mean = float(np.mean(losses))
    return {"loss": mean, "perplexity": math.exp(min(mean, 20))}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_bench(
    model: Transformer,
    tok: PicoTokenizer,
    device: str,
    val_bin: str | Path | None = None,
    instructions: str | Path | None = None,
    n_pairs: int = 400,
    n_cloze: int = 200,
    n_gen: int = 24,
    n_instruct: int = 200,
    seed: int = 0,
    speed: bool = True,
) -> dict:
    """Run every applicable probe and return a JSON-serializable report."""
    report: dict = {}
    if val_bin and Path(val_bin).exists():
        docs = val_documents(val_bin, tok, max_docs=1500)
        report["val_loss"] = val_loss(model, val_bin, device)
        report["minimal_pairs"] = score_minimal_pairs(
            model, tok, build_minimal_pairs(docs, n_pairs, seed), device
        )
        report["cloze"] = score_cloze(
            model, tok, build_cloze(docs, tok, n_cloze, seed=seed), device
        )
        prompts = [" ".join(d.split()[:6]) for d in docs[:n_gen]]
        report["genhealth"] = generation_health(model, tok, prompts, device, seed=seed)
    if instructions and Path(instructions).exists():
        examples = load_instruction_examples(instructions)[:n_instruct]
        report["instruct"] = score_instructions(model, tok, examples, device)
    if speed:
        report["speed"] = measure_speed(model, device)
    return report


def format_report(report: dict, title: str = "") -> str:
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    if "val_loss" in report:
        v = report["val_loss"]
        lines.append(f"val loss           {v['loss']:.4f}   (ppl {v['perplexity']:.2f})")
    if "minimal_pairs" in report:
        m = report["minimal_pairs"]
        by = "  ".join(f"{k}={v:.0%}" for k, v in m["by_kind"].items())
        lines.append(
            f"minimal pairs      {m['accuracy']:.1%}  (chance {m['chance']:.0%}, n={m['n']})"
        )
        lines.append(f"  by kind          {by}")
    if "cloze" in report:
        c = report["cloze"]
        lines.append(
            f"cloze {c['n_choices']}-way        {c['accuracy']:.1%}  "
            f"(chance {c['chance']:.0%}, n={c['n']})"
        )
    if "genhealth" in report:
        g = report["genhealth"]
        lines.append(
            f"generation         distinct-2 {g['distinct_2']:.3f}  "
            f"4-gram rep {g['repetition_4gram']:.3f}  "
            f"eos {g['eos_rate']:.0%}  len {g['mean_length']:.0f}"
        )
    if "instruct" in report:
        i = report["instruct"]
        lines.append(f"instruction EM     {i['exact_match']:.1%}  (n={i['n']})")
        for k, v in i["by_task"].items():
            lines.append(f"  {k:16s} {v:.1%}")
    if "speed" in report:
        s = report["speed"]
        lines.append(f"decode speed       {s['decode_tok_per_s']:.1f} tok/s on {s['device']}")
    return "\n".join(lines)
