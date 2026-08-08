"""Generate a small, *verifiable* instruction-following dataset.

Every task here has a mechanically checkable answer, which is the whole point:
the SFT run is judged by exact match on held-out prompts, not by loss. A task
whose answer is a matter of taste cannot tell you that a fine-tune broke the
model; "what is the last word of this sentence" can.

The vocabulary is deliberately TinyStories-shaped (simple names, concrete
nouns, short sentences) so that a model pretrained on TinyStories has actually
seen the tokens involved, and a failure means "did not learn the task" rather
than "cannot spell the words".

Splits share the word pool but never share an instance: the exact sentence in a
test prompt is guaranteed not to appear in training. That tests generalization
of the *instruction*, which is what SFT is supposed to buy.

There are three splits, not two, and the reason matters. ``dev`` is scored
during fine-tuning to choose which checkpoint to keep; ``test`` is scored once
at the end. Selecting on the same examples you report would quietly turn the
headline number into a training metric — the exact class of mistake this whole
evaluation setup exists to catch.

    python examples/make_instruct_data.py --out data_sft --train 6000 \
        --dev 200 --test 400
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

NAMES = [
    "Tom", "Lily", "Ben", "Anna", "Sam", "Mia", "Max", "Sara",
    "Tim", "Emma", "Jack", "Rosa", "Leo", "Nina", "Finn", "Zoe",
]  # fmt: skip
NOUNS = [
    "dog", "cat", "ball", "tree", "box", "bird", "hat", "car",
    "book", "star", "cake", "frog", "duck", "boat", "drum", "kite",
    "apple", "chair", "flower", "rock",
]  # fmt: skip
ADJS = [
    "big", "small", "red", "blue", "happy", "sad", "funny",
    "soft", "warm", "shiny", "tiny", "brave", "old", "new",
]  # fmt: skip
VERBS = [
    "found", "saw", "liked", "held", "dropped", "chased",
    "kicked", "hugged", "painted", "washed", "wanted", "carried",
]  # fmt: skip

TEMPLATES = [
    "{name} {verb} the {adj} {noun}",
    "the {adj} {noun} was near the {noun2}",
    "{name} and {name2} played with the {noun}",
    "a {adj} {noun} sat on the {noun2}",
    "{name} {verb} a {noun} and a {noun2}",
    "the {noun} was very {adj} today",
]

# Two phrasings per task: enough that the model must key on meaning rather than
# memorize one literal prefix, few enough that a 16M model can actually learn it.
PHRASINGS = {
    "last_word": [
        "What is the last word of this sentence: {s}",
        "Tell me the final word in: {s}",
    ],
    "first_word": [
        "What is the first word of this sentence: {s}",
        "Tell me the first word in: {s}",
    ],
    "word_count": [
        "How many words are in this sentence: {s}",
        "Count the words in: {s}",
    ],
    "uppercase": [
        "Write this in capital letters: {s}",
        "Say this in all caps: {s}",
    ],
    "repeat": [
        "Repeat this exactly: {s}",
        "Say this back to me: {s}",
    ],
    "reverse": [
        "Say these words in reverse order: {s}",
        "Reverse the word order of: {s}",
    ],
}

ANSWERS = {
    "last_word": lambda s: s.split()[-1],
    "first_word": lambda s: s.split()[0],
    "word_count": lambda s: str(len(s.split())),
    "uppercase": lambda s: s.upper(),
    "repeat": lambda s: s,
    "reverse": lambda s: " ".join(reversed(s.split())),
}


def make_sentence(rng: random.Random) -> str:
    t = rng.choice(TEMPLATES)
    return t.format(
        name=rng.choice(NAMES),
        name2=rng.choice(NAMES),
        verb=rng.choice(VERBS),
        adj=rng.choice(ADJS),
        noun=rng.choice(NOUNS),
        noun2=rng.choice(NOUNS),
    )


def build(n_total: int, seed: int) -> list[dict]:
    """Unique (task, sentence) pairs, round-robin across tasks."""
    rng = random.Random(seed)
    tasks = list(PHRASINGS)
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    guard = 0
    while len(out) < n_total and guard < n_total * 200:
        guard += 1
        task = tasks[len(out) % len(tasks)]
        sent = make_sentence(rng)
        # reverse/uppercase get short inputs so the answer stays inside the budget
        if task in ("reverse", "uppercase"):
            words = sent.split()
            sent = " ".join(words[: rng.randint(3, min(5, len(words)))])
        if (task, sent) in seen:
            continue
        seen.add((task, sent))
        out.append(
            {
                "task": task,
                "instruction": rng.choice(PHRASINGS[task]).format(s=sent),
                "response": ANSWERS[task](sent),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data_sft")
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--dev", type=int, default=200, help="scored during training to pick best.pt")
    ap.add_argument("--test", type=int, default=400, help="scored once, at the end")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    total = args.train + args.dev + args.test
    records = build(total, args.seed)
    if len(records) < total:
        raise SystemExit(f"could only build {len(records)} unique examples")
    rng = random.Random(args.seed + 1)
    rng.shuffle(records)
    dev = records[: args.dev]
    test = records[args.dev : args.dev + args.test]
    train = records[args.dev + args.test :]

    # instance-level disjointness is the property every reported number depends on
    held_out = {(r["task"], r["instruction"]) for r in dev + test}
    assert not any((r["task"], r["instruction"]) in held_out for r in train)
    dev_keys = {(r["task"], r["instruction"]) for r in dev}
    assert not any((r["task"], r["instruction"]) in dev_keys for r in test)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        path = out / f"instruct_{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
        )
        by_task: dict[str, int] = {}
        for r in rows:
            by_task[r["task"]] = by_task.get(r["task"], 0) + 1
        print(f"{path}: {len(rows)} examples  {by_task}")


if __name__ == "__main__":
    main()
