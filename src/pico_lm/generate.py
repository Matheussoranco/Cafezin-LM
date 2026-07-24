"""Inference helpers: completion and an interactive chat REPL."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import Transformer
from .tokenizer import PicoTokenizer
from .train import load_model, pick_device


def load_for_inference(
    ckpt_path: str | Path, tokenizer_path: str | Path | None = None, device: str | None = None
) -> tuple[Transformer, PicoTokenizer, str]:
    device = device or pick_device()
    model, ckpt = load_model(ckpt_path, device)
    model.eval()
    candidates = [
        tokenizer_path,
        ckpt.get("tokenizer_path"),
        # training stages a copy beside the checkpoints; fall back to it when the
        # recorded absolute path has moved (renamed folder, copied run dir, ...)
        Path(ckpt_path).parent / "tokenizer.json",
    ]
    tok_path = next((c for c in candidates if c and Path(c).exists()), None)
    if tok_path is None:
        raise FileNotFoundError(
            f"tokenizer not found for {ckpt_path}; pass --tokenizer, or keep "
            "tokenizer.json next to the checkpoint"
        )
    tokenizer = PicoTokenizer.load(tok_path)
    return model, tokenizer, device


def complete(
    model: Transformer,
    tokenizer: PicoTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.1,
) -> str:
    ids = tokenizer.encode(prompt, bos=True)
    # crop here (mirroring generate's internal window) so the output slice stays aligned
    keep = max(1, model.cfg.max_seq_len - max_new_tokens)
    if len(ids) > keep:
        ids = ids[-keep:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eos_id,
        repetition_penalty=repetition_penalty,
    )
    return tokenizer.decode(y[0, len(ids) :].tolist())


def chat_repl(
    model: Transformer,
    tokenizer: PicoTokenizer,
    device: str,
    system: str | None = None,
    **gen_kwargs,
) -> None:
    print("chat mode — type 'exit' to quit. (Meaningful replies require a chat-trained model.)")
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": user})
        prompt = tokenizer.format_chat(messages)
        reply = complete(model, tokenizer, prompt, device, **gen_kwargs)
        reply = reply.strip()
        print(f"pico> {reply}")
        messages.append({"role": "assistant", "content": reply})
