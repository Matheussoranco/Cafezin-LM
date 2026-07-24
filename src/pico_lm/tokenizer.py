"""Byte-level BPE tokenizer trained on your own corpus (HuggingFace `tokenizers`)."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD, BOS, EOS = "<|pad|>", "<|bos|>", "<|eos|>"
SYSTEM, USER, ASSISTANT = "<|system|>", "<|user|>", "<|assistant|>"
SPECIAL_TOKENS = [PAD, BOS, EOS, SYSTEM, USER, ASSISTANT]


class PicoTokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tk = tokenizer
        self.pad_id = tokenizer.token_to_id(PAD)
        self.bos_id = tokenizer.token_to_id(BOS)
        self.eos_id = tokenizer.token_to_id(EOS)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, save_path: str | Path) -> PicoTokenizer:
        tk = Tokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tk.train_from_iterator(texts, trainer=trainer)
        tok = cls(tk)
        tok.save(save_path)
        return tok

    @classmethod
    def load(cls, path: str | Path) -> PicoTokenizer:
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tk.save(str(path))
        return path

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = self.tk.encode(text).ids
        if bos:
            ids = [self.bos_id, *ids]
        if eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    def format_chat(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Render [{role, content}, ...] into the plain-text chat template."""
        tags = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
        out = []
        for m in messages:
            out.append(f"{tags[m['role']]}{m['content']}{EOS}")
        if add_generation_prompt:
            out.append(ASSISTANT)
        return "".join(out)
