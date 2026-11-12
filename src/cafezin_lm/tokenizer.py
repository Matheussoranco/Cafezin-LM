"""Vocabulary for the model: a Hugging Face ``tokenizer.json`` wrapper.

The vocabulary has to match the checkpoint's embedding, so training uses the
reference BPE rather than a re-trained one. A mismatched vocabulary is the
classic silent failure here — the ids stay in range and the model trains to
garbage — so the id space is checked when a corpus is prepared and again when
training starts.

`save` writes the underlying ``tokenizer.json`` verbatim, which is what makes an
exported checkpoint self-contained.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import REFERENCE_REPO, REFERENCE_REVISION

# Tokens that may carry the end-of-text id, tried in order.
_EOS_CANDIDATES = ("<|endoftext|>", "<|im_end|>", "<eos>", "</s>")


class Tokenizer:
    """Thin wrapper over a Hugging Face ``tokenizer.json``.

    Exposes just what corpus preparation, generation and export need. Special
    ids are read from the tokenizer where it declares them and otherwise fall
    back to the caller-supplied value.
    """

    def __init__(self, path: str | Path, eos_token_id: int | None = None) -> None:
        try:
            from tokenizers import Tokenizer as _HFTokenizer
        except ImportError as e:  # pragma: no cover - tokenizers is a hard dep
            raise ImportError("tokenizers is required to use a tokenizer.json") from e

        self.path = Path(path)
        self._tokenizer = _HFTokenizer.from_file(str(self.path))
        vocab = self._tokenizer.get_vocab(with_added_tokens=True)
        self._vocab_size = len(vocab)
        self.max_token_id = max(vocab.values())

        declared = None
        for token in _EOS_CANDIDATES:
            declared = self._tokenizer.token_to_id(token)
            if declared is not None:
                break
        self.eos_token_id = eos_token_id if eos_token_id is not None else declared

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, eos: bool = False) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if eos and self.eos_token_id is not None:
            ids = [*ids, self.eos_token_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=True)

    def save(self, path: str | Path) -> Path:
        """Copy the tokenizer file next to a corpus or checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != self.path.resolve():
            shutil.copyfile(self.path, path)
        return path

    @classmethod
    def from_hub(
        cls,
        repo: str = REFERENCE_REPO,
        revision: str = REFERENCE_REVISION,
        dest: str | Path | None = None,
        **kwargs,
    ) -> Tokenizer:
        """Download the reference tokenizer (a few MB) and wrap it."""
        from .checkpoint import download_file

        if dest is None:
            dest = Path.home() / ".cache" / "cafezin-lm" / repo / revision / "tokenizer.json"
        path = download_file("tokenizer.json", dest, repo=repo, revision=revision)
        return cls(path, **kwargs)
