import json

import numpy as np
import pytest
import torch

from pico_lm.data import TokenDataset, iter_texts, prepare
from pico_lm.tokenizer import PicoTokenizer

TEXT = "\n\n".join(f"Document number {i} talks about foxes and dogs." for i in range(200))


@pytest.fixture(scope="module")
def corpus_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("corpus") / "corpus.txt"
    p.write_text(TEXT, encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def tok(tmp_path_factory, corpus_file):
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    return PicoTokenizer.train(
        iter_texts(inputs=[str(corpus_file)]), vocab_size=512, save_path=path
    )


def test_iter_texts_splits_documents(corpus_file):
    docs = list(iter_texts(inputs=[str(corpus_file)]))
    assert len(docs) == 200
    assert docs[0].startswith("Document number 0")


def test_iter_texts_max_docs(corpus_file):
    assert len(list(iter_texts(inputs=[str(corpus_file)], max_docs=7))) == 7


def test_prepare_and_batches(tok, corpus_file, tmp_path):
    meta = prepare(tok, tmp_path, val_fraction=0.1, inputs=[str(corpus_file)])
    assert meta["n_docs"] == 200
    on_disk = np.memmap(tmp_path / "train.bin", dtype=np.uint16, mode="r")
    assert len(on_disk) == meta["train_tokens"]
    assert json.loads((tmp_path / "meta.json").read_text())["vocab_size"] == tok.vocab_size
    # prepare keeps a copy of the tokenizer with the bins
    assert (tmp_path / "tokenizer.json").exists()
    assert meta["tokenizer"] == "tokenizer.json"

    ds = TokenDataset(tmp_path / "train.bin")
    x, y = ds.get_batch(batch_size=4, seq_len=32, device="cpu")
    assert x.shape == y.shape == (4, 32)
    assert x.dtype == torch.int64
    # y is x shifted by one
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_chat_jsonl_source(tmp_path):
    p = tmp_path / "chat.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "hello"}]},
        {"messages": [{"role": "user", "content": "2+2?"},
                      {"role": "assistant", "content": "4"}]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    docs = list(iter_texts(chat_jsonl=str(p)))
    assert len(docs) == 2
    assert "<|user|>hi<|eos|>" in docs[0]
    assert "<|assistant|>hello<|eos|>" in docs[0]


def test_val_split_is_never_empty_on_small_corpora(tok, tmp_path):
    """A random coin flip used to leave val.bin empty, crashing training later."""
    small = tmp_path / "small.txt"
    body = "\n\n".join(f"Doc {i} about foxes." for i in range(30))
    small.write_text(body, encoding="utf-8")
    meta = prepare(tok, tmp_path / "out", val_fraction=0.01, inputs=[str(small)])
    assert meta["val_tokens"] > 0
    assert len(TokenDataset(tmp_path / "out" / "val.bin")) == meta["val_tokens"]


def test_prepare_rejects_a_corpus_too_small_to_split(tok, tmp_path):
    tiny = tmp_path / "tiny.txt"
    tiny.write_text("only one document here", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus too small"):
        prepare(tok, tmp_path / "out2", val_fraction=0.01, inputs=[str(tiny)])


def test_prepare_rejects_an_empty_corpus(tok, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no documents"):
        prepare(tok, tmp_path / "out3", inputs=[str(empty)])


def test_token_dataset_reports_a_useful_error(tmp_path):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="fewer than 2 tokens"):
        TokenDataset(empty)
    with pytest.raises(FileNotFoundError):
        TokenDataset(tmp_path / "nope.bin")
