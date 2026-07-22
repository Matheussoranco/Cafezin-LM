import json

import numpy as np
import pytest
import torch

from slm.data import TokenDataset, iter_texts, prepare
from slm.tokenizer import SLMTokenizer

TEXT = "\n\n".join(f"Document number {i} talks about foxes and dogs." for i in range(200))


@pytest.fixture(scope="module")
def corpus_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("corpus") / "corpus.txt"
    p.write_text(TEXT, encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def tok(tmp_path_factory, corpus_file):
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    return SLMTokenizer.train(iter_texts(inputs=[str(corpus_file)]), vocab_size=512, save_path=path)


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

    ds = TokenDataset(tmp_path / "train.bin")
    x, y = ds.get_batch(batch_size=4, seq_len=32, device="cpu")
    assert x.shape == y.shape == (4, 32)
    assert x.dtype == torch.int64
    # y is x shifted by one
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_chat_jsonl_source(tmp_path):
    p = tmp_path / "chat.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    docs = list(iter_texts(chat_jsonl=str(p)))
    assert len(docs) == 2
    assert "<|user|>hi<|eos|>" in docs[0]
    assert "<|assistant|>hello<|eos|>" in docs[0]
