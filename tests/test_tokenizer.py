import pytest

from slm.tokenizer import ASSISTANT, EOS, SLMTokenizer, USER

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "Sphinx of black quartz, judge my vow.",
] * 50


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    return SLMTokenizer.train(iter(CORPUS), vocab_size=512, save_path=path)


def test_roundtrip(tok):
    text = "The quick brown fox jumps over the lazy dog."
    assert tok.decode(tok.encode(text)) == text


def test_special_token_ids(tok):
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    ids = tok.encode("hello", bos=True, eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id


def test_special_tokens_encode_as_single_ids(tok):
    ids = tok.encode(f"{USER}hi{EOS}")
    assert ids[0] == tok.tk.token_to_id(USER)
    assert ids[-1] == tok.eos_id


def test_chat_template(tok):
    text = tok.format_chat([{"role": "user", "content": "hi"}])
    assert text == f"{USER}hi{EOS}{ASSISTANT}"


def test_save_and_load(tok, tmp_path):
    path = tmp_path / "t.json"
    tok.tk.save(str(path))
    loaded = SLMTokenizer.load(path)
    assert loaded.encode("zebras jump") == tok.encode("zebras jump")
