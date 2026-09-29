"""The Hugging Face backend on a tiny random GPT-NeoX (Pythia's architecture), built locally."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from explorers.learning import Checkpoint, number_comparison  # noqa: E402
from explorers.learning.activations import Cache, HFSource  # noqa: E402

N_LAYERS = 2


def tiny_loader(checkpoint):
    words = ["[PAD]", "[UNK]", "is", "greater", "than", "."] + [str(i) for i in range(100)]
    wordlevel = tokenizers.models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    tok = tokenizers.Tokenizer(wordlevel)
    tok.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")
    torch.manual_seed(checkpoint.step)
    config = transformers.GPTNeoXConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=N_LAYERS,
                                        num_attention_heads=4, intermediate_size=64, max_position_embeddings=32)
    return transformers.GPTNeoXForCausalLM(config).eval(), fast


def texts_dataset(n=12):
    ds = number_comparison(n=40)
    # the word-level tokenizer splits on spaces, so detach the full stop
    return type(ds)(ds.name, ds.version, [t.replace(".", " .") for t in ds.texts[:n]], ds.labels[:n], ds.groups[:n])


@pytest.mark.parametrize("position", ["last", "mean"])
def test_shapes_and_batch_invariance(position):
    ds = texts_dataset()
    ck = Checkpoint("tiny", "step7", 7)
    one = HFSource(batch_size=1, position=position, loader=tiny_loader).read(ck, ds, [0, N_LAYERS])
    many = HFSource(batch_size=5, position=position, loader=tiny_loader).read(ck, ds, [0, N_LAYERS])
    for layer in (0, N_LAYERS):
        assert one[layer].shape == (len(ds.texts), 32) and one[layer].dtype == np.float32
        np.testing.assert_allclose(one[layer], many[layer], atol=1e-5)   # padding must not leak in


def test_padding_uses_last_real_token():
    ds = type(texts_dataset())("pad", "0", ["5 is greater than 3 .", "5 ."], [True, False], ["a", "b"])
    ck = Checkpoint("tiny", "step7", 7)
    both = HFSource(batch_size=2, loader=tiny_loader).read(ck, ds, [1])[1]
    alone = HFSource(batch_size=1, loader=tiny_loader).read(
        ck, type(ds)("pad", "0", ["5 ."], [False], ["b"]), [1])[1]
    np.testing.assert_allclose(both[1], alone[0], atol=1e-5)


def test_cache_skips_the_model(tmp_path):
    ds, ck = texts_dataset(), Checkpoint("tiny", "step7", 7)
    calls = []

    def counting_loader(c):
        calls.append(c)
        return tiny_loader(c)

    src = HFSource(cache=Cache(tmp_path), loader=counting_loader)
    first = src.read(ck, ds, [0, 1])
    second = src.read(ck, ds, [0, 1])
    assert len(calls) == 1
    np.testing.assert_array_equal(first[1], second[1])
