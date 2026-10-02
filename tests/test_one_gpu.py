"""The library on one GPU: checkpoint handles, dtype, the result cache, and one execution path."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import explorers as ex  # noqa: E402
import explorers.model  # noqa: E402
from explorers import ops  # noqa: E402
from explorers import measures  # noqa: E402

V, D, L, SEQ = 16, 32, 3, 8


def tiny(seed):
    torch.manual_seed(seed)
    cfg = transformers.GPTNeoXConfig(vocab_size=V, hidden_size=D, num_hidden_layers=L, num_attention_heads=4,
                                     intermediate_size=2 * D, max_position_embeddings=32)
    return transformers.GPTNeoXForCausalLM(cfg).eval()


@pytest.fixture
def hub(monkeypatch):
    """A fake hub: `open(name, revision)` builds a tiny model seeded by the step in the revision."""
    loads = []

    def fake_open(name, revision=None, device="cpu", dtype=None, tokenizer=True, adapter=None):
        loads.append(revision)
        return explorers.model.Model(tiny(int(revision.removeprefix("step"))), name=name, revision=revision,
                                     device=device)

    monkeypatch.setattr(explorers.model, "open", fake_open)
    return loads


def tokens(n=6, seed=0):
    return np.random.default_rng(seed).integers(0, V, size=(n, SEQ))


def test_checkpoint_handles_have_keys_before_loading(hub):
    refs = ex.checkpoints("org/model", steps=[0, 10, 20])
    assert [r.key for r in refs] == ["org/model@step0", "org/model@step10", "org/model@step20"]
    assert hub == []                                              # nothing loaded yet
    experiment = ex.Experiment(refs, tokens()).read("residual", layers=[1], position=-1, reduce=ops.Norm())
    assert experiment.spec()["models"] == [r.key for r in refs]          # hashable before any download
    ds = experiment.compute()
    assert list(ds.step.values) == [0, 10, 20] and hub == ["step0", "step10", "step20"]


def test_store_resumes_and_follows_the_experiment(hub, tmp_path):
    refs = ex.checkpoints("org/model", steps=[0, 10])
    make = lambda s: ex.Experiment(refs, tokens()).read("residual", layers="*", position=-1, reduce=ops.Project(np.ones(D) * s))
    first = make(1.0).compute(store=tmp_path)
    assert hub == ["step0", "step10"]
    again = make(1.0).compute(store=tmp_path)
    assert hub == ["step0", "step10"]                               # all cached: nothing loaded
    np.testing.assert_array_equal(again.residual.values, first.residual.values)
    make(2.0).compute(store=tmp_path)                               # another experiment: another key
    assert hub == ["step0", "step10", "step0", "step10"]


def test_store_needs_a_serializable_experiment(tmp_path):
    experiment = ex.Experiment(ex.open(tiny(0)), tokens()).write("residual", 1, fn=lambda h: h)
    with pytest.raises(ValueError, match="callable"):
        experiment.compute(store=tmp_path)


def test_writes_apply_to_measures():
    """One execution path: measures see the model as the experiment modified it."""
    model, ids = ex.open(tiny(0)), tokens()
    base = ex.Experiment(model, ids).measure(measures.loss, measures.residual_norm(2)).compute()
    ablated = (ex.Experiment(model, ids).write("residual", 2, fn=ops.Ablate())
               .measure(measures.loss, measures.residual_norm(2)).compute())
    assert np.allclose(ablated.residual_norm_2.values, 0)
    assert not np.isclose(ablated.loss.item(), base.loss.item())


def test_logit_diff_is_the_same_as_a_patching_metric():
    model, ids = ex.open(tiny(0)), tokens()
    ld = measures.logit_diff(3, 5)
    ds = ex.Experiment(model, ids).measure(ld).compute()
    with model.trace(ids) as run:
        pass
    np.testing.assert_allclose(ds.logit_diff.values[0], (run.logits[:, -1, 3] - run.logits[:, -1, 5]).numpy(), rtol=1e-5)


def test_dtype_and_progress(capsys):
    model = ex.open(tiny(0), dtype="bfloat16")
    with model.trace(tokens()) as run:
        x = run.stream("residual").read(2, position=-1)
    assert x.value.dtype == torch.bfloat16 and run.logits.dtype == torch.bfloat16
    ex.Experiment(model, tokens()).read("residual", layers=[0], position=-1).compute(verbose=True)
    assert "[1/1]" in capsys.readouterr().out
