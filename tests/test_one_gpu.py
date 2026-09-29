"""The library on one GPU: checkpoint handles, dtype, the result cache, and one execution path."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import explorers as ex  # noqa: E402
import explorers.model  # noqa: E402
from explorers import ops  # noqa: E402
from explorers.core import Examples, observe  # noqa: E402
from explorers.core.engine import over  # noqa: E402
from explorers.core.state import Trajectory, snapshot  # noqa: E402

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

    def fake_open(name, revision=None, device="cpu", dtype=None, tokenizer=True):
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
    study = ex.Study(refs, tokens()).read("residual", layers=[1], position=-1, reduce=ops.Norm())
    assert study.spec()["models"] == [r.key for r in refs]          # hashable before any download
    ds = study.compute()
    assert list(ds.model.values) == [0, 10, 20] and hub == ["step0", "step10", "step20"]


def test_store_resumes_and_follows_the_study(hub, tmp_path):
    refs = ex.checkpoints("org/model", steps=[0, 10])
    make = lambda s: ex.Study(refs, tokens()).read("residual", layers="*", position=-1, reduce=ops.Project(np.ones(D) * s))
    first = make(1.0).compute(store=tmp_path)
    assert hub == ["step0", "step10"]
    again = make(1.0).compute(store=tmp_path)
    assert hub == ["step0", "step10"]                               # all cached: nothing loaded
    np.testing.assert_array_equal(again.residual.values, first.residual.values)
    make(2.0).compute(store=tmp_path)                               # another study: another key
    assert hub == ["step0", "step10", "step0", "step10"]


def test_store_needs_a_serializable_study(tmp_path):
    study = ex.Study(ex.open(tiny(0)), tokens()).write("residual", 1, fn=lambda h: h)
    with pytest.raises(ValueError, match="callable"):
        study.compute(store=tmp_path)


def test_observe_is_the_same_measurement_as_over():
    states = [snapshot(tiny(seed), "tiny", seed) for seed in (0, 1)]
    run = Trajectory(run="tiny", states=states)
    ex_ = Examples(tokens=tokens(), meta={"split": np.array(["fit"] * 4 + ["eval"] * 2)})
    obs = [observe.jlens_error([1, 2], skip_first=1), observe.logit_lens_error([1, 2], skip_first=1),
           observe.example_loss]
    a = over(run, obs, ex_)
    b = ex.Study(run, ex_).observe(*obs).compute()
    for name in ("jlens_error", "logit_lens_error", "example_loss"):
        np.testing.assert_allclose(b[name].values, a[name].values, rtol=1e-5, atol=1e-6)


def test_observables_refuse_writes():
    study = ex.Study(ex.open(tiny(0)), tokens()).write("residual", 1, fn=ops.Scale(2.0)).observe(observe.loss)
    with pytest.raises(ValueError, match="unmodified"):
        study.compute()


def test_dtype_and_progress(capsys):
    model = ex.open(tiny(0), dtype="bfloat16")
    with model.trace(tokens()) as run:
        x = run.stream("residual").read(2, position=-1)
    assert x.value.dtype == torch.bfloat16 and run.logits.dtype == torch.bfloat16
    ex.Study(model, tokens()).read("residual", layers=[0], position=-1).compute(verbose=True)
    assert "[1/1]" in capsys.readouterr().out
