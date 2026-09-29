"""Selective reads, reductions on the device, and interventions as data."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import explorers as ex  # noqa: E402
from explorers import ops  # noqa: E402

V, D, L, SEQ = 16, 32, 3, 6


def neox(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.GPTNeoXConfig(vocab_size=V, hidden_size=D, num_hidden_layers=L, num_attention_heads=4,
                                     intermediate_size=2 * D, max_position_embeddings=32)
    return ex.open(transformers.GPTNeoXForCausalLM(cfg))


def tokens(n=4, seed=0):
    return np.random.default_rng(seed).integers(0, V, size=(n, SEQ))


# --- reads keep only what they select --------------------------------------------------------------

def test_read_keeps_only_the_selection():
    model, ids = neox(), tokens()
    with model.trace(ids) as run:
        one = run.stream("residual").read(2, position=3)
        full = run.stream("residual").read(2)
    assert one.value.shape == (len(ids), D)
    assert one.value.untyped_storage().nbytes() == len(ids) * D * 4        # its own small buffer
    assert one._full is None                                                # no reference to the stream
    torch.testing.assert_close(one.value, full.value[:, 3])


def test_reductions_run_on_the_device():
    model, ids = neox(), tokens()
    u = torch.randn(D)
    with model.trace(ids) as run:
        x = run.stream("residual").read(2, position=-1)
        p = run.stream("residual").read(2, position=-1, reduce=ops.Project(u))
        n = run.stream("residual").read(2, reduce=ops.Norm())
    torch.testing.assert_close(p.value, x.value @ u)
    assert p.value.shape == (len(ids),) and n.value.shape == (len(ids), SEQ)
    ds = ex.Study(model, ids).read("residual", layers=[1, 2], position=-1, reduce=ops.Project(u)).compute()
    assert ds.residual.dims == ("model", "example", "layer")
    np.testing.assert_allclose(ds.residual.sel(layer=2).values[0], p.value.numpy(), rtol=1e-5)


# --- interventions --------------------------------------------------------------------------------

def run_with(model, ids, op, layer=2, position=-1):
    with model.trace(ids) as run:
        r = run.stream("residual")
        r.write(layer, position, fn=op)
        after = r.read(layer, position)
    return run, after


def test_intervention_semantics():
    model, ids = neox(), tokens()
    with model.trace(ids) as base:
        x = base.stream("residual").read(2, position=-1)
    u = torch.randn(D)
    _, a = run_with(model, ids, ops.Add(u, 2.0))
    torch.testing.assert_close(a.value, x.value + 2.0 * u)
    _, s = run_with(model, ids, ops.Scale(0.5))
    torch.testing.assert_close(s.value, 0.5 * x.value)
    _, z = run_with(model, ids, ops.Ablate())
    assert torch.all(z.value == 0)
    _, o = run_with(model, ids, ops.ProjectOut(u))
    torch.testing.assert_close(o.value @ (u / u.norm()), torch.zeros(len(ids)), atol=1e-5, rtol=0)
    run, w = run_with(model, ids, ops.Set(x.value))
    torch.testing.assert_close(run.logits, base.logits)                    # setting the same value: no change


@pytest.mark.parametrize("op", [ops.Add(np.arange(D, dtype=np.float32), 0.5), ops.Set(np.ones((4, D))),
                                ops.Scale(3.0), ops.Ablate(), ops.ProjectOut(np.ones(D)),
                                ops.Project(np.ones(D)), ops.Norm(), ops.LogitDiff(3, 5, -1),
                                ops.LogitDiff([1, 2, 3, 4], 5)])
def test_ops_round_trip_through_json(op):
    again = ops.Op.from_dict(json.loads(json.dumps(op.to_dict())))
    assert type(again) is type(op)
    h = torch.randn(4, SEQ, D) if not isinstance(op, ops.LogitDiff) else torch.randn(4, SEQ, V)
    x = h[:, -1] if isinstance(op, ops.Set) else h
    torch.testing.assert_close(again(x), op(x))


# --- studies as data ---------------------------------------------------------------------------------

def data_study(model, ids, scale=2.0):
    return (ex.Study(model, ids)
            .read("residual", layers="*", position=-1, reduce=ops.Norm())
            .write("residual", 1, position=-1, fn=ex.steering.add(np.ones(D), scale))
            .measure("ld", ex.patching.logit_diff(3, 5))
            .patch(source=ids[::-1].copy(), metric=ex.patching.logit_diff(3, 5), layers=[0, L], positions=[0]))


def test_study_spec_is_json_and_its_key_follows_content():
    model, ids = neox(), tokens()
    spec = data_study(model, ids).spec()
    assert spec["format"] == "explorers.study/v0"
    assert json.loads(json.dumps(spec)) == spec
    assert data_study(model, ids).key() == data_study(model, ids).key()
    assert data_study(model, ids, scale=3.0).key() != data_study(model, ids).key()
    assert data_study(neox(seed=1), ids).key() != data_study(model, ids).key()           # other weights


def test_callables_run_locally_but_do_not_serialize():
    model, ids = neox(), tokens()
    study = ex.Study(model, ids).write("residual", 1, position=-1, fn=lambda h: h * 2).measure("ld", ex.patching.logit_diff(3, 5))
    study.compute()                                                                        # runs
    with pytest.raises(ValueError, match="callable"):
        study.spec()
    with pytest.raises(ValueError, match="content key"):
        ex.Study(lambda: model, ids).spec()
    with model.trace(ids) as run:
        run.stream("residual").write(1, fn=ops.Scale(2.0))
    assert run.serializable
    with model.trace(ids) as run2:
        run2.stream("residual").write(1, fn=lambda h: h)
    assert not run2.serializable
