"""The Jacobian lens on tiny random GPT-NeoX models (Pythia's architecture), built locally."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import explorers as ex_  # noqa: E402
from explorers import measures  # noqa: E402
from explorers.data import Examples  # noqa: E402
from explorers.execute import jacobians, serve  # noqa: E402
from explorers.state import State, Trajectory, snapshot  # noqa: E402


def _jacobians(model, examples, requests, batch_size, device="cpu", dim_batch=1):
    return jacobians(model, examples, requests, batch_size, dim_batch)


def tiny_neox(d=8, layers=2, vocab=17, seed=0):
    torch.manual_seed(seed)
    cfg = transformers.GPTNeoXConfig(vocab_size=vocab, hidden_size=d, num_hidden_layers=layers,
                                     num_attention_heads=2, intermediate_size=2 * d, max_position_embeddings=16)
    return ex_.open(transformers.GPTNeoXForCausalLM(cfg).eval())


def random_examples(n=3, seq=6, vocab=17, seed=0, **meta):
    rng = np.random.default_rng(seed)
    return Examples(tokens=rng.integers(0, vocab, size=(n, seq)), meta=meta)


def brute_force(model, examples, layer, skip):
    """The estimator computed one scalar at a time: for every example, every output (t, k) and
    every source position p, d final[t, k] / d h_layer[p]; summed over lens targets t, then
    averaged over lens sources p and examples."""
    pos = measures.lens_positions(examples.tokens.shape[1], skip)
    d = model.d_model
    total = torch.zeros(d, d, dtype=torch.float64)
    captured = {}
    hook = model.norm.register_forward_pre_hook(lambda _m, a: captured.__setitem__("h", a[0]))
    for row in examples.tokens:
        out = model.module(input_ids=torch.as_tensor(row[None]), output_hidden_states=True)
        final, h = captured["h"][0], out.hidden_states[layer]
        for p in pos:
            for k in range(d):
                acc = torch.zeros(d, dtype=torch.float64)
                for t in pos:
                    g, = torch.autograd.grad(final[t, k], h, retain_graph=True)
                    acc += g[0, p].double()
                total[k] += acc
    hook.remove()
    return (total / (len(examples) * len(pos))).float().numpy()


@pytest.mark.parametrize("layer", [0, 1])
def test_jacobian_matches_brute_force(layer):
    model, ex = tiny_neox(), random_examples()
    fast = _jacobians(model, ex, {(layer, 1)}, batch_size=2, device="cpu")[(layer, 1)]
    np.testing.assert_allclose(fast, brute_force(model, ex, layer, 1), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dim_batch", [2, 3, 8])
def test_dim_batch_gives_the_same_jacobian(dim_batch):
    model, ex = tiny_neox(), random_examples()
    one = _jacobians(model, ex, {(0, 1), (1, 1), (1, 2)}, batch_size=2, device="cpu")
    many = _jacobians(model, ex, {(0, 1), (1, 1), (1, 2)}, batch_size=2, device="cpu", dim_batch=dim_batch)
    for key in one:
        np.testing.assert_allclose(many[key], one[key], rtol=1e-5, atol=1e-6)


def test_frozen_parameters():
    model, ex = tiny_neox(), random_examples()
    free = _jacobians(model, ex, {(1, 1)}, batch_size=3, device="cpu")[(1, 1)]
    model.module.requires_grad_(False)
    frozen = _jacobians(model, ex, {(1, 1)}, batch_size=3, device="cpu")[(1, 1)]
    np.testing.assert_allclose(frozen, free, rtol=1e-5, atol=1e-6)


def test_fit_uses_only_fit_rows():
    model = tiny_neox()
    ex = random_examples(n=4, split=np.array(["fit", "eval", "fit", "eval"]))
    only_fit = Examples(tokens=ex.tokens[[0, 2]])
    a = _jacobians(model, ex, {(1, 1)}, batch_size=4, device="cpu")[(1, 1)]
    b = _jacobians(model, only_fit, {(1, 1)}, batch_size=1, device="cpu")[(1, 1)]
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_lens_positions_and_errors():
    assert list(measures.lens_positions(6, 2)) == [2, 3, 4]
    with pytest.raises(ValueError):
        measures.lens_positions(3, 2)
    with pytest.raises(ValueError, match="unknown architecture"):
        ex_.open(torch.nn.Linear(2, 2))


def test_lens_measures_over_a_run_and_cache(tmp_path):
    ex = random_examples(n=6, seq=8, split=np.array(["fit"] * 4 + ["eval"] * 2))
    loads = []

    def state(step, seed):
        s = snapshot(tiny_neox(seed=seed).module, "tiny", step)
        return State(run="tiny", step=step, key=s.key, load=lambda s=s: loads.append(step) or s.load())

    run = Trajectory(run="tiny", states=[state(0, 0), state(10, 1)])
    obs = [measures.jlens_error([0, 1], skip_first=1), measures.logit_lens_error([0, 1], skip_first=1),
           measures.jacobian(1, skip_first=1)]
    ds = ex_.Study(run, ex).measure(*obs).compute(store=tmp_path)
    assert ds.jlens_error.dims == ("step", "layer") and list(ds.layer.values) == [0, 1]
    assert ((ds.jlens_error >= 0) & (ds.jlens_error <= 1)).all()
    assert ((ds.logit_lens_error >= 0) & (ds.logit_lens_error <= 1)).all()
    assert ds.jacobian_1.shape == (2, 8, 8)
    assert len(loads) == 2
    again = ex_.Study(run, ex).measure(*obs).compute(store=tmp_path)
    assert len(loads) == 2                                   # everything came from the store
    np.testing.assert_array_equal(again.jlens_error.values, ds.jlens_error.values)


def test_final_is_before_the_norm():
    model, ex = tiny_neox(), random_examples(n=2)
    ctx = serve(model, ex, {"residual:final"})
    with torch.no_grad():
        normed = model.norm(torch.as_tensor(ctx.final)).numpy()
        after = model.module(input_ids=torch.as_tensor(ex.tokens), output_hidden_states=True).hidden_states[-1]
    np.testing.assert_allclose(normed, after.numpy(), rtol=1e-5, atol=1e-5)   # HF's last hidden state is after the norm


def test_lens_refuses_writes():
    from explorers import ops

    model, ex = tiny_neox(), random_examples()
    study = ex_.Study(model, ex).write("residual", 1, fn=ops.Scale(2.0)).measure(measures.jlens_error([1], skip_first=1))
    with pytest.raises(ValueError, match="unmodified"):
        study.compute()
