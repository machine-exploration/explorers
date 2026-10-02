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


def brute_force(model, examples, layer, skip, target="final"):
    """The estimator computed one scalar at a time: for every example, every output (t, k) and
    every source position p, d out[t, k] / d h_layer[p]; summed over lens targets t, then
    averaged over lens sources p and examples. `out` is residual[n] (final) or residual[n-1]."""
    pos = measures.lens_positions(examples.tokens.shape[1], skip)
    d = model.d_model
    total = torch.zeros(d, d, dtype=torch.float64)
    captured = {}
    hook = model.norm.register_forward_pre_hook(lambda _m, a: captured.__setitem__("h", a[0]))
    for row in examples.tokens:
        out = model.module(input_ids=torch.as_tensor(row[None]), output_hidden_states=True)
        h = out.hidden_states[layer]
        final = captured["h"][0] if target == "final" else out.hidden_states[model.n_layers - 1][0]
        for p in pos:
            for k in range(d):
                acc = torch.zeros(d, dtype=torch.float64)
                for t in pos:
                    g, = torch.autograd.grad(final[t, k], h, retain_graph=True, allow_unused=True)
                    if g is None:
                        continue
                    acc += g[0, p].double()
                total[k] += acc
    hook.remove()
    return (total / (len(examples) * len(pos))).float().numpy()


@pytest.mark.parametrize("layer,target", [(0, "final"), (1, "final"), (0, "penultimate")])
def test_jacobian_matches_brute_force(layer, target):
    model, ex = tiny_neox(), random_examples()
    fast = _jacobians(model, ex, {(layer, 1, target)}, batch_size=2, device="cpu")[(layer, 1, target)]
    np.testing.assert_allclose(fast, brute_force(model, ex, layer, 1, target), rtol=1e-4, atol=1e-5)


def test_penultimate_jacobian_of_itself_is_identity():
    model, ex = tiny_neox(layers=3), random_examples()
    j = _jacobians(model, ex, {(2, 1, "penultimate"), (1, 1, "final")}, batch_size=2)
    np.testing.assert_allclose(j[(2, 1, "penultimate")], np.eye(8), atol=1e-6)
    with pytest.raises(ValueError, match="no Jacobian"):
        _jacobians(model, ex, {(3, 1, "penultimate")}, batch_size=2)


@pytest.mark.parametrize("dim_batch", [2, 3, 8])
def test_dim_batch_gives_the_same_jacobian(dim_batch):
    model, ex = tiny_neox(), random_examples()
    one = _jacobians(model, ex, {(0, 1, "final"), (1, 1, "final"), (1, 2, "final"), (0, 1, "penultimate")}, batch_size=2, device="cpu")
    many = _jacobians(model, ex, {(0, 1, "final"), (1, 1, "final"), (1, 2, "final"), (0, 1, "penultimate")}, batch_size=2, device="cpu", dim_batch=dim_batch)
    for key in one:
        np.testing.assert_allclose(many[key], one[key], rtol=1e-5, atol=1e-6)


def test_frozen_parameters():
    model, ex = tiny_neox(), random_examples()
    free = _jacobians(model, ex, {(1, 1, "final")}, batch_size=3, device="cpu")[(1, 1, "final")]
    model.module.requires_grad_(False)
    frozen = _jacobians(model, ex, {(1, 1, "final")}, batch_size=3, device="cpu")[(1, 1, "final")]
    np.testing.assert_allclose(frozen, free, rtol=1e-5, atol=1e-6)


def test_fit_uses_only_fit_rows():
    model = tiny_neox()
    ex = random_examples(n=4, split=np.array(["fit", "eval", "fit", "eval"]))
    only_fit = Examples(tokens=ex.tokens[[0, 2]])
    a = _jacobians(model, ex, {(1, 1, "final")}, batch_size=4, device="cpu")[(1, 1, "final")]
    b = _jacobians(model, only_fit, {(1, 1, "final")}, batch_size=1, device="cpu")[(1, 1, "final")]
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
    ds = ex_.Experiment(run, ex).measure(*obs).compute(store=tmp_path)
    assert ds.jlens_error.dims == ("step", "layer") and list(ds.layer.values) == [0, 1]
    assert ((ds.jlens_error >= 0) & (ds.jlens_error <= 1)).all()
    assert ((ds.logit_lens_error >= 0) & (ds.logit_lens_error <= 1)).all()
    assert ds.jacobian_1.shape == (2, 8, 8)
    assert len(loads) == 2
    again = ex_.Experiment(run, ex).measure(*obs).compute(store=tmp_path)
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
    experiment = ex_.Experiment(model, ex).write("residual", 1, fn=ops.Scale(2.0)).measure(measures.jlens_error([1], skip_first=1))
    with pytest.raises(ValueError, match="unmodified"):
        experiment.compute()


# --- workspace signatures ----------------------------------------------------------------------------

def test_unembed_matrix_is_the_linear_part_of_the_readout():
    """For LayerNorm, logits = W norm-free part: unembed(x) - unembed(0-mean part) is linear in the
    normalized input, so W @ (x - mean) / sigma + const equals the logits."""
    from explorers.execute import unembed_matrix

    model = tiny_neox()
    x = torch.randn(5, 8)
    with torch.no_grad():
        logits = model.unembed(x).numpy()
    w = unembed_matrix(model)
    sigma = torch.sqrt(x.var(dim=-1, unbiased=False) + model.norm.eps).numpy()
    const = model.unembed(torch.zeros(1, 8)).detach().numpy()          # the norm's bias through W_U
    np.testing.assert_allclose(x.numpy() @ w.T / sigma[:, None] + const, logits, rtol=1e-4, atol=1e-4)


def test_dimension_fraction_exact():
    cov = np.diag([4.0, 1.0, 0.0, 0.0])
    assert measures.dimension_fraction(cov, 0.8) == 0.25          # 4 / 5 of the variance in one direction
    assert measures.dimension_fraction(cov, 0.9) == 0.5
    assert measures.dimension_fraction(np.eye(4), 1.0) == 1.0


def test_linear_cka_invariances():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 3))
    cov = a @ a.T + np.eye(8)
    j = rng.normal(size=(8, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert abs(measures.linear_cka(j, j, cov) - 1) < 1e-9
    assert abs(measures.linear_cka(j, 3.0 * j @ q, cov) - 1) < 1e-9          # W J Q: same geometry
    other = rng.normal(size=(8, 8))
    assert measures.linear_cka(j, other, cov) < 1
    assert abs(measures.linear_cka(j, other, cov) - measures.linear_cka(other, j, cov)) < 1e-9


def test_repeat_rate_exact():
    constant_per_text = np.repeat(np.arange(4)[:, None], 10, axis=1)       # each text repeats one token
    r = measures.repeat_rate(constant_per_text, (1, 3))
    assert (r > 0).all()                                                    # persists within, never across
    same_everywhere = np.tile(np.arange(10), (4, 1))                       # position-locked, no context
    np.testing.assert_allclose(measures.repeat_rate(same_everywhere, (1, 3)), 0.0)
    assert np.isnan(measures.repeat_rate(constant_per_text, (10,))[0])


def test_signatures_with_identity_jacobian_equal_the_unembedding():
    """With J = identity the J-lens is the logit lens: its dimension is the unembedding's own, and
    its kurtosis and persistence equal the logit-lens measures."""
    model = tiny_neox(layers=3)
    ex = random_examples(n=6, seq=10, split=np.array(["fit"] * 3 + ["eval"] * 3))
    layers = [1, 2]
    ctx = serve(model, ex, set().union(*(m.reads for m in [
        measures.lens_kurtosis(layers, 1), measures.lens_kurtosis(layers, 1, lens="logit"),
        measures.jlens_dimension(layers, 1)])))
    ctx.jacobian = {(l, 1, "final"): np.eye(8, dtype=np.float32) for l in layers}
    w = ctx.unembed_matrix.astype(np.float64)
    w = w - w.mean(axis=0)
    direct = measures.dimension_fraction(w.T @ w / len(w), 0.9)
    assert (measures.jlens_dimension(layers, 1)(ctx).values == direct).all()
    np.testing.assert_allclose(measures.lens_kurtosis(layers, 1)(ctx).values,
                               measures.lens_kurtosis(layers, 1, lens="logit")(ctx).values)
    np.testing.assert_array_equal(measures.lens_persistence(layers, 1)(ctx).values,
                                  measures.lens_persistence(layers, 1, lens="logit")(ctx).values)
    np.testing.assert_allclose(measures.jlens_cka(layers, 1)(ctx).values, 1.0)


def test_kurtosis_matches_direct_computation():
    model = tiny_neox()
    ex = random_examples(n=4, seq=8)
    m = measures.lens_kurtosis([1], 1)
    ds = ex_.Experiment(model, ex).measure(m).compute()
    ctx = serve(model, ex, set(m.reads))
    h = ctx.stream("residual", 1)[:, 1:7] @ ctx.jacobian[(1, 1, "final")].T
    with torch.no_grad():
        z = model.unembed(torch.as_tensor(h)).double()
    z = z - z.mean(-1, keepdim=True)
    k = ((z ** 4).mean(-1) / (z ** 2).mean(-1) ** 2 - 3).numpy()
    np.testing.assert_allclose(ds.jlens_kurtosis.sel(model=model.key).values, [np.median(k)], rtol=1e-4)


def test_signatures_in_an_experiment():
    model = tiny_neox(layers=3)
    ex = random_examples(n=6, seq=10, split=np.array(["fit"] * 3 + ["eval"] * 3))
    layers = [0, 1, 2]
    ds = ex_.Experiment(model, ex).measure(
        measures.jlens_dimension(layers, 1, target="penultimate"), measures.jlens_cka(layers, 1),
        measures.lens_kurtosis(layers, 1), measures.lens_persistence(layers, 1, offsets=(1, 2))).compute()
    assert ds.jlens_cka.dims == ("model", "layer", "layer2")
    assert ds.jlens_persistence.dims == ("model", "layer", "offset")
    assert ((ds.jlens_dimension > 0) & (ds.jlens_dimension <= 1)).all()
    np.testing.assert_allclose(np.diagonal(ds.jlens_cka.values[0]), 1.0, atol=1e-9)
