from types import SimpleNamespace

from explorers.episode import BoardMessage, Episode, Rollout, Span, write_episode, read_episodes
from explorers.exploration import Exploration, append_new


def ep(eid, status="ok", passes=(1.0, 0.0), board=()):
    rs = [Rollout(agent_id=f"agent_{i}", status="ok", token_ids=[1], tokens_recorded=True,
                  spans=[Span(id=f"{eid}{i}", kind="turn", agent_id=f"agent_{i}", round=2, text="")],
                  metrics={"tests_pass": p, "tests_modified": p}) for i, p in enumerate(passes)]
    return Episode(id=eid, scenario="s", scenario_version="0", model="m", seed=0, status=status, rounds=3,
                   board=list(board), rollouts=rs, source={})


def test_report():
    eps = [ep("e1"), ep("e2", board=[BoardMessage(round=0, agent_id="agent_0", text="edit the test")]),
           ep("e3", status="infra_error")]
    r = Exploration.report(eps)
    assert r["episodes_total"] == 3 and r["episodes_excluded"] == {"infra_error": 1}
    assert r["rollouts_scored"] == 4 and r["hack_rate"][0] == 0.5
    assert r["board_posts_per_episode"] == 0.5


def test_store_does_not_duplicate(tmp_path):
    assert append_new(tmp_path, [ep("e1")]) == 1
    assert append_new(tmp_path, [ep("e1"), ep("e2"), ep("e2")]) == 1
    assert [e.id for e in read_episodes(tmp_path)] == ["e1", "e2"]


def test_run_passes_max_concurrent_and_seed_offset(tmp_path, monkeypatch):
    calls = {}

    def fake_compile_scenario(scenario, out_dir, model, base_url, seed, episodes, seed_offset=0, max_concurrent=2):
        calls.update(seed=seed, episodes=episodes, seed_offset=seed_offset, max_concurrent=max_concurrent)
        return (tmp_path / "pkg", tmp_path / "eval.toml")

    def fake_run_vf_eval(pkg, toml, run_dir):
        calls["run_dir"] = run_dir
        return run_dir

    def fake_episodes_from_run(run_dir, model, commit, fallback_scenario="", fallback_version=""):
        calls["fallback_scenario"] = fallback_scenario
        return [ep("e1", passes=(1.0,))]

    monkeypatch.setattr("explorers.runtime.verifiers.compile.compile_scenario", fake_compile_scenario)
    monkeypatch.setattr("explorers.runtime.verifiers.run.run_vf_eval", fake_run_vf_eval)
    monkeypatch.setattr("explorers.runtime.verifiers.convert.episodes_from_run", fake_episodes_from_run)

    exploration = Exploration.__new__(Exploration)
    exploration.scenario = SimpleNamespace(name="s", version="0")
    exploration.model, exploration.base_url = "m", "http://h"
    exploration.episodes, exploration.seed = 1, 5
    exploration.max_concurrent = 9
    exploration.out_dir = tmp_path / "out"

    report = exploration.run()
    assert calls["seed"] == 5 and calls["seed_offset"] == 0 and calls["max_concurrent"] == 9
    assert calls["fallback_scenario"] == "s"
    assert report["episodes_total"] == 1


def test_run_raises_when_batch_is_all_non_ok(tmp_path, monkeypatch):
    def fake_compile_scenario(*a, **kw):
        return (tmp_path / "pkg", tmp_path / "eval.toml")

    def fake_run_vf_eval(pkg, toml, run_dir):
        return run_dir

    def fake_episodes_from_run(run_dir, model, commit, fallback_scenario="", fallback_version=""):
        return [ep("e1", status="infra_error")]

    monkeypatch.setattr("explorers.runtime.verifiers.compile.compile_scenario", fake_compile_scenario)
    monkeypatch.setattr("explorers.runtime.verifiers.run.run_vf_eval", fake_run_vf_eval)
    monkeypatch.setattr("explorers.runtime.verifiers.convert.episodes_from_run", fake_episodes_from_run)

    exploration = Exploration.__new__(Exploration)
    exploration.scenario = SimpleNamespace(name="s", version="0")
    exploration.model, exploration.base_url = "m", "http://h"
    exploration.episodes, exploration.seed = 1, 0
    exploration.max_concurrent = 2
    exploration.out_dir = tmp_path / "out2"

    try:
        exploration.run()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "infra_error" in str(e)
