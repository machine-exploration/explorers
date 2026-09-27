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
