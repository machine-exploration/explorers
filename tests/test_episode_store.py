import pytest
from pydantic import ValidationError

from explorers.episode import BoardMessage, Episode, Rollout, Span, episode_ids, read_episodes, write_episode


def make_episode(eid="e1", status="ok", tokens=(1, 2, 3)):
    span = Span(id="s1", kind="turn", agent_id="a0", round=0, text="hi")
    rollout = Rollout(agent_id="a0", status="ok", token_ids=list(tokens), tokens_recorded=bool(tokens),
                      spans=[span], metrics={"tests_pass": 0.0})
    return Episode(id=eid, scenario="hf-incident-mini", scenario_version="0", model="m", seed=0,
                   status=status, rounds=1, board=[BoardMessage(round=0, agent_id="a0", text="x")],
                   rollouts=[rollout], source={"backend": "test"})


def test_roundtrip(tmp_path):
    write_episode(tmp_path, make_episode("e1"))
    write_episode(tmp_path, make_episode("e2", tokens=()))
    eps = read_episodes(tmp_path)
    assert [e.id for e in eps] == ["e1", "e2"]
    assert eps[1].rollouts[0].tokens_recorded is False
    assert episode_ids(tmp_path) == {"e1", "e2"}


def test_frozen():
    ep = make_episode()
    with pytest.raises(ValidationError):
        ep.status = "timeout"


def test_empty_dir(tmp_path):
    assert read_episodes(tmp_path) == []
