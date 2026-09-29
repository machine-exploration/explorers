import pytest
from pydantic import ValidationError

from explorers.populations.episode import BoardMessage, Episode, Rollout, Span, episode_ids, read_episodes, write_episode


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


def test_truncated_last_line_warns_and_is_skipped(tmp_path):
    write_episode(tmp_path, make_episode("e1"))
    with (tmp_path / "episodes.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"format": "explorers.populations.episode/v0", "id": "e2", "trun')  # no closing, no newline
    with pytest.warns(UserWarning):
        eps = read_episodes(tmp_path)
    assert [e.id for e in eps] == ["e1"]


def test_bad_line_elsewhere_still_raises(tmp_path):
    write_episode(tmp_path, make_episode("e1"))
    write_episode(tmp_path, make_episode("e2"))
    path = tmp_path / "episodes.jsonl"
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    lines[0] = "{not json at all"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(Exception):
        read_episodes(tmp_path)
