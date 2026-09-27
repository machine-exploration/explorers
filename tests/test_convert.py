from explorers.runtime.verifiers.convert import episode_from_records


def trace(agent, ok=True, timeout=False, ids=(5, 6), metrics=None):
    return {"info": {"explorers": {"agent_id": agent, "turns": [{"round": 0, "reply": "r"}],
                                   "board": [{"round": 0, "agent_id": "agent_0", "text": "p"}],
                                   "rounds": 2, "scenario": "hf-incident-mini", "scenario_version": "0", "seed": 7}},
            "metrics": metrics or {"tests_pass": 1.0, "tests_modified": 0.0, "skipped": None},
            "ok": ok, "is_timeout": timeout, "errors": [], "token_ids": list(ids)}


def test_convert_ok():
    ep = episode_from_records("e1", True, False, [trace("agent_1"), trace("agent_0")], "m", "c", "/r")
    assert ep.status == "ok" and ep.seed == 7 and ep.rounds == 2
    assert [r.agent_id for r in ep.rollouts] == ["agent_0", "agent_1"]
    assert ep.rollouts[0].metrics == {"tests_pass": 1.0, "tests_modified": 0.0}
    assert ep.board[0].text == "p"


def test_missing_tokens_flagged():
    ep = episode_from_records("e1", True, False, [trace("agent_0", ids=())], "m", "c", "/r")
    assert ep.rollouts[0].tokens_recorded is False


def test_timeout_agent_kept():
    ep = episode_from_records("e1", True, False, [trace("agent_0"), trace("agent_1", ok=False, timeout=True)], "m", "c", "/r")
    assert {r.agent_id: r.status for r in ep.rollouts} == {"agent_0": "ok", "agent_1": "timeout"}


def test_no_explorers_traces_is_infra_error():
    ep = episode_from_records("e1", False, False, [], "m", "c", "/r")
    assert ep.status == "infra_error"
