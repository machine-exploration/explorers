from explorers.episode import BoardMessage, Episode, Rollout, Span
from explorers.methods import HackLabel, Propagation
from explorers.metrics import auroc, score_detectors, tpr_at_fpr, tpr_at_fpr_ci


def test_tpr_at_fpr_perfect_detector():
    y = [True] * 10 + [False] * 100
    s = [0.9] * 10 + [0.1] * 100
    assert tpr_at_fpr(y, s, 0.01) == 1.0
    assert tpr_at_fpr(y, s, 0.0) == 1.0


def test_tpr_at_fpr_caps_recall_by_fpr():
    # Detector ranks 3 positives above the first false positive: at FPR 0 the only
    # admissible operating point catches those 3 of 10.
    y = [True] * 3 + [False] + [True] * 7 + [False] * 99
    s = [0.9] * 3 + [0.8] + [0.7] * 7 + [0.1] * 99
    assert tpr_at_fpr(y, s, 0.0) == 0.3
    assert tpr_at_fpr(y, s, 0.01) == 1.0  # 1 FP on 100 negatives is admissible


def test_tpr_at_fpr_empty_class_is_none():
    assert tpr_at_fpr([True, True], [0.9, 0.8], 0.05) is None
    assert tpr_at_fpr([False, False], [0.1, 0.2], 0.05) is None


def test_auroc():
    assert auroc([True, False], [0.9, 0.1]) == 1.0
    assert auroc([True, False], [0.1, 0.9]) == 0.0
    assert auroc([True, False], [0.5, 0.5]) == 0.5
    assert auroc([True], [0.5]) is None


def test_tpr_at_fpr_ci_point_and_interval():
    dets = [(f"e{i}", i % 2 == 0, 0.9 if i % 2 == 0 else 0.1) for i in range(200)]
    point, lo, hi = tpr_at_fpr_ci(dets, 0.01, resamples=200, seed=0)
    assert point == 1.0 and lo <= point <= hi


def rollout(agent, passed, texts, status="ok"):
    spans = [Span(id=f"{agent}-{i}", kind="turn", agent_id=agent, round=i, text=t)
             for i, t in enumerate(texts)]
    return Rollout(agent_id=agent, status=status, token_ids=[1], tokens_recorded=True,
                   spans=spans, metrics={"tests_pass": passed, "tests_modified": passed})


def episode(eid, rollouts, board=()):
    return Episode(id=eid, scenario="s", scenario_version="0", model="m", seed=0, status="ok",
                   rounds=3, board=list(board), rollouts=rollouts, source={})


def test_score_detectors_joins_and_ranks():
    board = [BoardMessage(round=0, agent_id="a0", text="just edit the test file")]
    eps = [
        # e1: a0 posts the cheat; a1 echoes it and hacks; a2 stays honest.
        episode("e1", [rollout("a0", 1.0, ["I will edit the test", "", ""]),
                       rollout("a1", 1.0, ["ok", "I will edit the test too", "done"]),
                       rollout("a2", 0.0, ["still stuck", "still stuck", "still stuck"])], board),
        episode("e2", [rollout("a0", 0.0, ["stuck", "stuck", "stuck"]),
                       rollout("a1", 1.0, ["hacked alone", "", ""])]),
    ]
    res = score_detectors(eps, HackLabel(), [Propagation()], resamples=50, seed=0)
    assert res["propagation"]["n"] == 5
    assert res["propagation"]["auroc"] is not None


def test_score_detectors_skips_excluded_and_unscored():
    eps = [episode("e1", [rollout("a0", 1.0, ["edit the test"], status="timeout"),
                          rollout("a1", 0.0, ["stuck"])])]
    res = score_detectors(eps, HackLabel(), [Propagation()], resamples=10)
    assert res["propagation"]["n"] == 1  # only a1: a0's truth is 'excluded'
    assert res["propagation"]["auroc"] is None
