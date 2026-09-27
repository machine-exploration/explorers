"""verifiers traces.jsonl -> explorers Episodes. verifiers is imported lazily and only here."""

from pathlib import Path

from explorers.episode import BoardMessage, Episode, Rollout, Span


def _rollout_status(trace: dict) -> str:
    if trace.get("is_timeout"):
        return "timeout"
    if trace.get("errors") or not trace.get("ok", False):
        return "error"
    return "ok"


def episode_from_records(episode_id: str, ok: bool, is_timeout: bool, traces: list[dict],
                         model: str, commit: str, run_dir: str) -> Episode:
    rollouts: list[Rollout] = []
    board: list[BoardMessage] = []
    meta: dict = {}
    for index, t in enumerate(traces):
        info = (t.get("info") or {}).get("explorers")
        ids = list(t.get("token_ids") or [])
        metrics = {k: float(v) for k, v in (t.get("metrics") or {}).items() if v is not None}
        if info is None:
            agent = t.get("agent") or f"unknown_{index}"
            rollouts.append(Rollout(agent_id=agent, status="error", token_ids=ids,
                                    tokens_recorded=bool(ids), spans=[], metrics=metrics))
        else:
            meta = info
            agent = info["agent_id"]
            spans = [Span(id=f"{episode_id}:{agent}:{i}", kind="turn", agent_id=agent, round=turn["round"], text=turn["reply"])
                     for i, turn in enumerate(info["turns"])]
            rollouts.append(Rollout(agent_id=agent, status=_rollout_status(t), token_ids=ids,
                                    tokens_recorded=bool(ids), spans=spans, metrics=metrics))
            if not board:
                board = [BoardMessage(**m) for m in info["board"]]
    status = "timeout" if is_timeout else ("ok" if ok and rollouts else "infra_error")
    return Episode(id=episode_id, scenario=meta.get("scenario", ""), scenario_version=meta.get("scenario_version", ""),
                   model=model, seed=int(meta.get("seed", 0)), status=status, rounds=int(meta.get("rounds", 0)),
                   board=board, rollouts=sorted(rollouts, key=lambda r: r.agent_id),
                   source={"backend": "verifiers", "commit": commit, "run_dir": run_dir})


def episodes_from_run(run_dir: Path, model: str, commit: str) -> list[Episode]:
    from verifiers.v1.trace import WireTrace
    from verifiers.v1.utils.trace_store import read_episodes

    out: list[Episode] = []
    for ep in read_episodes(Path(run_dir), WireTrace):
        traces = []
        for tr in ep.traces:
            branches = tr.branches
            traces.append({"info": tr.info, "metrics": tr.metrics, "ok": tr.ok, "is_timeout": tr.is_timeout,
                           "errors": [e.model_dump() for e in tr.errors],
                           "token_ids": branches[-1].token_ids if branches else [],
                           "agent": tr.agent.name})
        out.append(episode_from_records(ep.id, ep.ok, ep.is_timeout, traces, model, commit, str(run_dir)))
    return out
