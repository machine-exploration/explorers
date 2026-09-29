"""explorers.populations: many agents across turns. Not ported yet.

Planned contents, moved from the multi-agent prototype at the repository root and rebuilt on
explorers.core:

- the scenario format (agents, tasks, shared channels, sandbox, checks);
- the runtime that plays scenarios and records episodes;
- an adapter from episodes to explorers.core Examples, with one span per (agent, turn) and the
  board messages each agent read, so any core observable runs on agent episodes;
- population observables and labels (hack label, spread, exposure).
"""
