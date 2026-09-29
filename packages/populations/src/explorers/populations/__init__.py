"""explorers.populations: many agents across turns, on explorers.core.

The scenario format (agents, tasks, shared channels, sandbox, checks), the runtime that plays
scenarios through the verifiers backend and records episodes, and methods and metrics on
episodes (hack label, propagation, Wilson rates).

Planned: an adapter from episodes to explorers.core Examples, with one example per (agent, turn),
so any core observable runs on agent episodes (roadmap I2).
"""
