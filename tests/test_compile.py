import ast

import pytest

from explorers.runtime.verifiers.compile import compile_scenario
from explorers.scenario import load_scenario

SCN = """
format: explorers.scenario/v0
name: hf-incident-mini
version: "0"
agents: {count: 3, role_prompt: "You are an engineer."}
tasks: {directory: tasks}
sandbox: {network: none}
limits: {max_rounds: 4}
checks: {test_command: "python -m pytest -q tests", test_files: ["tests/test_solution.py"]}
"""


def scenario(tmp_path, with_tests=True):
    t = tmp_path / "tasks" / "t1"
    (t / "src").mkdir(parents=True)
    (t / "prompt.md").write_text("Make the tests pass.")
    (t / "src" / "solution.py").write_text("def f():\n    return 1\n")
    if with_tests:
        (t / "tests").mkdir()
        (t / "tests" / "test_solution.py").write_text("from src.solution import f\n\ndef test():\n    assert f() == 1 and f() == 2\n")
    (tmp_path / "scenario.yaml").write_text(SCN)
    return load_scenario(tmp_path / "scenario.yaml")


def test_compile_outputs(tmp_path):
    pkg, toml = compile_scenario(scenario(tmp_path), tmp_path / "out", "m", "http://h:8000", 100, 5)
    ast.parse((pkg / "taskset.py").read_text())      # valid Python
    text = toml.read_text()
    assert 'id = "explorers_scn_hf_incident_mini"' in text and "num_tasks = 5" in text
    assert text.count("allow = []") == 3 and text.count('type = "train"') == 3
    assert "agent_2: vf.AgentConfig" in (pkg / "taskset.py").read_text()


def test_missing_test_file_rejected(tmp_path):
    with pytest.raises(ValueError, match="lacks test files"):
        compile_scenario(scenario(tmp_path, with_tests=False), tmp_path / "out", "m", "u", 0, 1)


def test_round_robin_rejected(tmp_path):
    scn = scenario(tmp_path)
    scn = scn.model_copy(update={"tasks": scn.tasks.model_copy(update={"assignment": "round_robin"})})
    with pytest.raises(ValueError, match="round_robin"):
        compile_scenario(scn, tmp_path / "out", "m", "u", 0, 1)
