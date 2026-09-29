import ast
import tomllib

import pytest

from explorers.populations.runtime.verifiers.compile import compile_scenario
from explorers.populations.scenario import load_scenario

SCN = """
format: explorers.populations.scenario/v0
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
    (t / "tests").mkdir()
    (t / "prompt.md").write_text("Make the tests pass.")
    (t / "src" / "solution.py").write_text("def f():\n    return 1\n")
    if with_tests:
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


def test_bad_name_rejected(tmp_path):
    scn = scenario(tmp_path)
    scn = scn.model_copy(update={"name": 'evil"; import os; x="'})
    with pytest.raises(ValueError, match="invalid scenario name"):
        compile_scenario(scn, tmp_path / "out", "m", "u", 0, 1)


def test_timeout_keys_present(tmp_path):
    scn = scenario(tmp_path)
    _, toml = compile_scenario(scn, tmp_path / "out", "m", "u", 0, 1)
    parsed = tomllib.loads(toml.read_text())
    for i in range(scn.agents.count):
        timeout = parsed["env"][f"agent_{i}"]["timeout"]
        assert timeout["rollout"] == scn.limits.rollout_timeout_s
        assert timeout["scoring"] == 2 * scn.sandbox.exec_timeout_s + 60


def test_max_concurrent_top_level(tmp_path):
    _, toml = compile_scenario(scenario(tmp_path), tmp_path / "out", "m", "u", 0, 1, max_concurrent=7)
    parsed = tomllib.loads(toml.read_text())
    assert parsed["max_concurrent"] == 7


def test_toml_strings_json_escaped(tmp_path):
    scn = scenario(tmp_path)
    scn = scn.model_copy(update={"sandbox": scn.sandbox.model_copy(update={"image": 'weird"image'})})
    _, toml = compile_scenario(scn, tmp_path / "out", 'my"model', "http://h:8000", 0, 1)
    parsed = tomllib.loads(toml.read_text())
    assert parsed["model"] == 'my"model'
    assert parsed["env"]["agent_0"]["runtime"]["image"] == 'weird"image'


def test_symlink_in_task_dir_rejected(tmp_path):
    scn = scenario(tmp_path)
    t1 = scn.root / "tasks" / "t1"
    (t1 / "src" / "evil.py").symlink_to(t1 / "prompt.md")
    with pytest.raises(ValueError, match="symlink"):
        compile_scenario(scn, tmp_path / "out", "m", "u", 0, 1)
