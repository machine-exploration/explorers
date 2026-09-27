import pytest
from pydantic import ValidationError

from explorers.scenario import load_scenario

VALID = """
format: explorers.scenario/v0
name: hf-incident-mini
version: "0"
agents: {count: 3, role_prompt: "You are an engineer."}
tasks: {directory: tasks}
sandbox: {network: none}
checks: {test_command: "python -m pytest -q tests", test_files: ["tests/test_solution.py"]}
"""


def write(tmp_path, text):
    (tmp_path / "tasks" / "t1").mkdir(parents=True)
    p = tmp_path / "scenario.yaml"
    p.write_text(text)
    return p


def test_valid(tmp_path):
    s = load_scenario(write(tmp_path, VALID))
    assert s.agents.count == 3 and s.limits.max_rounds == 8
    assert [d.name for d in s.task_dirs()] == ["t1"]


def test_network_must_be_none(tmp_path):
    with pytest.raises(ValidationError):
        load_scenario(write(tmp_path, VALID.replace("network: none", "network: bridge")))


def test_test_files_required(tmp_path):
    bad = VALID.replace(', test_files: ["tests/test_solution.py"]', "")
    with pytest.raises(ValidationError):
        load_scenario(write(tmp_path, bad))


def test_unknown_field_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_scenario(write(tmp_path, VALID + "extra: 1\n"))
