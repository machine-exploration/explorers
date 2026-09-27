"""Scenario -> a verifiers taskset package + a vf-eval TOML. Pure file generation."""

import json
import re
from pathlib import Path
from string import Template

from explorers.scenario import Scenario

TEMPLATES = Path(__file__).with_name("package")


def module_name(scenario: Scenario) -> str:
    return "explorers_scn_" + scenario.name.replace("-", "_").lower()


def _read_task(task_dir: Path) -> dict:
    files = {str(p.relative_to(task_dir)): p.read_text(encoding="utf-8")
             for p in sorted(task_dir.rglob("*")) if p.is_file() and p != task_dir / "prompt.md"}
    return {"name": task_dir.name, "prompt": (task_dir / "prompt.md").read_text(encoding="utf-8"), "files": files}


def _agent_fields(n: int) -> str:
    return "\n".join(f'    agent_{i}: vf.AgentConfig = vf.AgentConfig(harness={{"id": "bash"}})' for i in range(n))


def _toml(scenario: Scenario, module: str, model: str, base_url: str, episodes: int) -> str:
    lines = [f'model = "{model}"', f"num_tasks = {episodes}", "push = false", "",
             "[env]", f"max_concurrent_agents = {scenario.agents.count}", "",
             "[env.taskset]", f'id = "{module}"', ""]
    for i in range(scenario.agents.count):
        lines += [f"[env.agent_{i}.runtime]", 'type = "docker"', f'image = "{scenario.sandbox.image}"',
                  "allow = []", f"memory = {scenario.sandbox.memory_gb}", "",
                  f"[env.agent_{i}.client]", 'type = "train"', f'base_url = "{base_url}"',
                  'api_key_var = "VLLM_API_KEY"', f'renderer_model_name = "{model}"', ""]
    return "\n".join(lines)


def compile_scenario(scenario: Scenario, out_dir: Path, model: str, base_url: str,
                     seed: int, episodes: int) -> tuple[Path, Path]:
    if scenario.tasks.assignment != "same":
        raise ValueError(f"tasks.assignment '{scenario.tasks.assignment}' is not supported in v0")
    if not re.fullmatch(r"^[a-z0-9][a-z0-9_-]{0,63}$", scenario.name):
        raise ValueError(f"invalid scenario name {scenario.name!r}")
    if not re.fullmatch(r"^[A-Za-z0-9._-]{1,32}$", scenario.version):
        raise ValueError(f"invalid scenario version {scenario.version!r}")

    module = module_name(scenario)
    pkg = out_dir / "package" / module
    pkg.mkdir(parents=True, exist_ok=True)
    tasks = [_read_task(d) for d in scenario.task_dirs()]
    for spec in tasks:
        missing = [f for f in scenario.checks.test_files if f not in spec["files"]]
        if missing:
            raise ValueError(f"task {spec['name']} lacks test files {missing}")
    (pkg / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    values = dict(
        SCENARIO=scenario.name, VERSION=scenario.version, N_AGENTS=scenario.agents.count,
        MAX_ROUNDS=scenario.limits.max_rounds, BOARD=repr(scenario.board.enabled),
        SCENARIO_JSON=repr(json.dumps(scenario.name)),
        VERSION_JSON=repr(json.dumps(scenario.version)),
        ROLE_JSON=repr(json.dumps(scenario.agents.role_prompt)),
        TEST_COMMAND_JSON=repr(json.dumps(scenario.checks.test_command)),
        TEST_FILES_JSON=repr(json.dumps(scenario.checks.test_files)),
        AGENT_FIELDS=_agent_fields(scenario.agents.count), SEED=seed, MODULE=module,
    )
    for name in ("taskset.py", "__init__.py"):
        text = Template((TEMPLATES / f"{name}.tmpl").read_text(encoding="utf-8")).substitute(values)
        (pkg / name).write_text(text, encoding="utf-8")
    toml = out_dir / "vf-eval.toml"
    toml.write_text(_toml(scenario, module, model, base_url, episodes), encoding="utf-8")
    return pkg, toml
