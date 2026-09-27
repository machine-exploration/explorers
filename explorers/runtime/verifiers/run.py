import subprocess
import sys
from pathlib import Path

VERIFIERS_COMMIT = "e66ec52966327e79011f4c3bb34a0f9817ce32b8"


def run_vf_eval(package_dir: Path, toml: Path, run_dir: Path) -> Path:
    """Install the generated taskset package and run this one batch of vf-eval into
    exactly `run_dir`. vf-eval always writes to `output_dir / run.dir` (see
    `EvalConfig.output_dir` / `RunConfig.dir`), so we pass `--output-dir <run_dir.parent>
    --run.dir <run_dir.name>` to pin the two together and return `run_dir` itself -
    never a glob over whatever else lives under `run_dir.parent`."""
    pkg_root = package_dir.parent
    (pkg_root / "pyproject.toml").write_text(
        f'[project]\nname = "{package_dir.name.replace("_", "-")}"\nversion = "0.0.0"\n'
        'requires-python = ">=3.11"\n[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        f'[tool.hatch.build.targets.wheel]\npackages = ["{package_dir.name}"]\n', encoding="utf-8")
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, "-q", "-e", str(pkg_root)], check=True)
    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    vf_eval = Path(sys.executable).with_name("vf-eval")   # console script installed by verifiers
    log_path = run_dir.parent / f"{run_dir.name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run([str(vf_eval), "@", str(toml), "--no-push", "--no-rich",
                        "--output-dir", str(run_dir.parent), "--run.dir", run_dir.name],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    return run_dir
