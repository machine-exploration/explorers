import subprocess
import sys
from pathlib import Path

VERIFIERS_COMMIT = "e66ec52966327e79011f4c3bb34a0f9817ce32b8"


def run_vf_eval(package_dir: Path, toml: Path, out_dir: Path) -> Path:
    pkg_root = package_dir.parent
    (pkg_root / "pyproject.toml").write_text(
        f'[project]\nname = "{package_dir.name.replace("_", "-")}"\nversion = "0.0.0"\n'
        'requires-python = ">=3.11"\n[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        f'[tool.hatch.build.targets.wheel]\npackages = ["{package_dir.name}"]\n', encoding="utf-8")
    subprocess.run(["uv", "pip", "install", "-q", "-e", str(pkg_root)], check=True)
    vf_dir = out_dir / "vf"
    vf_eval = Path(sys.executable).with_name("vf-eval")   # console script installed by verifiers
    subprocess.run([str(vf_eval), "@", str(toml), "--no-push", "--no-rich",
                    "--output-dir", str(vf_dir)], check=True)
    return vf_dir
