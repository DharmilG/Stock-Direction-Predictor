from __future__ import annotations

import importlib
import os
import subprocess
from importlib import metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "optuna": "optuna",
    "mlflow": "mlflow",
    "yfinance": "yfinance",
    "shap": "shap",
    "matplotlib": "matplotlib",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
    "yaml": "PyYAML",
    "joblib": "joblib",
    "requests": "requests",
}


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def in_target_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def run(cmd: list[str]) -> None:
    print("+", " ".join(map(str, cmd)))
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def ensure_venv() -> Path:
    py = venv_python()
    if not py.exists():
        print(f"Creating virtual environment: {VENV}")
        run([sys.executable, "-m", "venv", str(VENV)])
    return py


def _requirements_satisfied(py: Path) -> bool:
    code = r"""
import importlib.util
from importlib import metadata
from pathlib import Path
from packaging.requirements import Requirement

requirements = []
for raw in Path(%r).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        continue
    try:
        requirements.append(Requirement(line))
    except Exception:
        continue
missing = []
for req in requirements:
    try:
        metadata.version(req.name)
    except metadata.PackageNotFoundError:
        missing.append(f"{req.name}:not-installed")
        continue
    try:
        if req.specifier and not req.specifier.contains(metadata.version(req.name), prereleases=True):
            missing.append(f"{req.name}:version-mismatch")
    except Exception:
        missing.append(f"{req.name}:check-failed")
print("\n".join(missing))
raise SystemExit(0 if not missing else 2)
""" % str(REQUIREMENTS)
    result = subprocess.run([str(py), "-c", code], cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0


def dependencies_ready(py: Path) -> bool:
    # Check package versions as well as importability. This prevents a silently
    # incompatible future release from being accepted just because it imports.
    if not _requirements_satisfied(py):
        return False
    code = "import importlib.util; mods=%r; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('\n'.join(missing)); raise SystemExit(0 if not missing else 2)" % list(REQUIRED_IMPORTS)
    result = subprocess.run([str(py), "-c", code], cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0 and not result.stdout.strip()

def install_dependencies(py: Path) -> None:
    print("Installing/updating project dependencies...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    run([str(py), "-m", "pip", "install", "-e", str(ROOT)])


def main() -> None:
    if sys.version_info < (3, 11):
        print("Python 3.11+ is required. Current version:", sys.version)
        raise SystemExit(2)
    py = ensure_venv()
    if not in_target_venv():
        if not dependencies_ready(py):
            install_dependencies(py)
        else:
            # Keep editable install synchronized without reinstalling all packages.
            run([str(py), "-m", "pip", "install", "-e", str(ROOT), "--no-deps"])
        # "activate" is unnecessary for a subprocess; executing with the venv interpreter is deterministic.
        run([str(py), str(ROOT / "start_run.py"), "--inside-venv"])
        return

    if not dependencies_ready(py):
        install_dependencies(py)
    run([str(py), "-m", "stockml.main", "all"])
    # F-22 steps 5/6/8/9: reference-data + news-coverage warnings, then the
    # §12.2 final block. `report` exits 1 when below the majority baseline,
    # and run() propagates that code so "did it work" is machine-checkable.
    _warn_reference_and_news_gaps(py)
    run([str(py), "-m", "stockml.main", "report"])


def _warn_reference_and_news_gaps(py: Path) -> None:
    code = (
        "import json; from pathlib import Path; "
        "root = Path('.'); "
        "missing = [p for p in ["
        "'config/universe_history.json', "
        "'data/reference/entity_map.json', "
        "'data/reference/group_map.json', "
        "'data/reference/sector_map.json', "
        "'data/reference/trading_calendar.parquet'] if not (root / p).exists()]; "
        "print('Reference data missing (Week 3-5 work): ' + ', '.join(missing) if missing else 'Reference data: OK'); "
        "cov = 0.0; "
        "mp = root / 'results' / 'monitoring.json'; "
        "mon = json.loads(mp.read_text()) if mp.exists() else {}; "
        "psi = mon.get('feature_psi', {}) or {}; "
        "news_keys = [k for k in psi if k.startswith('news_')]; "
        "cov = sum(1 for k in news_keys if (psi.get(k) or 0) > 0) / max(len(news_keys), 1) if news_keys else 0.0; "
        "print(f'News feature coverage: {cov:.0%} (warns below 20%)' if news_keys else 'News feature coverage: no news_* features found'); "
    )
    try:
        subprocess.run([str(py), "-c", code], cwd=ROOT)
    except Exception:
        pass


if __name__ == "__main__":
    main()
