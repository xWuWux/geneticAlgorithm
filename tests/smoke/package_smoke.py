"""Smoke tests for the *installed* wheel and console entry point.

Run explicitly against a built wheel, e.g.:

    python -m pip install --no-index --find-links dist --no-deps "dnnga[sklearn]"
    python -m pytest tests/smoke/package_smoke.py -q

The file is named ``package_smoke.py`` (not ``test_*.py``/``*_test.py``) so the
default ``pytest`` run does not collect it; CI runs it in the ``package-smoke``
job on Linux and Windows. These checks cover what the unit suite cannot see:
the wheel's metadata, the generated console script, ``python -m dnnga``, and
``pip check``.
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def _run(cmd, *, timeout=300):
    result = subprocess.run(
        [str(part) for part in cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(str(p) for p in cmd)}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result


def _console_script() -> Path:
    scripts = Path(sysconfig.get_path("scripts"))
    script = scripts / ("dnnga.exe" if os.name == "nt" else "dnnga")
    assert script.exists(), f"console script missing: {script}"
    return script


def test_import_metadata_and_extras():
    import dnnga

    version = importlib.metadata.version("dnnga")
    assert dnnga.__version__ == version
    assert version.count(".") >= 2

    requires = importlib.metadata.requires("dnnga") or []
    assert any(req.startswith("numpy") for req in requires), requires
    assert any("scikit-learn" in req and "sklearn" in req for req in requires), requires


def test_console_script_entry_point():
    script = _console_script()
    entry_points = importlib.metadata.entry_points(group="console_scripts")
    matches = [ep for ep in entry_points if ep.name == "dnnga"]
    assert matches, "dnnga console_scripts entry point is missing"
    assert matches[0].value == "dnnga.cli:main", matches[0].value

    result = _run([script, "--version"])
    assert result.stdout.strip() == f"dnnga {importlib.metadata.version('dnnga')}"


def test_cli_and_module_entry_points():
    script = _console_script()
    args = [
        "--mode",
        "synthetic",
        "--population",
        "2",
        "--tournament-size",
        "2",
        "--elitism-count",
        "0",
        "--generations",
        "1",
        "--seed",
        "0",
    ]
    _run([script, *args])
    _run([sys.executable, "-m", "dnnga", *args])


def test_core_import_is_lazy():
    """Importing dnnga must not pull in optional backends."""
    _run(
        [
            sys.executable,
            "-c",
            (
                "import sys, dnnga\n"
                "assert 'sklearn' not in sys.modules, 'sklearn imported eagerly'\n"
                "assert 'tensorflow' not in sys.modules, 'tensorflow imported eagerly'\n"
                "print('lazy import ok', dnnga.__version__)\n"
            ),
        ]
    )


def test_pip_check():
    result = _run([sys.executable, "-m", "pip", "check"])
    assert "No broken requirements found" in result.stdout, result.stdout
