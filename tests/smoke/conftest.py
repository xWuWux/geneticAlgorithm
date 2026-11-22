"""Register the ``smoke`` marker without touching the global pytest config."""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "smoke: installed-wheel smoke checks; run explicitly from "
        "tests/smoke/package_smoke.py",
    )
