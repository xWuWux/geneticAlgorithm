"""Backwards-compatible entry point.

Historically this file *was* the genetic algorithm implementation. All of
that logic now lives in the tested `dnn_ga` package under `src/`; this is a
thin shim so `python geneticAlgorithm.py` still works for anyone used to the
old layout. See README.md for the list of bugs that made the old version
unsafe to keep as-is (aliased genomes, a divide-by-zero-adjacent wraparound
mutation, no elitism, float-equality stagnation check, ...).
"""
from dnn_ga.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
