# DNN_GA — Genetic algorithm search over DNN architectures

A genetic algorithm that searches over neural-network hidden-layer
architectures (number of layers and their widths), using classification
accuracy as fitness. Originally built for the thesis *"Optymalizacja sieci
neuronowej za pomocą algorytmów genetycznych"* (A. Bielewicz, W. Woźniak;
see `docs/`), with the GA architecture and original implementation
co-designed with [Francja](https://github.com/Francja), whose
[`geneticAlgorithm`](https://github.com/Francja/geneticAlgorithm) repo this
project is forked from (via [xWuWux/geneticAlgorithm](https://github.com/xWuWux/geneticAlgorithm)).

This version consolidates several divergent local copies of the code (the
fork's `geneticAlgorithm.py` + notebook, and a separately patched personal
notebook copy) into a single, tested implementation under `src/dnn_ga/`, and
fixes a number of correctness bugs found across those copies. The original,
unmodified copies are kept for provenance under `reference/original_fork/`.

## Layout

```
src/dnn_ga/          the package: GA core, operators, fitness functions, CLI
tests/                pytest suite
notebooks/            fixed notebook, now a thin driver over src/dnn_ga
geneticAlgorithm.py    backwards-compatible CLI shim (`python geneticAlgorithm.py`)
reference/original_fork/  unmodified copies of the original files, for provenance
docs/                  the thesis PDF
```

## Install & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"        # core + pytest
pip install -e ".[dnn]"         # + tensorflow/pandas/numpy, for real DNN training

pytest                          # run the test suite

python geneticAlgorithm.py --mode synthetic --population 20 --generations 10
python geneticAlgorithm.py --mode dnn --seed 0 --log-file run.log
```

`--mode synthetic` uses a fast placeholder fitness function (no TensorFlow or
network access needed) — good for trying the GA itself, CI, and the test
suite. `--mode dnn` trains a `tf.keras` classifier per individual on the iris
dataset, matching what the original notebook did.

Note: this sandbox runs Python 3.14, for which TensorFlow does not yet
publish wheels, so the two DNN-fitness tests in `tests/test_fitness.py`
`pytest.importorskip` and report as skipped here rather than failing — they
will run for real on a TensorFlow-supported interpreter (3.9–3.12 as of
writing).

## Bugs found and fixed

Found across the fork's `geneticAlgorithm.py`, the fork's notebook, and a
separately patched personal notebook copy — reconciled here into one fixed
implementation:

1. **Syntax error in the original notebook's last line**:
   `final_text(pop_bag, text_summary))` — wrong argument (`pop_bag` instead
   of the stop-condition string) and an unmatched `)`. The notebook could not
   run as-is. Fixed by `run_ga()` simply returning a proper `GAResult`
   instead of relying on a hand-written final print statement.

2. **Genome aliasing during selection** (`src/dnn_ga/selection.py`,
   `src/dnn_ga/operators.py`): `roulette()` handed out the *same* list object
   whenever an individual was selected more than once — expected for
   fitness-proportionate selection with replacement. `mutate()` then edited
   that list **in place**, silently corrupting every other slot holding the
   same reference: it ended up with a stale cached score paired with a
   genome that had actually changed underneath it. Invisible from the
   outside; only shows up by tracing object identity through
   `roulette → crossover → mutate`. Fixed by cloning on every selection and
   on every mutation path, so no two population slots ever share a genome
   list. Regression test:
   `tests/test_selection.py::test_roulette_select_returns_independent_clones_not_shared_references`.

3. **Best architecture never returned** — only a numeric score was ever
   printed (`gather_text`/`final_text`); recovering which network achieved
   it meant reading logs by eye. Fixed: `run_ga()` returns a `GAResult` whose
   `best_individual` carries both the genome and its score.

4. **No elitism** — a generation's best individual could be lost if it
   wasn't reselected by roulette or got mutated away. Fixed:
   `evolve()` clones the top `elitism_count` individuals through unchanged
   each generation (`src/dnn_ga/ga.py`).

5. **Mutation only ever tweaked one layer's width by ±1** — it never
   added/removed a layer, so the search could never change a network's
   depth via mutation (only crossover could, incidentally, by combining
   parents of different lengths). Fixed: `mutate_genome()` now also applies
   structural mutations (insert/delete a layer), gated by
   `structural_mutation_chance` and respecting `min_layers`/`max_layers`.

6. **`tf.estimator.DNNClassifier` is deprecated** in TensorFlow (confirmed,
   not assumed). Fixed: `make_keras_dnn_cost()` builds and trains an
   equivalent `tf.keras.Sequential` MLP instead.

7. **Stagnation-stop compared floats for exact equality**
   (`score_table[-N:].count(score_table[-1]) >= N`), which essentially never
   fires on noisy real accuracy values. Fixed: `has_stagnated()` compares the
   trailing window's spread (`max - min`) against a `stagnation_tolerance`.

8. **(found during this audit) Mutation wrapped around instead of
   clamping.** When a width mutation pushed a layer above `MAX_NEURONS`, the
   original code snapped it all the way down to `1` (and a value that
   underflowed below `1` snapped up to `MAX_NEURONS`) — so a "small" ±1
   mutation could occasionally jump a layer from e.g. 128 neurons to 1.
   Fixed: `_mutate_width()` clamps into `[1, max_neurons]` instead.

9. **(found during this audit) No input validation.** Both prior versions
   guarded `POPULATION % 4 == 0` and `MIN_LAYER >= 2` in the
   `if __name__ == "__main__"` condition; if that failed, the script just
   printed `"Please correct POPULATION or MIN_LAYER"` and silently did
   nothing (no exception, no non-zero exit code). Fixed: `GAConfig`
   validates all parameters eagerly in `__post_init__` and raises
   `ValueError` with a specific message per problem.

## What's genuinely new vs. the fork

- `elitism_count` and structural mutation (bugs 4 and 5 above) are new
  capabilities, not just bug fixes — the original design had neither.
- The personal notebook copy's per-generation file logging and `STEPS`
  variable are preserved, generalized into `src/dnn_ga/logging_utils.py`
  (configurable log path instead of a hardcoded Google Drive path, and
  works outside Colab).
- `run_ga`/`evolve`/`mutate_genome`/etc. all take an explicit
  `random.Random` instance rather than relying on the global `random` module
  state, which is what makes the test suite (and any future run)
  reproducible via a seed.
