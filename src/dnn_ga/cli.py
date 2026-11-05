from __future__ import annotations

import argparse
from random import Random
from typing import List, Optional

from .config import GAConfig
from .fitness import load_iris_dataset, make_keras_dnn_cost, synthetic_cost
from .ga import run_ga
from .logging_utils import configure_logging, log_generation
from .presets import DNN_PRESET


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genetic algorithm search over DNN hidden-layer architectures."
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "dnn"],
        default="synthetic",
        help=(
            "synthetic: fast placeholder fitness, no TensorFlow/network needed. "
            "dnn: train a tf.keras classifier per individual on the iris dataset "
            "(requires tensorflow, pandas, and network access)."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--log-file", type=str, default=None, help="optional path to also write logs to a file")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--target-score", type=float, default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logger = configure_logging(args.log_file)

    cfg_overrides = {}
    if args.population is not None:
        cfg_overrides["population"] = args.population
    if args.generations is not None:
        cfg_overrides["generations"] = args.generations
    if args.target_score is not None:
        cfg_overrides["target_score"] = args.target_score

    if args.mode == "dnn":
        # Merge as a dict, not as separate keyword args: passing the preset
        # values as explicit kwargs *and* spreading cfg_overrides after them
        # raised "got multiple values for keyword argument" whenever a CLI
        # flag (e.g. --population) overrode a key the preset also set.
        cfg = GAConfig(**{**DNN_PRESET, **cfg_overrides})
        train_x, train_y, test_x, test_y = load_iris_dataset()
        cost_fn = make_keras_dnn_cost(train_x, train_y, test_x, test_y)
    else:
        cfg = GAConfig(**cfg_overrides)
        cost_fn = synthetic_cost

    rng = Random(args.seed)

    def on_generation(gen, population, best):
        log_generation(logger, gen, population, best)

    result = run_ga(cfg, cost_fn, rng=rng, on_generation=on_generation)

    logger.info("Stopped after %d generation(s): %s", result.generations_run, result.stop_reason)
    logger.info(
        "Best architecture: %s (score=%.4f)", result.best_individual.genome, result.best_individual.score
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
