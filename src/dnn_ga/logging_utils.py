from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .individual import Individual

_LOGGER_NAME = "dnn_ga"


def configure_logging(
    log_path: Optional[Union[str, Path]] = None, level: int = logging.INFO
) -> logging.Logger:
    """Set up console (+ optional file) logging for a GA run.

    Replaces the notebook's manual ``open(filename, 'a')`` calls hardcoded to
    a Google Drive path - that only worked inside Colab with Drive mounted,
    and any I/O error there simply crashed the run. `log_path` here is
    caller-supplied and optional; the run always logs to the console even if
    no file path is given.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_path is not None:
        path = Path(log_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not open log file %s (%s); logging to console only.", path, exc)
        else:
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def _group_by_genome(population: List[Individual]) -> Dict[Tuple[int, ...], List[float]]:
    by_genome: Dict[Tuple[int, ...], List[float]] = {}
    for ind in population:
        by_genome.setdefault(tuple(ind.genome), []).append(ind.score)
    return by_genome


def log_generation(
    logger: logging.Logger, gen: int, population: List[Individual], best: Individual, top_n: int = 5
) -> None:
    """Log a generation summary: best score so far, the top individuals, and
    per-architecture averages (an architecture can appear more than once in
    a generation via roulette selection with replacement)."""
    ranked = sorted(population, key=lambda ind: ind.score, reverse=True)
    logger.info("Generation %d - best score so far: %.4f (genome=%s)", gen + 1, best.score, best.genome)

    for rank, ind in enumerate(ranked[:top_n], start=1):
        logger.info("  #%d: %s -> score %.4f", rank, ind.genome, ind.score)

    for genome, scores in _group_by_genome(population).items():
        avg = sum(scores) / len(scores)
        logger.debug(
            "  architecture %s seen %d time(s), avg score %.4f, best %.4f",
            genome,
            len(scores),
            avg,
            max(scores),
        )
