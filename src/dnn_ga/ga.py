from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Callable, List, Optional

from .config import GAConfig
from .individual import Individual, initialize_population
from .operators import crossover_population, maybe_mutate
from .selection import roulette_select

CostFn = Callable[[List[int]], float]
OnGenerationFn = Callable[[int, List[Individual], Individual], None]


@dataclass
class GAResult:
    """What a run produced.

    Crucially this carries the winning genome, not just its score: the
    original scripts only ever printed a numeric best score via
    ``gather_text``/``final_text`` and never returned the architecture that
    achieved it, so recovering "which network got that score" meant reading
    the printed log by eye and cross-referencing generation dumps.
    """

    best_individual: Individual
    stop_reason: str  # "target_reached" | "stagnation" | "max_generations"
    generations_run: int
    history: List[float] = field(default_factory=list)


def evaluate_population(population: List[Individual], cost_fn: CostFn) -> None:
    """Fill in missing scores in place. Individuals that already have a score
    (unmutated survivors) are skipped, so an expensive cost_fn - e.g. training
    a network - is only ever paid for genomes that actually changed."""
    for ind in population:
        if ind.score is None:
            ind.score = cost_fn(ind.genome)


def has_stagnated(history: List[float], cfg: GAConfig) -> bool:
    """True if the best score hasn't meaningfully moved over the trailing
    `no_evolve_generations` window.

    The original check compared floats for exact equality
    (``score_table[-N:].count(score_table[-1]) >= N``), which essentially
    never fires on a real, noisy fitness signal (e.g. model accuracy) since
    two independently computed floats are almost never bit-for-bit equal.
    Comparing the window's spread against a tolerance is the fix.
    """
    window = history[-cfg.no_evolve_generations:]
    if len(window) < cfg.no_evolve_generations:
        return False
    return (max(window) - min(window)) <= cfg.stagnation_tolerance


def evolve(population: List[Individual], cfg: GAConfig, rng: Random) -> List[Individual]:
    """Produce the next generation.

    Elitism: the top `elitism_count` individuals are cloned through
    unchanged, so the best individual found so far can never be lost to
    selection variance or a bad mutation roll (the original had no elitism at
    all - a generation's best individual survived only if it happened to be
    reselected by roulette and then happened not to be mutated).

    The rest of the population follows the original scheme: select
    `(population - elitism_count) / 2` parents via roulette, cross them in
    pairs into an equal number of children, and keep both the parents *and*
    the children (a simple mu+lambda-style scheme), then apply mutation to
    that whole combined set.
    """
    if len(population) != cfg.population:
        raise ValueError(
            f"evolve() expected a population of size {cfg.population} (per cfg), "
            f"got {len(population)}"
        )

    ranked = sorted(population, key=lambda ind: ind.score, reverse=True)
    elites = [ind.clone() for ind in ranked[: cfg.elitism_count]]

    n_parents = (len(population) - cfg.elitism_count) // 2
    parents = roulette_select(ranked, n_parents, rng)
    children_genomes = crossover_population([p.genome for p in parents], rng)
    children = [Individual(genome=g) for g in children_genomes]

    combined = parents + children
    mutated = [maybe_mutate(ind, cfg, rng) for ind in combined]
    return elites + mutated


def run_ga(
    cfg: GAConfig,
    cost_fn: CostFn,
    rng: Optional[Random] = None,
    on_generation: Optional[OnGenerationFn] = None,
) -> GAResult:
    """Run the genetic search to completion and return the result.

    `cost_fn` maps a genome (list of hidden-layer widths) to a fitness score;
    higher is better. Pass `dnn_ga.fitness.synthetic_cost` for a fast
    placeholder, or a function built by `dnn_ga.fitness.make_keras_dnn_cost`
    to actually train a network per individual.
    """
    rng = rng or Random()
    population = initialize_population(cfg, rng)
    evaluate_population(population, cost_fn)

    best = max(population, key=lambda ind: ind.score).clone()
    history = [best.score]
    stop_reason = "max_generations"
    generations_run = cfg.generations

    for gen in range(cfg.generations):
        if on_generation is not None:
            on_generation(gen, population, best)

        if best.score >= cfg.target_score:
            stop_reason = "target_reached"
            generations_run = gen
            break
        if has_stagnated(history, cfg):
            stop_reason = "stagnation"
            generations_run = gen
            break

        population = evolve(population, cfg, rng)
        evaluate_population(population, cost_fn)

        gen_best = max(population, key=lambda ind: ind.score)
        if gen_best.score > best.score:
            best = gen_best.clone()
        history.append(gen_best.score)

    return GAResult(
        best_individual=best,
        stop_reason=stop_reason,
        generations_run=generations_run,
        history=history,
    )
