from __future__ import annotations

from random import Random
from typing import List

from .individual import Individual


def roulette_select(population: List[Individual], count: int, rng: Random) -> List[Individual]:
    """Fitness-proportionate ("roulette wheel") selection, with replacement.

    Returns brand-new ``Individual`` clones, never references into
    `population`. This is the fix for a real aliasing bug: the original
    roulette() appended the same list object into its result whenever an
    individual was selected more than once (expected behaviour for
    fitness-proportionate selection - nothing stops the same parent being
    picked twice). Downstream, mutate() edited a selected individual's genome
    list *in place*. If two selected slots pointed at the same object, editing
    one silently corrupted the other too, leaving it with a now-different
    genome but its old, stale score - a bug that's essentially invisible
    unless you trace object identity through roulette -> crossover -> mutate.
    Cloning here means every selected slot owns an independent genome list.
    """
    if not population:
        raise ValueError("cannot select from an empty population")
    if count < 0:
        raise ValueError("count must be >= 0")

    scores = [ind.score for ind in population]
    if any(s is None for s in scores):
        raise ValueError("all individuals must have a score before selection")

    total = sum(scores)
    selected: List[Individual] = []

    if total <= 0:
        # Fitness-proportionate selection is undefined when every score is
        # zero or negative (would divide by zero / produce nonsense
        # probabilities). Fall back to uniform sampling rather than crashing.
        for _ in range(count):
            selected.append(rng.choice(population).clone())
        return selected

    cumulative: List[float] = []
    running = 0.0
    for s in scores:
        running += s / total
        cumulative.append(running)

    for _ in range(count):
        r = rng.random()
        for individual, threshold in zip(population, cumulative):
            if r <= threshold:
                selected.append(individual.clone())
                break
        else:
            # Floating point rounding can leave the last cumulative bucket
            # just under 1.0; fall back to the last individual instead of
            # silently dropping a selection.
            selected.append(population[-1].clone())

    return selected
