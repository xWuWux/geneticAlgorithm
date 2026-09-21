from __future__ import annotations

from random import Random
from typing import List, Tuple

from .config import GAConfig
from .individual import Genome, Individual


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def cross_parents(parent1: Genome, parent2: Genome, cross_point: int) -> Genome:
    return list(parent1[:cross_point]) + list(parent2[cross_point:])


def crossover_pair(parent1: Genome, parent2: Genome, rng: Random) -> Tuple[Genome, Genome]:
    max_cross_point = min(len(parent1), len(parent2)) - 1
    if max_cross_point < 1:
        # Both parents are single-layer (or one is): there's no interior cut
        # point that wouldn't produce an empty genome, so just swap them
        # whole rather than letting randint(1, 0) raise.
        return list(parent2), list(parent1)
    cross_point = rng.randint(1, max_cross_point)
    return (
        cross_parents(parent1, parent2, cross_point),
        cross_parents(parent2, parent1, cross_point),
    )


def crossover_population(genomes: List[Genome], rng: Random) -> List[Genome]:
    if len(genomes) % 2 != 0:
        raise ValueError("crossover_population requires an even number of parent genomes")

    order = list(range(len(genomes)))
    rng.shuffle(order)

    children: List[Genome] = []
    for i in range(0, len(order), 2):
        p1 = genomes[order[i]]
        p2 = genomes[order[i + 1]]
        c1, c2 = crossover_pair(p1, p2, rng)
        children.append(c1)
        children.append(c2)
    return children


def _mutate_width(genome: Genome, cfg: GAConfig, rng: Random) -> Genome:
    genome = list(genome)
    idx = rng.randrange(len(genome))
    delta = rng.choice((-1, 1))
    # Clamp instead of wrap around. The original code wrapped a value that
    # overflowed MAX_NEURONS down to 1, and a value that underflowed to 0 up
    # to MAX_NEURONS, i.e. a "mutation" at the boundary could jump a layer
    # from e.g. 128 neurons straight to 1. Clamping keeps mutation a small,
    # local perturbation, which is the point of a +/-1 mutation operator.
    genome[idx] = _clamp(genome[idx] + delta, 1, cfg.max_neurons)
    return genome


def _mutate_structure(genome: Genome, cfg: GAConfig, rng: Random) -> Genome:
    """Add or remove a layer. This is the capability the original mutate()
    never had: it only ever tweaked one layer's width."""
    genome = list(genome)
    can_grow = len(genome) < cfg.max_layers
    can_shrink = len(genome) > cfg.min_layers

    if can_grow and (not can_shrink or rng.random() < 0.5):
        idx = rng.randint(0, len(genome))
        genome.insert(idx, rng.randint(1, cfg.max_neurons))
    elif can_shrink:
        idx = rng.randrange(len(genome))
        del genome[idx]
    # else: genome is pinned at a single allowed length (min_layers ==
    # max_layers); leave it as-is rather than violating the bounds.
    return genome


def mutate_genome(genome: Genome, cfg: GAConfig, rng: Random) -> Genome:
    if rng.random() < cfg.structural_mutation_chance:
        return _mutate_structure(genome, cfg, rng)
    return _mutate_width(genome, cfg, rng)


def maybe_mutate(individual: Individual, cfg: GAConfig, rng: Random) -> Individual:
    """Return a mutated-or-not clone of `individual`.

    Always returns a fresh Individual with its own genome list so genomes are
    never aliased between population slots. In the original code, roulette()
    could hand out the same list object to more than one slot (expected for
    fitness-proportionate selection with replacement); mutate() then edited
    that list in place, silently corrupting every other slot holding the same
    reference and leaving it with a stale score paired with a genome that had
    actually changed underneath it. Cloning on every path here removes the
    possibility of that aliasing entirely, independent of how selection
    behaves upstream (see selection.roulette_select for the matching fix on
    the selection side).

    An individual that is *not* mutated keeps its cached score, so the caller
    doesn't need to re-run an expensive fitness evaluation for it. A mutated
    individual has its score cleared to signal that it must be re-evaluated.
    """
    if rng.random() > cfg.mutation_chance:
        return individual.clone()
    new_genome = mutate_genome(individual.genome, cfg, rng)
    return Individual(genome=new_genome, score=None)
