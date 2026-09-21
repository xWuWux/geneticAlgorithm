from random import Random

import pytest

from dnn_ga.config import GAConfig
from dnn_ga.individual import Individual
from dnn_ga.operators import (
    cross_parents,
    crossover_pair,
    crossover_population,
    maybe_mutate,
    mutate_genome,
)


def test_cross_parents_basic():
    child = cross_parents([1, 2, 3, 4], [10, 20, 30, 40], cross_point=2)
    assert child == [1, 2, 30, 40]


def test_crossover_pair_never_produces_empty_genome_for_single_layer_parents():
    rng = Random(0)
    c1, c2 = crossover_pair([5], [9], rng)
    assert c1 and c2


def test_crossover_population_requires_even_count():
    with pytest.raises(ValueError):
        crossover_population([[1, 2], [3, 4], [5, 6]], Random(0))


def test_crossover_population_preserves_count_and_does_not_mutate_inputs():
    genomes = [[1, 2, 3], [4, 5, 6], [7, 8], [9, 10, 11, 12]]
    snapshot = [list(g) for g in genomes]
    rng = Random(42)
    children = crossover_population(genomes, rng)
    assert len(children) == len(genomes)
    assert genomes == snapshot  # inputs untouched


def test_mutate_genome_width_stays_within_bounds():
    cfg = GAConfig(min_layers=2, max_layers=6, max_neurons=5, structural_mutation_chance=0.0)
    rng = Random(3)
    genome = [1, 5, 1, 5]  # boundary values, likely to trigger clamping
    for _ in range(500):
        genome = mutate_genome(genome, cfg, rng)
        assert all(1 <= w <= cfg.max_neurons for w in genome)


def test_mutate_genome_width_clamps_instead_of_wrapping():
    # Old behaviour wrapped a value that overflowed max_neurons down to 1
    # (and one that underflowed below 1 up to max_neurons). We want clamping.
    cfg = GAConfig(min_layers=2, max_layers=6, max_neurons=5, structural_mutation_chance=0.0)

    class ForcedUpRng(Random):
        def randrange(self, *a, **k):
            return 0

        def choice(self, seq):
            return 1  # always +1

    genome = mutate_genome([5, 2], cfg, ForcedUpRng())
    assert genome[0] == 5  # clamped at max, not wrapped to 1

    class ForcedDownRng(Random):
        def randrange(self, *a, **k):
            return 0

        def choice(self, seq):
            return -1  # always -1

    genome = mutate_genome([1, 2], cfg, ForcedDownRng())
    assert genome[0] == 1  # clamped at min, not wrapped to max_neurons


def test_mutate_genome_structural_respects_min_max_layers():
    cfg = GAConfig(min_layers=2, max_layers=4, max_neurons=10, structural_mutation_chance=1.0)
    rng = Random(7)

    genome = [1, 1]  # at min_layers: structural mutation must only grow
    for _ in range(100):
        genome = mutate_genome(genome, cfg, rng)
        assert cfg.min_layers <= len(genome) <= cfg.max_layers


def test_maybe_mutate_never_aliases_the_original_genome():
    cfg = GAConfig(mutation_chance=1.0, structural_mutation_chance=0.0)
    rng = Random(5)
    original = Individual(genome=[1, 2, 3, 4], score=1.0)

    mutated = maybe_mutate(original, cfg, rng)

    assert mutated.genome is not original.genome
    assert mutated.score is None  # cache invalidated since it actually changed


def test_maybe_mutate_keeps_cached_score_when_unchanged():
    cfg = GAConfig(mutation_chance=0.0)
    rng = Random(5)
    original = Individual(genome=[1, 2, 3, 4], score=9.0)

    unchanged = maybe_mutate(original, cfg, rng)

    assert unchanged.genome == original.genome
    assert unchanged.genome is not original.genome
    assert unchanged.score == 9.0
