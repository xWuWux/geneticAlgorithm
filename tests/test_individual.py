from random import Random

from dnn_ga.config import GAConfig
from dnn_ga.individual import Individual, generate_genome, initialize_population


def test_generate_genome_respects_bounds():
    cfg = GAConfig(min_layers=3, max_layers=5, max_neurons=10, population=8)
    rng = Random(0)
    for _ in range(200):
        genome = generate_genome(cfg, rng)
        assert cfg.min_layers <= len(genome) <= cfg.max_layers
        assert all(1 <= w <= cfg.max_neurons for w in genome)


def test_initialize_population_size():
    cfg = GAConfig(population=12)
    rng = Random(1)
    pop = initialize_population(cfg, rng)
    assert len(pop) == cfg.population
    assert all(ind.score is None for ind in pop)


def test_clone_is_independent():
    original = Individual(genome=[1, 2, 3], score=4.0)
    clone = original.clone()

    assert clone.genome == original.genome
    assert clone is not original
    assert clone.genome is not original.genome

    clone.genome[0] = 999
    assert original.genome == [1, 2, 3]
