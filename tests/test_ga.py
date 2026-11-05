from random import Random

import pytest

from dnn_ga.config import GAConfig
from dnn_ga.fitness import synthetic_cost
from dnn_ga.ga import evolve, has_stagnated, run_ga
from dnn_ga.individual import Individual


def test_has_stagnated_true_for_near_identical_noisy_floats():
    cfg = GAConfig(no_evolve_generations=5, stagnation_tolerance=1e-6)
    # These floats are never bit-for-bit equal (unlike the exact-equality
    # check the original code used, which would never fire on values like
    # these), but they're stagnant for any reasonable tolerance.
    history = [0.9000000001, 0.9000000002, 0.9000000000, 0.9000000003, 0.9000000001]
    assert has_stagnated(history, cfg) is True


def test_has_stagnated_false_when_still_improving():
    cfg = GAConfig(no_evolve_generations=5, stagnation_tolerance=1e-6)
    history = [0.5, 0.6, 0.7, 0.8, 0.9]
    assert has_stagnated(history, cfg) is False


def test_has_stagnated_false_before_window_is_full():
    cfg = GAConfig(no_evolve_generations=10)
    assert has_stagnated([1.0, 1.0, 1.0], cfg) is False


def test_evolve_preserves_best_individual_via_elitism():
    cfg = GAConfig(population=12, elitism_count=4, mutation_chance=1.0, structural_mutation_chance=1.0)
    rng = Random(0)

    champion = Individual(genome=[7, 7, 7, 7], score=10_000.0)
    rest = [Individual(genome=[1, 1], score=0.01) for _ in range(cfg.population - 1)]
    population = [champion] + rest

    next_gen = evolve(population, cfg, rng)

    assert len(next_gen) == cfg.population
    assert any(ind.genome == [7, 7, 7, 7] and ind.score == 10_000.0 for ind in next_gen)


def test_evolve_rejects_a_population_whose_length_does_not_match_cfg():
    """Regression test: evolve() used to derive n_parents from cfg.population
    instead of len(population), silently assuming the two always match. A
    caller that passed a population of a different size (e.g. a
    resumed/trimmed run) would get a silently wrong number of children
    instead of a clear error."""
    cfg = GAConfig(population=12, elitism_count=4)
    mismatched = [Individual(genome=[1, 1], score=1.0) for _ in range(11)]

    with pytest.raises(ValueError):
        evolve(mismatched, cfg, Random(0))


def test_run_ga_returns_the_best_genome_not_just_a_score():
    cfg = GAConfig(
        min_layers=2, max_layers=4, max_neurons=20, population=12, generations=5, elitism_count=4
    )
    rng = Random(0)
    result = run_ga(cfg, synthetic_cost, rng=rng)

    assert isinstance(result.best_individual.genome, list)
    assert result.best_individual.score == max(result.history)
    assert result.best_individual.score is not None


def test_run_ga_stops_on_target_reached():
    cfg = GAConfig(
        min_layers=2, max_layers=4, max_neurons=20, population=12, generations=50,
        target_score=0.0, elitism_count=4,
    )
    result = run_ga(cfg, synthetic_cost, rng=Random(1))
    assert result.stop_reason == "target_reached"
    assert result.generations_run <= cfg.generations


def test_run_ga_is_reproducible_given_the_same_seed():
    cfg = GAConfig(
        min_layers=2, max_layers=4, max_neurons=20, population=12, generations=8, elitism_count=4
    )
    result_a = run_ga(cfg, synthetic_cost, rng=Random(99))
    result_b = run_ga(cfg, synthetic_cost, rng=Random(99))
    assert result_a.best_individual.genome == result_b.best_individual.genome
    assert result_a.history == result_b.history
