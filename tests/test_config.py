import pytest

from dnn_ga.config import GAConfig


def test_default_config_is_valid():
    cfg = GAConfig()
    assert cfg.population > 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_layers": 1},
        {"max_layers": 1, "min_layers": 2},
        {"max_neurons": 0},
        {"population": 0},
        {"population": 10, "elitism_count": 10},
        {"population": 10, "elitism_count": 1},  # 10-1=9, not divisible by 4
        {"generations": 0},
        {"mutation_chance": 1.5},
        {"structural_mutation_chance": -0.1},
        {"no_evolve_generations": 0},
        {"stagnation_tolerance": -1},
    ],
)
def test_invalid_config_raises(overrides):
    with pytest.raises(ValueError):
        GAConfig(**overrides)


def test_population_minus_elitism_must_be_divisible_by_four():
    with pytest.raises(ValueError):
        GAConfig(population=10, elitism_count=1)  # 10-1=9, not divisible by 4

    # Should not raise:
    GAConfig(population=12, elitism_count=0)
