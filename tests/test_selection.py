from random import Random

import pytest

from dnn_ga.individual import Individual
from dnn_ga.selection import roulette_select


def test_roulette_select_requires_scored_population():
    with pytest.raises(ValueError):
        roulette_select([Individual(genome=[1, 2])], count=1, rng=Random(0))


def test_roulette_select_rejects_empty_population():
    with pytest.raises(ValueError):
        roulette_select([], count=1, rng=Random(0))


def test_roulette_select_falls_back_to_uniform_when_all_scores_nonpositive():
    population = [Individual(genome=[1, 2], score=0.0) for _ in range(3)]
    selected = roulette_select(population, count=5, rng=Random(0))
    assert len(selected) == 5


def test_roulette_select_returns_independent_clones_not_shared_references():
    """Regression test for the aliasing bug: the original roulette()
    appended the *same* list object into its result every time an individual
    was selected more than once. A later in-place mutation of one slot then
    silently corrupted every other slot holding that same object, leaving it
    with a stale score paired with a genome that had actually changed.

    With only one individual in the population, every single selection below
    is forced to pick it - so if selection were still handing out shared
    references, all 5 results would be the same object / share the same
    genome list.
    """
    rng = Random(1)
    source = Individual(genome=[5, 5, 5], score=1.0)
    population = [source]

    selected = roulette_select(population, count=5, rng=rng)

    assert len(selected) == 5
    assert len({id(ind) for ind in selected}) == 5  # distinct Individual objects
    assert len({id(ind.genome) for ind in selected}) == 5  # distinct genome lists

    # Mutating one selected individual's genome must not affect the others,
    # nor the original individual still sitting in `population`.
    selected[0].genome[0] = -1
    assert all(ind.genome[0] == 5 for ind in selected[1:])
    assert source.genome == [5, 5, 5]


def test_roulette_select_is_proportional_to_score():
    rng = Random(123)
    population = [
        Individual(genome=[1], score=1.0),
        Individual(genome=[2], score=99.0),
    ]
    selected = roulette_select(population, count=2000, rng=rng)
    high_score_picks = sum(1 for ind in selected if ind.genome == [2])
    # The heavily-weighted individual should dominate selections.
    assert high_score_picks > 1800
