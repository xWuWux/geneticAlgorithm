import pytest

from dnn_ga.fitness import synthetic_cost


def test_synthetic_cost_matches_expected_formula():
    genome = [2, 3, 5, 7]
    # first*second/second_last + last*8 = 2*3/5 + 7*8 = 1.2 + 56 = 57.2
    assert synthetic_cost(genome) == pytest.approx(57.2)


def test_synthetic_cost_rejects_too_short_genome():
    with pytest.raises(ValueError):
        synthetic_cost([5])


def test_synthetic_cost_rejects_zero_width_layer_instead_of_raising_zerodivisionerror():
    with pytest.raises(ValueError):
        synthetic_cost([1, 2, 0, 4])


def test_keras_dnn_cost_trains_and_scores_a_tiny_model():
    np = pytest.importorskip("numpy")
    pytest.importorskip("tensorflow")
    from dnn_ga.fitness import make_keras_dnn_cost

    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(20, 4)).astype("float32")
    train_y = rng.integers(0, 3, size=20)
    test_x = rng.normal(size=(10, 4)).astype("float32")
    test_y = rng.integers(0, 3, size=10)

    cost_fn = make_keras_dnn_cost(train_x, train_y, test_x, test_y, epochs=1, batch_size=8, verbose=0)
    score = cost_fn([4, 3])

    assert 0.0 <= score <= 1.0


def test_keras_dnn_cost_validates_genome():
    np = pytest.importorskip("numpy")
    pytest.importorskip("tensorflow")
    from dnn_ga.fitness import make_keras_dnn_cost

    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(10, 4)).astype("float32")
    train_y = rng.integers(0, 3, size=10)
    cost_fn = make_keras_dnn_cost(train_x, train_y, train_x, train_y, epochs=1, verbose=0)

    with pytest.raises(ValueError):
        cost_fn([])
    with pytest.raises(ValueError):
        cost_fn([4, 0])
