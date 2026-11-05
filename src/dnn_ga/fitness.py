from __future__ import annotations

from typing import Callable, List, Sequence, Tuple


def synthetic_cost(genome: Sequence[int]) -> float:
    """Fast, deterministic placeholder fitness used for tests, demos, and any
    run where training a real network per individual would be far too slow.

    Mirrors the toy formula from the original geneticAlgorithm.py so existing
    intuition about the search space still applies; unlike the original, it
    validates its input instead of raising an opaque IndexError/ZeroDivisionError.
    """
    if len(genome) < 2:
        raise ValueError(f"genome must have at least 2 layers to score, got {genome!r}")
    first, second, second_last, last = genome[0], genome[1], genome[-2], genome[-1]
    if second_last == 0:
        raise ValueError(f"layer width of 0 is invalid (would divide by zero): {genome!r}")
    return first * second / second_last + last * 8


CSV_COLUMN_NAMES = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"


def load_iris_dataset() -> Tuple["object", "object", "object", "object"]:
    """Download (and cache) the iris train/test split used by the notebook.

    Requires network access and the optional `pandas`/`tensorflow`
    dependencies; imported lazily so importing `dnn_ga` doesn't require them
    just to run the GA with `synthetic_cost`.
    """
    import pandas as pd
    import tensorflow as tf

    try:
        train_path = tf.keras.utils.get_file("iris_training.csv", TRAIN_URL)
        test_path = tf.keras.utils.get_file("iris_test.csv", TEST_URL)
    except Exception as exc:  # noqa: BLE001 - re-raise with actionable context
        raise RuntimeError(
            "Failed to download the iris dataset. Check network access, or "
            "pre-populate the tf.keras cache directory manually."
        ) from exc

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    train_y = train.pop("Species")
    test_y = test.pop("Species")
    return train, train_y, test, test_y


def make_keras_dnn_cost(
    train_features,
    train_labels,
    test_features,
    test_labels,
    n_classes: int = 3,
    epochs: int = 10,
    batch_size: int = 32,
    verbose: int = 0,
) -> Callable[[List[int]], float]:
    """Build a fitness function that trains a small `tf.keras` MLP per genome
    and returns its held-out accuracy.

    This replaces `tf.estimator.DNNClassifier`, which the notebook's `DNN()`
    was built on: `tf.estimator` is a deprecated part of the TensorFlow API
    (deprecated and slated for removal outside `compat.v1`), whereas
    `tf.keras` is the actively supported way to build/train/evaluate this
    kind of classifier.
    """
    import tensorflow as tf

    n_features = train_features.shape[1]

    def cost(genome: Sequence[int]) -> float:
        if not genome:
            raise ValueError("genome must contain at least one layer")
        if any(width < 1 for width in genome):
            raise ValueError(f"all layer widths must be >= 1, got {list(genome)!r}")

        model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(n_features,))]
            + [tf.keras.layers.Dense(units, activation="relu") for units in genome]
            + [tf.keras.layers.Dense(n_classes, activation="softmax")]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_features,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        _, accuracy = model.evaluate(test_features, test_labels, verbose=verbose)
        return float(accuracy)

    return cost
