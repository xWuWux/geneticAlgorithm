import dnn_ga.cli as cli_module
from dnn_ga.cli import main


def test_cli_synthetic_mode_runs_and_logs_best_architecture(tmp_path, capsys):
    log_file = tmp_path / "run.log"
    exit_code = main(
        [
            "--mode",
            "synthetic",
            "--population",
            "12",
            "--generations",
            "3",
            "--seed",
            "1",
            "--log-file",
            str(log_file),
        ]
    )

    assert exit_code == 0
    assert log_file.exists()
    contents = log_file.read_text()
    assert "Best architecture" in contents


def test_cli_dnn_mode_accepts_population_override_without_crashing(monkeypatch, tmp_path):
    """Regression test: --mode dnn merged the DNN_PRESET's population/generations
    kwargs with the CLI's own overrides positionally, so a --population flag
    raised "got multiple values for keyword argument 'population'". Stub out
    the TensorFlow-dependent pieces so this can run without tensorflow/network."""

    def fake_load_iris_dataset():
        return object(), object(), object(), object()

    def fake_make_keras_dnn_cost(*_args, **_kwargs):
        def cost(genome):
            return 0.5

        return cost

    monkeypatch.setattr(cli_module, "load_iris_dataset", fake_load_iris_dataset)
    monkeypatch.setattr(cli_module, "make_keras_dnn_cost", fake_make_keras_dnn_cost)

    log_file = tmp_path / "dnn_run.log"
    exit_code = main(
        [
            "--mode",
            "dnn",
            "--population",
            "96",  # overrides DNN_PRESET's population=100
            "--generations",
            "2",
            "--seed",
            "0",
            "--log-file",
            str(log_file),
        ]
    )

    assert exit_code == 0
    assert "Best architecture" in log_file.read_text()
