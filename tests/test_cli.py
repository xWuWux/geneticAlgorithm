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
