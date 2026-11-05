from dnn_ga.individual import Individual
from dnn_ga.logging_utils import configure_logging, log_generation


def test_configure_logging_closes_previous_file_handler_instead_of_leaking_it(tmp_path):
    """Regression test: configure_logging() used to call handlers.clear()
    without closing the handlers first, leaking the previous run's open file
    descriptor whenever it was called more than once in the same process."""
    first_path = tmp_path / "first.log"
    second_path = tmp_path / "second.log"

    configure_logging(first_path)
    logger = configure_logging(first_path)
    first_handler = next(h for h in logger.handlers if hasattr(h, "baseFilename"))

    configure_logging(second_path)

    assert first_handler.stream is None  # closed, not just detached


def test_log_generation_runs_without_error(tmp_path):
    logger = configure_logging(tmp_path / "gen.log")
    population = [Individual(genome=[1, 2, 3], score=float(i)) for i in range(5)]
    best = population[-1]
    log_generation(logger, gen=0, population=population, best=best)
