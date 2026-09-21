from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GAConfig:
    """Tunable parameters for the genetic search.

    Validated eagerly in ``__post_init__`` so a bad configuration fails loudly
    at construction time instead of silently doing nothing (the original
    scripts guarded ``population % 4 == 0`` in the module-level
    ``if __name__ == "__main__"`` condition and, if it failed, just printed
    "Please correct POPULATION or MIN_LAYER" and exited without running or
    raising).
    """

    min_layers: int = 2
    max_layers: int = 12
    max_neurons: int = 128
    population: int = 80
    generations: int = 10
    target_score: float = 1.0
    # Chance, per individual per generation, that a mutation is applied at all.
    mutation_chance: float = 0.05
    # Given that a mutation happens, the chance it changes the *shape* of the
    # network (add/remove a layer) rather than just nudging one layer's width.
    structural_mutation_chance: float = 0.3
    no_evolve_generations: int = 25
    # Max spread allowed within the trailing `no_evolve_generations` window of
    # best-scores before we call it stagnation. See `ga.has_stagnated`.
    stagnation_tolerance: float = 1e-6
    elitism_count: int = 4

    def __post_init__(self) -> None:
        errors: list[str] = []

        if self.min_layers < 2:
            errors.append("min_layers must be >= 2")
        if self.max_layers < self.min_layers:
            errors.append("max_layers must be >= min_layers")
        if self.max_neurons < 1:
            errors.append("max_neurons must be >= 1")
        if self.population <= 0:
            errors.append("population must be a positive integer")
        if self.generations <= 0:
            errors.append("generations must be a positive integer")
        if not 0.0 <= self.mutation_chance <= 1.0:
            errors.append("mutation_chance must be within [0, 1]")
        if not 0.0 <= self.structural_mutation_chance <= 1.0:
            errors.append("structural_mutation_chance must be within [0, 1]")
        if self.no_evolve_generations <= 0:
            errors.append("no_evolve_generations must be a positive integer")
        if self.stagnation_tolerance < 0:
            errors.append("stagnation_tolerance must be >= 0")
        if self.elitism_count < 0 or self.elitism_count >= self.population:
            errors.append("elitism_count must be in [0, population)")
        elif (self.population - self.elitism_count) % 4 != 0:
            errors.append(
                "population - elitism_count must be divisible by 4 "
                "(parents are selected in pairs, and each pair produces 2 "
                "children alongside the 2 surviving parents)"
            )

        if errors:
            raise ValueError("Invalid GAConfig: " + "; ".join(errors))
