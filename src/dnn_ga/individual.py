from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import List, Optional

from .config import GAConfig

# A genome is the list of hidden-layer widths, e.g. [12, 64, 8] means a
# 3-hidden-layer network with those widths.
Genome = List[int]


@dataclass
class Individual:
    genome: Genome
    score: Optional[float] = None

    def clone(self) -> "Individual":
        """Deep-enough copy: a new genome list, so mutating the clone can
        never affect the original (see the roulette-selection aliasing bug
        described in ga.py / selection.py)."""
        return Individual(list(self.genome), self.score)


def generate_genome(cfg: GAConfig, rng: Random) -> Genome:
    n_layers = rng.randint(cfg.min_layers, cfg.max_layers)
    return [rng.randint(1, cfg.max_neurons) for _ in range(n_layers)]


def initialize_population(cfg: GAConfig, rng: Random) -> List[Individual]:
    return [Individual(generate_genome(cfg, rng)) for _ in range(cfg.population)]
