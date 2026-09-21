"""Shared hyperparameter presets.

Single source of truth for the `--mode dnn` config, imported by both the CLI
(`cli.py`) and the notebook (`notebooks/Genetic_algorithm_for_DNN.ipynb`) so
the two can't silently drift apart - a code review of this project caught
`no_evolve_generations` diverging between them (25 in the CLI's inline
kwargs vs. 8 in the notebook's) precisely because the values were duplicated
by hand in two places.
"""

DNN_PRESET = {
    "min_layers": 4,
    "max_layers": 4,
    "max_neurons": 6,
    "population": 100,
    "generations": 30,
    "target_score": 0.995,
    "mutation_chance": 0.05,
    "structural_mutation_chance": 0.3,
    "no_evolve_generations": 8,
    "elitism_count": 4,
}
