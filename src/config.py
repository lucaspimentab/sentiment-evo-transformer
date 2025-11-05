from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

BASELINE_TRAINING: Dict[str, object] = {
    "processed_dir": Path("data"),
    "max_samples": 2000,
    "validation_split": 0.2,
    "random_state": 42,
    "max_length": 96,
    "batch_size": 64,
    "epochs": 3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "embedding_dim": 96,
    "feedforward_dim": 256,
    "num_heads": 2,
    "num_layers": 1,
    "dropout": 0.2,
}

META_SEARCH_TRAINING: Dict[str, object] = {
    "processed_dir": Path("data"),
    "max_samples": 4000,
    "validation_split": 0.2,
    "random_state": 123,
    "max_length": 128,
    "batch_size": 64,
    "epochs": 3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "embedding_dim": 128,
    "feedforward_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
}

META_SEARCH_OPTIONS: Dict[str, object] = {
    "algorithms": ("de", "pso", "sa"),
    "evaluations": 12,
    "random_state": 123,
    "final_epochs": 5,
    "final_max_samples": None,
}


def baseline_training() -> Dict[str, object]:
    return deepcopy(BASELINE_TRAINING)


def meta_search_training() -> Dict[str, object]:
    return deepcopy(META_SEARCH_TRAINING)


def meta_search_options() -> Dict[str, object]:
    return deepcopy(META_SEARCH_OPTIONS)
