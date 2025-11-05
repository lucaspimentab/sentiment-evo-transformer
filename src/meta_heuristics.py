from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pyswarms.single import GlobalBestPSO
from scipy.optimize import differential_evolution as scipy_differential_evolution
from scipy.optimize import dual_annealing

from .training import train_model


PARAMETER_SPECS = [
    {"name": "embedding_dim", "type": "choice", "options": [64, 96, 128, 160, 192, 224, 256]},
    {"name": "feedforward_dim", "type": "choice", "options": [256, 320, 384, 448, 512, 640, 768, 896, 1024]},
    {"name": "num_heads", "type": "choice", "options": [2, 4, 8]},
    {"name": "num_layers", "type": "choice", "options": [1, 2, 3, 4]},
    {"name": "dropout", "type": "continuous", "low": 0.05, "high": 0.5},
    {"name": "learning_rate", "type": "log", "low": 1e-4, "high": 5e-3},
    {"name": "weight_decay", "type": "log", "low": 1e-6, "high": 5e-4},
    {"name": "batch_size", "type": "choice", "options": [32, 48, 64, 80, 96, 112, 128]},
    {"name": "max_length", "type": "choice", "options": [64, 96, 128, 160, 192]},
]

PARAMETER_NAMES = [spec["name"] for spec in PARAMETER_SPECS]


def _choice_values(spec: Dict[str, object]) -> Sequence[int]:
    return spec["options"]  # type: ignore[index]


def clamp_vector(vector: Sequence[float]) -> List[float]:
    clamped: List[float] = []
    for value, spec in zip(vector, PARAMETER_SPECS):
        if spec["type"] == "choice":
            options = _choice_values(spec)
            clamped.append(max(0.0, min(float(len(options) - 1), value)))
        elif spec["type"] == "continuous":
            clamped.append(max(spec["low"], min(spec["high"], value)))  # type: ignore[index]
        elif spec["type"] == "log":
            low = math.log10(spec["low"])  # type: ignore[index]
            high = math.log10(spec["high"])  # type: ignore[index]
            clamped.append(max(low, min(high, value)))
        else:
            raise ValueError("Unknown parameter type.")
    return clamped


def decode_vector(vector: Sequence[float]) -> Dict[str, Union[float, int]]:
    params: Dict[str, Union[float, int]] = {}
    for value, spec in zip(vector, PARAMETER_SPECS):
        if spec["type"] == "choice":
            options = _choice_values(spec)
            index = int(round(max(0.0, min(float(len(options) - 1), value))))
            params[spec["name"]] = options[index]
        elif spec["type"] == "continuous":
            clipped = max(spec["low"], min(spec["high"], value))  # type: ignore[index]
            params[spec["name"]] = round(clipped, 6)
        elif spec["type"] == "log":
            low = math.log10(spec["low"])  # type: ignore[index]
            high = math.log10(spec["high"])  # type: ignore[index]
            clipped = max(low, min(high, value))
            params[spec["name"]] = 10 ** clipped
        else:
            raise ValueError("Unknown parameter type.")
    return params


def config_to_key(config: Dict[str, Union[float, int]]) -> Tuple[Union[float, int], ...]:
    return tuple(config[name] for name in PARAMETER_NAMES)


def parameter_bounds() -> Tuple[List[Tuple[float, float]], np.ndarray, np.ndarray]:
    lowers: List[float] = []
    uppers: List[float] = []
    for spec in PARAMETER_SPECS:
        if spec["type"] == "choice":
            lowers.append(0.0)
            uppers.append(float(len(_choice_values(spec)) - 1))
        elif spec["type"] == "continuous":
            lowers.append(float(spec["low"]))  # type: ignore[index]
            uppers.append(float(spec["high"]))  # type: ignore[index]
        elif spec["type"] == "log":
            lowers.append(math.log10(spec["low"]))  # type: ignore[index]
            uppers.append(math.log10(spec["high"]))  # type: ignore[index]
        else:
            raise ValueError("Unknown parameter type.")
    bounds = list(zip(lowers, uppers))
    return bounds, np.array(lowers, dtype=float), np.array(uppers, dtype=float)


class ObjectiveEvaluator:
    def __init__(
        self,
        base_config: Dict[str, object],
        algorithm: str,
        max_evals: int,
        device: Optional[torch.device],
    ) -> None:
        self.base_config = dict(base_config)
        self.algorithm = algorithm
        self.max_evals = max_evals
        self.device = device
        self.cache: Dict[Tuple[Union[float, int], ...], float] = {}
        self.history: List[Dict[str, object]] = []
        self.count = 0

    def evaluate(self, vector: Sequence[float]) -> float:
        config = decode_vector(vector)
        key = config_to_key(config)
        if key in self.cache:
            print(f"[{self.algorithm.upper()}] reutilizando configuração (score {self.cache[key]:.4f}).")
            return self.cache[key]

        if self.count >= self.max_evals:
            raise RuntimeError("Evaluation budget exhausted.")

        idx = self.count + 1
        print(f"[{self.algorithm.upper()}] avaliação {idx}/{self.max_evals}...")
        self.count = idx

        training_config = dict(self.base_config)
        training_config.update(config)

        metrics = train_model(training_config, save_path=None, verbose=False, device=self.device)
        score = metrics.get("accuracy", 0.0)

        record = {
            "evaluation": idx,
            "config": config,
            "metrics": metrics,
            "score": score,
            "algorithm": self.algorithm,
        }
        self.history.append(record)
        self.cache[key] = score
        print(f"[{self.algorithm.upper()}] conclusão da avaliação {idx}: accuracy {score:.4f}")
        return score


def run_differential_evolution(
    evaluator: ObjectiveEvaluator,
    bounds: Sequence[Tuple[float, float]],
) -> Tuple[List[float], float]:
    def objective(vec: Sequence[float]) -> float:
        return -evaluator.evaluate(vec)

    maxiter = max(1, evaluator.max_evals // 5)
    result = scipy_differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        polish=False,
        updating="deferred",
    )
    vector = clamp_vector(result.x.tolist())
    score = -float(result.fun)
    return vector, score


def run_particle_swarm(
    evaluator: ObjectiveEvaluator,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[List[float], float]:
    dimensions = len(PARAMETER_SPECS)
    n_particles = min(8, max(2, evaluator.max_evals))
    options = {"c1": 1.5, "c2": 1.5, "w": 0.5}
    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=dimensions,
        options=options,
        bounds=(lower, upper),
    )

    def objective(X: np.ndarray) -> np.ndarray:
        return np.array([-evaluator.evaluate(vec) for vec in X])

    iters = max(1, evaluator.max_evals // n_particles)
    best_cost, best_pos = optimizer.optimize(objective, iters=iters, verbose=False)
    vector = clamp_vector(best_pos.tolist())
    score = -float(best_cost)
    return vector, score


def run_simulated_annealing(
    evaluator: ObjectiveEvaluator,
    bounds: Sequence[Tuple[float, float]],
) -> Tuple[List[float], float]:
    def objective(vec: Sequence[float]) -> float:
        return -evaluator.evaluate(vec)

    result = dual_annealing(
        objective,
        bounds,
        maxiter=max(1, evaluator.max_evals),
        no_local_search=True,
    )
    vector = clamp_vector(result.x.tolist())
    score = -float(result.fun)
    return vector, score


def run_algorithm(
    name: str,
    evaluator: ObjectiveEvaluator,
    bounds: Sequence[Tuple[float, float]],
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[Dict[str, Union[float, int]], float, ObjectiveEvaluator]:
    if name == "de":
        vector, score = run_differential_evolution(evaluator, bounds)
    elif name == "pso":
        vector, score = run_particle_swarm(evaluator, lower, upper)
    elif name == "sa":
        vector, score = run_simulated_annealing(evaluator, bounds)
    else:
        raise ValueError(f"Unknown algorithm '{name}'.")

    best_config = decode_vector(vector)
    return best_config, score, evaluator


def run_search(
    base_training: Dict[str, object],
    options: Dict[str, object],
    run_dir: Path,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    algorithms: Tuple[str, ...] = tuple(options.get("algorithms", ("de", "pso", "sa")))
    evaluations = int(options.get("evaluations", 10))
    random_state = int(options.get("random_state", 0))
    final_epochs = int(options.get("final_epochs", base_training.get("epochs", 5)))
    final_max_samples = options.get("final_max_samples")

    random.seed(random_state)
    torch.manual_seed(random_state)

    bounds, lower, upper = parameter_bounds()

    results: Dict[str, Dict[str, object]] = {}
    best_overall = {"algorithm": None, "config": None, "score": float("-inf"), "metrics": None}

    report_path = run_dir / "meta_results.json"
    start = time.time()

    for algorithm in algorithms:
        evaluator = ObjectiveEvaluator(base_training, algorithm, evaluations, device)
        try:
            best_config, best_score, evaluator = run_algorithm(algorithm, evaluator, bounds, lower, upper)
        except RuntimeError as exc:
            best_config = None
            best_score = float("-inf")
            evaluator.history.append(
                {
                    "error": str(exc),
                    "algorithm": algorithm,
                    "evaluation": evaluator.count,
                }
            )

        best_metrics = None
        if evaluator.history:
            best_record = max(evaluator.history, key=lambda item: item.get("score", float("-inf")))
            best_score = best_record.get("score", float("-inf"))
            best_metrics = best_record.get("metrics")
            best_config = best_record.get("config")

        results[algorithm] = {
            "best_config": best_config,
            "best_score": best_score,
            "evaluations": evaluator.count,
            "history": evaluator.history,
            "best_metrics": best_metrics,
        }

        print(
            f"[{algorithm.upper()}] melhor accuracy {best_score:.4f} "
            f"após {evaluator.count} avaliações."
        )

        if best_score > best_overall["score"]:
            best_overall = {
                "algorithm": algorithm,
                "config": best_config,
                "score": best_score,
                "metrics": best_metrics,
            }

    total_duration = time.time() - start
    print(f"Meta-heurísticas finalizadas em {total_duration:.1f} segundos.")

    summary: Dict[str, object] = {
        "algorithms": results,
        "best_overall": best_overall,
        "total_duration_sec": total_duration,
        "evaluation_budget": evaluations,
        "report_path": str(report_path),
    }

    _write_json(report_path, summary)

    best_config = best_overall["config"]
    if best_config is None:
        summary["final_training"] = None
        return summary

    final_config = dict(base_training)
    final_config.update(best_config)
    final_config["epochs"] = final_epochs
    if final_max_samples is not None:
        final_config["max_samples"] = final_max_samples

    best_model_path = run_dir / "best_model.pt"
    final_metrics = train_model(final_config, save_path=best_model_path, verbose=True, device=device)
    summary["final_training"] = {
        "config": best_config,
        "metrics": final_metrics,
        "output": str(best_model_path),
        "epochs": final_epochs,
        "score": final_metrics.get("accuracy", 0.0),
    }
    _write_json(report_path, summary)
    return summary


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
