from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import torch

from .config import baseline_training, meta_search_options, meta_search_training
from .meta_heuristics import run_search
from .training import train_model


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _serialise_config(config: Dict[str, object]) -> Dict[str, object]:
    serialised: Dict[str, object] = {}
    for key, value in config.items():
        serialised[key] = str(value) if isinstance(value, Path) else value
    return serialised


def run_pipeline(skip_meta: bool = False, device: Optional[torch.device] = None) -> Dict[str, object]:
    run_id = datetime.now(timezone.utc).strftime("run-%Y-%m-%d_%H-%M-%S")
    run_dir = Path("artifacts") / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_steps = 1 if skip_meta else 2

    # Baseline training
    baseline_params = baseline_training()
    baseline_path = run_dir / "baseline_model.pt"
    print(f"[1/{total_steps}] Running baseline (no metaheuristics)...")
    baseline_metrics = train_model(
        baseline_params,
        save_path=baseline_path,
        verbose=True,
        device=device,
    )
    baseline_report = {
        "config": _serialise_config(baseline_params),
        "metrics": baseline_metrics,
        "model_path": str(baseline_path),
    }
    _write_json(run_dir / "baseline_report.json", baseline_report)

    summary: Dict[str, object] = {"run_id": run_id, "baseline": baseline_report}

    if skip_meta:
        summary["meta_search"] = None
        _write_json(run_dir / "pipeline_summary.json", summary)
        print("[1/1] Pipeline finished (metaheuristics skipped).")
        return summary

    # Metaheuristic search
    training_params = meta_search_training()
    search_params = meta_search_options()

    print(f"[2/{total_steps}] Running metaheuristics...")
    started = time.time()
    meta_summary = run_search(training_params, search_params, run_dir, device=device)
    elapsed = time.time() - started
    print(f"[2/{total_steps}] Metaheuristics finished in {elapsed:.1f}s.")

    summary["meta_search"] = meta_summary
    _write_json(run_dir / "pipeline_summary.json", summary)
    return summary
