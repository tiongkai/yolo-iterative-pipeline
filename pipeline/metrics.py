"""Metrics calculation and tracking for training pipeline."""

from typing import Dict
import json
from pathlib import Path
from datetime import datetime


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall.

    Args:
        precision: Precision value [0, 1]
        recall: Recall value [0, 1]

    Returns:
        F1 score [0, 1]
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def format_metrics(metrics: Dict[str, float], add_f1: bool = True) -> Dict[str, float]:
    """Format metrics dictionary, optionally adding F1 score.

    Args:
        metrics: Dictionary with precision, recall, mAP, etc.
        add_f1: Whether to calculate and add F1 score

    Returns:
        Formatted metrics dictionary
    """
    result = metrics.copy()

    if add_f1 and "precision" in metrics and "recall" in metrics:
        result["f1"] = calculate_f1_score(metrics["precision"], metrics["recall"])

    return result


def load_training_history(log_path: Path) -> list:
    """Load training history from JSON log.

    Args:
        log_path: Path to training_history.json

    Returns:
        List of training history entries
    """
    if not log_path.exists():
        return []

    with open(log_path) as f:
        return json.load(f)


def append_training_history(
    log_path: Path,
    version: str,
    train_images: int,
    eval_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    training_time_minutes: float,
    notes: str = ""
):
    """Append new entry to training history.

    Args:
        log_path: Path to training_history.json
        version: Model version (e.g., "v003")
        train_images: Number of training images
        eval_metrics: Evaluation metrics on eval set
        test_metrics: Evaluation metrics on test set
        training_time_minutes: Training duration
        notes: Optional notes about this training run
    """
    history = load_training_history(log_path)

    # Calculate improvement over previous
    improvement = {}
    if history:
        prev = history[-1]
        for key in ["mAP50", "f1"]:
            if key in eval_metrics and f"eval_{key}" in prev:
                improvement[f"eval_{key}"] = eval_metrics[key] - prev[f"eval_{key}"]
            if key in test_metrics and f"test_{key}" in prev:
                improvement[f"test_{key}"] = test_metrics[key] - prev[f"test_{key}"]

    entry = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "train_images": train_images,
        **{f"eval_{k}": v for k, v in eval_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "training_time_minutes": training_time_minutes,
        "improvement": improvement,
        "notes": notes,
    }

    history.append(entry)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
