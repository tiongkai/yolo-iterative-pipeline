"""Unit tests for metrics calculation."""

import pytest
from pathlib import Path
import json
import tempfile
from pipeline.metrics import (
    calculate_f1_score,
    format_metrics,
    load_training_history,
    append_training_history
)


def test_calculate_f1_score():
    """Test F1 score calculation."""
    precision = 0.8
    recall = 0.7
    f1 = calculate_f1_score(precision, recall)
    expected = 2 * (0.8 * 0.7) / (0.8 + 0.7)
    assert abs(f1 - expected) < 0.01


def test_calculate_f1_score_zero_case():
    """Test F1 when precision or recall is zero."""
    assert calculate_f1_score(0, 0.5) == 0.0
    assert calculate_f1_score(0.5, 0) == 0.0
    assert calculate_f1_score(0, 0) == 0.0


def test_format_metrics():
    """Test metrics formatting."""
    metrics = {
        "precision": 0.881,
        "recall": 0.792,
        "mAP50": 0.847,
        "mAP50-95": 0.612,
    }
    formatted = format_metrics(metrics)
    assert "mAP50" in formatted
    assert "f1" in formatted
    assert abs(formatted["f1"] - 0.833) < 0.01  # Expected F1


def test_format_metrics_no_f1():
    """Test format_metrics with add_f1=False."""
    metrics = {
        "precision": 0.8,
        "recall": 0.7,
    }
    formatted = format_metrics(metrics, add_f1=False)
    assert "f1" not in formatted
    assert formatted["precision"] == 0.8


def test_format_metrics_missing_precision_recall():
    """Test format_metrics when precision/recall missing."""
    metrics = {
        "mAP50": 0.85,
    }
    formatted = format_metrics(metrics, add_f1=True)
    assert "f1" not in formatted  # Can't calculate without precision/recall


def test_load_training_history_nonexistent():
    """Test loading history from nonexistent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "nonexistent.json"
        history = load_training_history(log_path)
        assert history == []


def test_load_training_history_existing():
    """Test loading history from existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "history.json"
        test_data = [
            {"version": "v001", "eval_mAP50": 0.75},
            {"version": "v002", "eval_mAP50": 0.82}
        ]
        with open(log_path, "w") as f:
            json.dump(test_data, f)

        history = load_training_history(log_path)
        assert len(history) == 2
        assert history[0]["version"] == "v001"
        assert history[1]["eval_mAP50"] == 0.82


def test_append_training_history():
    """Test appending to training history."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "history.json"

        eval_metrics = {"mAP50": 0.75, "f1": 0.72, "precision": 0.8, "recall": 0.65}
        test_metrics = {"mAP50": 0.73, "f1": 0.70, "precision": 0.78, "recall": 0.63}

        append_training_history(
            log_path=log_path,
            version="v001",
            train_images=100,
            eval_metrics=eval_metrics,
            test_metrics=test_metrics,
            training_time_minutes=15.5,
            notes="First training"
        )

        history = load_training_history(log_path)
        assert len(history) == 1
        assert history[0]["version"] == "v001"
        assert history[0]["train_images"] == 100
        assert history[0]["eval_mAP50"] == 0.75
        assert history[0]["test_mAP50"] == 0.73
        assert history[0]["training_time_minutes"] == 15.5
        assert history[0]["notes"] == "First training"
        assert history[0]["improvement"] == {}  # No previous entry


def test_append_training_history_with_improvement():
    """Test appending with improvement calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "history.json"

        # First entry
        eval_metrics_1 = {"mAP50": 0.75, "f1": 0.72}
        test_metrics_1 = {"mAP50": 0.73, "f1": 0.70}
        append_training_history(
            log_path=log_path,
            version="v001",
            train_images=100,
            eval_metrics=eval_metrics_1,
            test_metrics=test_metrics_1,
            training_time_minutes=15.0
        )

        # Second entry
        eval_metrics_2 = {"mAP50": 0.82, "f1": 0.79}
        test_metrics_2 = {"mAP50": 0.80, "f1": 0.77}
        append_training_history(
            log_path=log_path,
            version="v002",
            train_images=200,
            eval_metrics=eval_metrics_2,
            test_metrics=test_metrics_2,
            training_time_minutes=18.0
        )

        history = load_training_history(log_path)
        assert len(history) == 2

        # Check improvement calculation
        improvement = history[1]["improvement"]
        assert abs(improvement["eval_mAP50"] - 0.07) < 0.01
        assert abs(improvement["eval_f1"] - 0.07) < 0.01
        assert abs(improvement["test_mAP50"] - 0.07) < 0.01
        assert abs(improvement["test_f1"] - 0.07) < 0.01
