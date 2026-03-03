"""Tests for PathManager."""

import pytest
from pathlib import Path
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


def test_path_manager_working_paths(tmp_path):
    """Test PathManager returns correct working directory paths."""
    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=50,
        eval_split_ratio=0.15,
        stratify=True,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(tmp_path, config)

    # Test working directory paths
    assert paths.working_dir() == tmp_path / "data" / "working"
    assert paths.working_images() == tmp_path / "data" / "working" / "images"
    assert paths.working_labels() == tmp_path / "data" / "working" / "labels"


def test_path_manager_verified_paths(tmp_path):
    """Test PathManager returns correct verified directory paths."""
    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=50,
        eval_split_ratio=0.15,
        stratify=True,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(tmp_path, config)

    # Test verified directory paths
    assert paths.verified_dir() == tmp_path / "data" / "verified"
    assert paths.verified_images() == tmp_path / "data" / "verified" / "images"
    assert paths.verified_labels() == tmp_path / "data" / "verified" / "labels"


def test_path_manager_eval_and_test_paths(tmp_path):
    """Test PathManager returns correct eval and test directory paths."""
    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=50,
        eval_split_ratio=0.15,
        stratify=True,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(tmp_path, config)

    # Test eval paths
    assert paths.eval_dir() == tmp_path / "data" / "eval"
    assert paths.eval_images() == tmp_path / "data" / "eval" / "images"
    assert paths.eval_labels() == tmp_path / "data" / "eval" / "labels"

    # Test test paths
    assert paths.test_dir() == tmp_path / "data" / "test"
    assert paths.test_images() == tmp_path / "data" / "test" / "images"
    assert paths.test_labels() == tmp_path / "data" / "test" / "labels"
