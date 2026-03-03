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

    # Test sam3 path
    assert paths.sam3_dir() == tmp_path / "data" / "sam3_annotations"


def test_path_manager_manifest_paths(tmp_path):
    """Test PathManager returns correct manifest paths."""
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

    # Test manifest paths
    assert paths.splits_dir() == tmp_path / "data" / "splits"
    assert paths.train_manifest() == tmp_path / "data" / "splits" / "train.txt"
    assert paths.eval_manifest() == tmp_path / "data" / "splits" / "eval.txt"


def test_path_manager_model_paths(tmp_path):
    """Test PathManager returns correct model paths."""
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

    # Test model paths
    assert paths.active_model() == tmp_path / "models" / "active" / "best.pt"
    assert paths.checkpoint_dir() == tmp_path / "models" / "checkpoints"
    assert paths.deployed_dir() == tmp_path / "models" / "deployed"


def test_path_manager_config_and_log_paths(tmp_path):
    """Test PathManager returns correct config and log paths."""
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

    # Test config paths
    assert paths.data_yaml() == tmp_path / "data" / "data.yaml"
    assert paths.pipeline_config() == tmp_path / "configs" / "pipeline_config.yaml"
    assert paths.yolo_config() == tmp_path / "configs" / "yolo_config.yaml"

    # Test log paths
    assert paths.logs_dir() == tmp_path / "logs"
    assert paths.training_history() == tmp_path / "logs" / "training_history.json"
    assert paths.watcher_log() == tmp_path / "logs" / "watcher.log"
    assert paths.auto_move_log() == tmp_path / "logs" / "auto_move.log"
    assert paths.training_lock() == tmp_path / "logs" / ".training.lock"
    assert paths.priority_queue() == tmp_path / "logs" / "priority_queue.txt"
