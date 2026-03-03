# tests/integration/test_watcher.py
import pytest
import time
from pathlib import Path
from pipeline.watcher import FileWatcher, should_trigger_training
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


def test_should_trigger_training():
    """Test training trigger logic."""
    assert should_trigger_training(
        current_count=100,
        last_train_count=50,
        trigger_threshold=50,
        iteration=0
    ) is True

    assert should_trigger_training(
        current_count=70,
        last_train_count=50,
        trigger_threshold=50,
        iteration=0
    ) is False


def test_should_trigger_training_early_iterations():
    """Test early iteration trigger (lower threshold)."""
    # Iteration 0, 1, 2 should use early_trigger (25)
    assert should_trigger_training(
        current_count=75,
        last_train_count=50,
        trigger_threshold=50,
        iteration=0,
        early_trigger=25
    ) is True

    assert should_trigger_training(
        current_count=75,
        last_train_count=50,
        trigger_threshold=50,
        iteration=2,
        early_trigger=25
    ) is True

    # Iteration 3+ should use standard trigger_threshold (50)
    assert should_trigger_training(
        current_count=75,
        last_train_count=50,
        trigger_threshold=50,
        iteration=3,
        early_trigger=25
    ) is False


def test_file_watcher_initialization(tmp_path):
    """Test file watcher initialization."""
    # Create pipeline structure
    (tmp_path / "data" / "verified" / "images").mkdir(parents=True)
    (tmp_path / "data" / "verified" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "working" / "images").mkdir(parents=True)
    (tmp_path / "data" / "working" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "images").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "test" / "images").mkdir(parents=True)
    (tmp_path / "data" / "test" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "sam3_annotations").mkdir(parents=True)
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "models" / "active").mkdir(parents=True)
    (tmp_path / "models" / "checkpoints").mkdir(parents=True)
    (tmp_path / "models" / "deployed").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)

    # Create config
    config = PipelineConfig(
        project_name="test_project",
        classes=["boat", "human", "motor"],
        trigger_threshold=50
    )

    # Create PathManager
    paths = PathManager(tmp_path, config)

    watcher = FileWatcher(
        paths=paths,
        trigger_threshold=50,
        pipeline_config_path=None,
        yolo_config_path=None
    )

    assert watcher.paths == paths
    assert watcher.trigger_threshold == 50
    assert watcher.last_train_count == 0
    assert watcher.iteration == 0
    assert watcher.is_training is False


def test_count_verified_images(tmp_path):
    """Test counting annotation files."""
    # Create pipeline structure
    verified_labels = tmp_path / "data" / "verified" / "labels"
    verified_labels.mkdir(parents=True)
    (tmp_path / "data" / "verified" / "images").mkdir(parents=True)
    (tmp_path / "data" / "working" / "images").mkdir(parents=True)
    (tmp_path / "data" / "working" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "images").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "test" / "images").mkdir(parents=True)
    (tmp_path / "data" / "test" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "sam3_annotations").mkdir(parents=True)
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "models" / "active").mkdir(parents=True)
    (tmp_path / "models" / "checkpoints").mkdir(parents=True)
    (tmp_path / "models" / "deployed").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)

    # Create some annotation files in the labels subdirectory
    (verified_labels / "img1.txt").write_text("0 0.5 0.5 0.1 0.1")
    (verified_labels / "img2.txt").write_text("1 0.3 0.3 0.2 0.2")
    (verified_labels / "img3.jpg").write_text("not a label")  # Should not count

    # Create config
    config = PipelineConfig(
        project_name="test_project",
        classes=["boat", "human", "motor"],
        trigger_threshold=50
    )

    # Create PathManager
    paths = PathManager(tmp_path, config)

    watcher = FileWatcher(
        paths=paths,
        trigger_threshold=50,
        pipeline_config_path=None,
        yolo_config_path=None
    )

    assert watcher.count_verified_images() == 2


def test_lock_file_prevents_concurrent_training(tmp_path):
    """Test that lock file prevents concurrent training."""
    # Create pipeline structure
    (tmp_path / "data" / "verified" / "images").mkdir(parents=True)
    (tmp_path / "data" / "verified" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "working" / "images").mkdir(parents=True)
    (tmp_path / "data" / "working" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "images").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "test" / "images").mkdir(parents=True)
    (tmp_path / "data" / "test" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "sam3_annotations").mkdir(parents=True)
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "models" / "active").mkdir(parents=True)
    (tmp_path / "models" / "checkpoints").mkdir(parents=True)
    (tmp_path / "models" / "deployed").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)

    # Create config
    config = PipelineConfig(
        project_name="test_project",
        classes=["boat", "human", "motor"],
        trigger_threshold=50
    )

    # Create PathManager
    paths = PathManager(tmp_path, config)

    watcher = FileWatcher(
        paths=paths,
        trigger_threshold=50,
        pipeline_config_path=None,
        yolo_config_path=None
    )

    # Override lock file path to use tmp_path
    watcher.lock_file = logs_dir / ".training.lock"

    # Create lock file
    watcher.lock_file.touch()

    # Attempting to check should not trigger (lock file exists)
    watcher.check_and_trigger()

    # is_training should remain False (didn't start)
    assert watcher.is_training is False

    # Clean up
    watcher.lock_file.unlink()
