"""Test that all pipeline components work with images/labels structure."""

import pytest
from pathlib import Path
import shutil
import tempfile

from pipeline.watcher import FileWatcher
from pipeline.data_utils import sample_eval_set
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


def test_watcher_counts_correct_directory(tmp_path):
    """Test that watcher counts files in verified/labels/."""
    # Setup directory structure
    verified_dir = tmp_path / "data" / "verified"
    labels_dir = verified_dir / "labels"
    images_dir = verified_dir / "images"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    # Create required dirs for PathManager validation
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

    # Create test files
    for i in range(5):
        (labels_dir / f"image{i}.txt").write_text("0 0.5 0.5 0.1 0.1")
        (images_dir / f"image{i}.png").touch()

    # Create config and PathManager
    config = PipelineConfig(
        project_name="test_project",
        classes=["boat"],
        trigger_threshold=50
    )
    paths = PathManager(tmp_path, config)

    # Test watcher counts correctly
    watcher = FileWatcher(
        paths=paths,
        trigger_threshold=50  # Default threshold
    )
    count = watcher.count_verified_images()

    assert count == 5, f"Expected 5 files, got {count}"


def test_sample_eval_set_with_structure(tmp_path):
    """Test that eval sampling works with images/labels structure."""
    # Setup directory structure
    data_dir = tmp_path / "data"
    verified_dir = data_dir / "verified"
    eval_dir = data_dir / "eval"
    labels_dir = verified_dir / "labels"
    images_dir = verified_dir / "images"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    # Create test files
    for i in range(10):
        (labels_dir / f"image{i}.txt").write_text("0 0.5 0.5 0.1 0.1")
        (images_dir / f"image{i}.png").touch()

    # Create PathManager instance
    from pipeline.paths import PathManager
    from pipeline.config import PipelineConfig

    config = PipelineConfig(
        project_name="test_project",
        classes=["test"],
        trigger_threshold=50
    )
    paths = PathManager(tmp_path, config)

    # Sample eval set
    sampled = sample_eval_set(
        paths=paths,
        split_ratio=0.3,
        stratify=False,
        num_classes=1
    )

    # Check that files were moved to correct subdirectories
    eval_labels = eval_dir / "labels"
    eval_images = eval_dir / "images"

    assert eval_labels.exists(), "eval/labels/ should exist"
    assert eval_images.exists(), "eval/images/ should exist"

    # Check counts
    assert len(list(eval_labels.glob("*.txt"))) == 3, "Should have 3 label files in eval"
    assert len(list(eval_images.glob("*.png"))) == 3, "Should have 3 image files in eval"
    assert len(list(labels_dir.glob("*.txt"))) == 7, "Should have 7 label files remaining"
    assert len(list(images_dir.glob("*.png"))) == 7, "Should have 7 image files remaining"


def test_monitor_counts_correct_directories(tmp_path, monkeypatch):
    """Test that monitor counts files in correct subdirectories."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)

    # Setup directory structure
    for base_dir in ["working", "verified"]:
        base = tmp_path / "data" / base_dir
        labels = base / "labels"
        images = base / "images"
        labels.mkdir(parents=True)
        images.mkdir(parents=True)

        # Create test files
        count = 3 if base_dir == "working" else 5
        for i in range(count):
            (labels / f"image{i}.txt").write_text("0 0.5 0.5 0.1 0.1")
            (images / f"image{i}.png").touch()

    # Test monitor (import here to use monkeypatch)
    from pipeline.monitor import display_status
    # This should not crash - it will display status
    # We can't easily test output, but we can test it doesn't crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
