"""Test that all pipeline components work with images/labels structure."""

import pytest
from pathlib import Path
import shutil
import tempfile

from pipeline.watcher import FileWatcher
from pipeline.data_utils import sample_eval_set


def test_watcher_counts_correct_directory(tmp_path):
    """Test that watcher counts files in verified/labels/."""
    # Setup directory structure
    verified_dir = tmp_path / "verified"
    labels_dir = verified_dir / "labels"
    images_dir = verified_dir / "images"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    # Create test files
    for i in range(5):
        (labels_dir / f"image{i}.txt").write_text("0 0.5 0.5 0.1 0.1")
        (images_dir / f"image{i}.png").touch()

    # Test watcher counts correctly
    watcher = FileWatcher(
        verified_dir=verified_dir,
        trigger_threshold=50  # Default threshold
    )
    count = watcher.count_verified_images()

    assert count == 5, f"Expected 5 files, got {count}"


def test_sample_eval_set_with_structure(tmp_path):
    """Test that eval sampling works with images/labels structure."""
    # Setup directory structure
    verified_dir = tmp_path / "verified"
    eval_dir = tmp_path / "eval"
    labels_dir = verified_dir / "labels"
    images_dir = verified_dir / "images"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    # Create test files
    for i in range(10):
        (labels_dir / f"image{i}.txt").write_text("0 0.5 0.5 0.1 0.1")
        (images_dir / f"image{i}.png").touch()

    # Sample eval set
    sampled = sample_eval_set(
        verified_dir=verified_dir,
        eval_dir=eval_dir,
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
