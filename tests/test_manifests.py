"""Tests for manifest generation."""

import pytest
from pathlib import Path
from pipeline.data_utils import generate_manifests
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


@pytest.fixture
def setup_verified_data(tmp_path):
    """Create verified dataset with images and labels."""
    verified_labels = tmp_path / "data" / "verified" / "labels"
    verified_images = tmp_path / "data" / "verified" / "images"
    verified_labels.mkdir(parents=True)
    verified_images.mkdir(parents=True)

    # Create 10 label/image pairs
    for i in range(10):
        (verified_labels / f"img{i:03d}.txt").write_text(f"0 0.5 0.5 0.1 0.1")
        (verified_images / f"img{i:03d}.png").touch()

    return tmp_path


def test_generate_manifests_creates_files(setup_verified_data):
    """Test generate_manifests creates train.txt and eval.txt."""
    root = setup_verified_data

    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=10,
        eval_split_ratio=0.2,  # 80/20 split
        stratify=False,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(root, config)

    train_count, eval_count = generate_manifests(paths, config)

    # Check manifests exist
    assert paths.train_manifest().exists()
    assert paths.eval_manifest().exists()

    # Check counts (8 train, 2 eval for 80/20 split)
    assert train_count == 8
    assert eval_count == 2


def test_generate_manifests_correct_split_ratio(setup_verified_data):
    """Test generate_manifests maintains correct split ratio."""
    root = setup_verified_data

    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=10,
        eval_split_ratio=0.3,  # 70/30 split
        stratify=False,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(root, config)

    train_count, eval_count = generate_manifests(paths, config)

    # 10 images: 70% = 7 train, 30% = 3 eval
    assert train_count == 7
    assert eval_count == 3


def test_generate_manifests_content_format(setup_verified_data):
    """Test manifest files contain correct image paths."""
    root = setup_verified_data

    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=10,
        eval_split_ratio=0.2,
        stratify=False,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(root, config)

    generate_manifests(paths, config)

    # Read manifests
    with open(paths.train_manifest()) as f:
        train_paths = [line.strip() for line in f]

    with open(paths.eval_manifest()) as f:
        eval_paths = [line.strip() for line in f]

    # Check format: should be paths to images
    for path_str in train_paths + eval_paths:
        path = Path(path_str)
        assert path.suffix in ['.png', '.jpg', '.jpeg']
        assert "verified/images" in str(path)

    # Check no overlap
    assert len(set(train_paths) & set(eval_paths)) == 0


def test_generate_manifests_insufficient_data(tmp_path):
    """Test generate_manifests raises error with insufficient data."""
    verified_labels = tmp_path / "data" / "verified" / "labels"
    verified_images = tmp_path / "data" / "verified" / "images"
    verified_labels.mkdir(parents=True)
    verified_images.mkdir(parents=True)

    # Create only 5 images (less than min_train_images=10)
    for i in range(5):
        (verified_labels / f"img{i:03d}.txt").write_text(f"0 0.5 0.5 0.1 0.1")
        (verified_images / f"img{i:03d}.png").touch()

    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=10,
        eval_split_ratio=0.2,
        stratify=False,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(tmp_path, config)

    # Should raise ValueError
    with pytest.raises(ValueError, match="Need at least 10 labels"):
        generate_manifests(paths, config)
