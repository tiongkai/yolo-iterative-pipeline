"""Tests for train.py integration with PathManager and manifests."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from pipeline.train import train_model, create_data_yaml
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig, YOLOConfig
import yaml


@pytest.fixture
def setup_training_env(tmp_path):
    """Create complete training environment."""
    # Create directory structure
    verified_labels = tmp_path / "data" / "verified" / "labels"
    verified_images = tmp_path / "data" / "verified" / "images"
    test_labels = tmp_path / "data" / "test" / "labels"
    test_images = tmp_path / "data" / "test" / "images"

    verified_labels.mkdir(parents=True)
    verified_images.mkdir(parents=True)
    test_labels.mkdir(parents=True)
    test_images.mkdir(parents=True)

    # Create classes.txt in verified directory
    verified_dir = tmp_path / "data" / "verified"
    (verified_dir / "classes.txt").write_text("boat\n")

    # Create training data (20 images)
    for i in range(20):
        (verified_labels / f"img{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1")
        (verified_images / f"img{i:03d}.png").touch()

    # Create test data (5 images)
    for i in range(5):
        (test_labels / f"test{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1")
        (test_images / f"test{i:03d}.png").touch()

    # Create configs
    config = PipelineConfig(
        project_name="test",
        classes=["boat"],
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

    yolo_config = YOLOConfig(
        model="yolo11n.pt",
        epochs=1,  # Short for testing
        batch_size=16,
        imgsz=640
    )

    paths = PathManager(tmp_path, config)

    return tmp_path, paths, config, yolo_config


def test_create_data_yaml_with_manifests(tmp_path):
    """Test create_data_yaml uses manifest paths."""
    # Create manifest files
    splits_dir = tmp_path / "data" / "splits"
    splits_dir.mkdir(parents=True)

    train_manifest = splits_dir / "train.txt"
    eval_manifest = splits_dir / "eval.txt"
    test_dir = tmp_path / "data" / "test"

    train_manifest.write_text("/path/to/img1.png\n/path/to/img2.png")
    eval_manifest.write_text("/path/to/img3.png")

    output_path = tmp_path / "data.yaml"

    # Create data.yaml
    create_data_yaml(
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        test_dir=test_dir,
        classes=["boat", "person"],
        output_path=output_path,
        root_dir=tmp_path
    )

    # Verify file was created
    assert output_path.exists()

    # Verify content
    with open(output_path) as f:
        data = yaml.safe_load(f)

    assert "train" in data
    assert "val" in data
    assert "test" in data
    assert "names" in data

    # Check it references manifest files, not directories
    assert "splits/train.txt" in data["train"]
    assert "splits/eval.txt" in data["val"]
    assert data["names"] == {0: "boat", 1: "person"}


def test_train_model_uses_pathmanager(setup_training_env):
    """Test train_model uses PathManager for all paths."""
    root, paths, config, yolo_config = setup_training_env

    # Mock YOLO training to avoid actual model training
    with patch('pipeline.train.YOLO') as mock_yolo_class:
        # Create mock model and results
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = str(paths.checkpoint_dir() / "test_run")

        # Create mock validation results
        mock_box = MagicMock()
        mock_box.map50 = 0.85
        mock_box.map = 0.75
        mock_box.mp = 0.82
        mock_box.mr = 0.78

        mock_val_results = MagicMock()
        mock_val_results.box = mock_box

        # Setup mocks
        mock_yolo_class.return_value = mock_model
        mock_model.train.return_value = mock_results
        mock_model.val.return_value = mock_val_results

        # Run training
        version, checkpoint_dir = train_model(
            pipeline_config=config,
            yolo_config=yolo_config,
            paths=paths,
            bootstrap=False,
            from_scratch=True
        )

        # Verify manifests were generated
        assert paths.train_manifest().exists()
        assert paths.eval_manifest().exists()

        # Verify data.yaml was created
        assert paths.data_yaml().exists()

        # Verify data.yaml uses manifest paths
        with open(paths.data_yaml()) as f:
            data = yaml.safe_load(f)

        assert "splits/train.txt" in data["train"]
        assert "splits/eval.txt" in data["val"]

        # Verify YOLO was called with correct project path
        mock_model.train.assert_called_once()
        call_kwargs = mock_model.train.call_args[1]
        assert str(paths.checkpoint_dir()) in call_kwargs['project']

        # Verify training history was saved
        assert paths.training_history().exists()


def test_train_model_generates_manifests(setup_training_env):
    """Test train_model generates manifests before training."""
    root, paths, config, yolo_config = setup_training_env

    # Mock YOLO to avoid actual training
    with patch('pipeline.train.YOLO') as mock_yolo_class:
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = str(paths.checkpoint_dir() / "test_run")

        mock_box = MagicMock()
        mock_box.map50 = 0.85
        mock_box.map = 0.75
        mock_box.mp = 0.82
        mock_box.mr = 0.78

        mock_val_results = MagicMock()
        mock_val_results.box = mock_box

        mock_yolo_class.return_value = mock_model
        mock_model.train.return_value = mock_results
        mock_model.val.return_value = mock_val_results

        # Verify manifests don't exist yet
        assert not paths.train_manifest().exists()
        assert not paths.eval_manifest().exists()

        # Run training
        train_model(
            pipeline_config=config,
            yolo_config=yolo_config,
            paths=paths,
            bootstrap=False,
            from_scratch=True
        )

        # Verify manifests were created
        assert paths.train_manifest().exists()
        assert paths.eval_manifest().exists()

        # Verify manifest contents
        with open(paths.train_manifest()) as f:
            train_images = [line.strip() for line in f]

        with open(paths.eval_manifest()) as f:
            eval_images = [line.strip() for line in f]

        # 20 images total, 20% eval = 4 eval, 16 train
        assert len(train_images) == 16
        assert len(eval_images) == 4

        # Verify no overlap
        assert len(set(train_images) & set(eval_images)) == 0


def test_train_model_minimum_images_check(setup_training_env):
    """Test train_model checks minimum images requirement."""
    root, paths, config, yolo_config = setup_training_env

    # Remove most images (keep only 5)
    labels_dir = paths.verified_labels()
    for label_file in list(labels_dir.glob("*.txt"))[5:]:
        label_file.unlink()
        image_file = paths.verified_images() / label_file.with_suffix(".png").name
        if image_file.exists():
            image_file.unlink()

    # Should raise ValueError
    with pytest.raises(ValueError, match="Need at least 10"):
        train_model(
            pipeline_config=config,
            yolo_config=yolo_config,
            paths=paths,
            bootstrap=False,
            from_scratch=True
        )
