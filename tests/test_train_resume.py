"""Tests for smart resume training."""

import pytest
from pathlib import Path
from ultralytics import YOLO
from pipeline.train import init_model
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig, YOLOConfig


@pytest.fixture
def setup_model_dirs(tmp_path):
    """Create directory structure for model tests."""
    models_active = tmp_path / "models" / "active"
    models_active.mkdir(parents=True)

    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=10,
        eval_split_ratio=0.15,
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
        epochs=100,
        batch_size=16,
        imgsz=1280
    )

    paths = PathManager(tmp_path, config)

    return tmp_path, paths, yolo_config


def test_init_model_no_active_model(setup_model_dirs):
    """Test init_model loads pretrained when no active model exists."""
    root, paths, yolo_config = setup_model_dirs

    model, source = init_model(paths, yolo_config, from_scratch=False)

    assert isinstance(model, YOLO)
    assert source == "pretrained"
    assert "yolo11n.pt" in str(model.ckpt_path)


def test_init_model_with_active_model(setup_model_dirs):
    """Test init_model loads active model when it exists."""
    root, paths, yolo_config = setup_model_dirs

    # Create a valid active model by copying pretrained
    pretrained = YOLO("yolo11n.pt")
    pretrained.save(str(paths.active_model()))

    model, source = init_model(paths, yolo_config, from_scratch=False)

    assert isinstance(model, YOLO)
    assert source == "active"
    assert str(paths.active_model()) in str(model.ckpt_path)


def test_init_model_from_scratch_flag(setup_model_dirs):
    """Test init_model uses pretrained when from_scratch=True."""
    root, paths, yolo_config = setup_model_dirs

    # Create an active model
    pretrained = YOLO("yolo11n.pt")
    pretrained.save(str(paths.active_model()))

    # Force from scratch
    model, source = init_model(paths, yolo_config, from_scratch=True)

    assert isinstance(model, YOLO)
    assert source == "pretrained"
    assert "yolo11n.pt" in str(model.ckpt_path)


def test_init_model_corrupted_active(setup_model_dirs):
    """Test init_model falls back to pretrained if active model corrupted."""
    root, paths, yolo_config = setup_model_dirs

    # Create corrupted active model file
    paths.active_model().write_text("corrupted data")

    model, source = init_model(paths, yolo_config, from_scratch=False)

    assert isinstance(model, YOLO)
    assert source == "pretrained"
    assert "yolo11n.pt" in str(model.ckpt_path)
