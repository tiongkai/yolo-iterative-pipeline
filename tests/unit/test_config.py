# tests/unit/test_config.py
import pytest
from pathlib import Path
from pipeline.config import PipelineConfig, YOLOConfig

def test_load_pipeline_config_from_yaml():
    """Test loading pipeline configuration from YAML."""
    config = PipelineConfig.from_yaml("configs/pipeline_config.yaml")

    assert config.project_name is not None
    assert isinstance(config.classes, list)
    assert config.trigger_threshold > 0
    assert 0 < config.eval_split_ratio < 1

def test_pipeline_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError, match="trigger_threshold must be positive"):
        PipelineConfig(
            project_name="test",
            classes=["class1"],
            trigger_threshold=-1,
            eval_split_ratio=0.15
        )

def test_load_yolo_config_from_yaml():
    """Test loading YOLO configuration from YAML."""
    config = YOLOConfig.from_yaml("configs/yolo_config.yaml")
    assert config.model == "yolo11n.pt"  # Will upgrade to yolo26n.pt when available
    assert config.imgsz == 1280
    assert config.epochs == 50

def test_pipeline_config_invalid_eval_split():
    """Test validation of invalid eval_split_ratio."""
    with pytest.raises(ValueError, match="eval_split_ratio must be between 0 and 1"):
        PipelineConfig(
            project_name="test",
            classes=["class1"],
            eval_split_ratio=1.5
        )

def test_weights_do_not_sum_to_one():
    """Test validation of weights that don't sum to 1.0."""
    with pytest.raises(ValueError, match="Active learning weights must sum to 1.0"):
        PipelineConfig(
            project_name="test",
            classes=["class1"],
            uncertainty_weight=0.5,
            disagreement_weight=0.3,
            diversity_weight=0.1  # Sum = 0.9, not 1.0
        )

def test_from_yaml_missing_file():
    """Test loading from non-existent file."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        PipelineConfig.from_yaml("nonexistent.yaml")

def test_yolo_config_invalid_epochs():
    """Test validation of negative epochs."""
    with pytest.raises(ValueError, match="epochs must be positive"):
        YOLOConfig(epochs=-1)

def test_yolo_config_invalid_batch_size():
    """Test validation of invalid batch size."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        YOLOConfig(batch_size=0)

def test_yolo_config_invalid_imgsz():
    """Test validation of too small image size."""
    with pytest.raises(ValueError, match="imgsz must be >= 320"):
        YOLOConfig(imgsz=100)
