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
