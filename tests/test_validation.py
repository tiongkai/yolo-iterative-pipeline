"""Tests for PipelineValidator."""

import pytest
from pathlib import Path
from pipeline.validation import ValidationResult, HealthReport, PipelineValidator
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


def test_validation_result_creation():
    """Test ValidationResult can be created with required fields."""
    result = ValidationResult(
        status="pass",
        messages=["All checks passed"],
        details={"count": 10}
    )

    assert result.status == "pass"
    assert result.messages == ["All checks passed"]
    assert result.details == {"count": 10}


def test_validation_result_status_values():
    """Test ValidationResult accepts valid status values."""
    for status in ["pass", "warning", "error"]:
        result = ValidationResult(
            status=status,
            messages=[],
            details={}
        )
        assert result.status == status


def test_health_report_creation():
    """Test HealthReport can be created with all validation results."""
    structure = ValidationResult("pass", [], {})
    config = ValidationResult("pass", [], {})
    annotations = ValidationResult("warning", ["3 missing images"], {})
    models = ValidationResult("pass", [], {})

    report = HealthReport(
        structure=structure,
        config=config,
        annotations=annotations,
        models=models,
        overall_status="warnings"
    )

    assert report.structure.status == "pass"
    assert report.config.status == "pass"
    assert report.annotations.status == "warning"
    assert report.models.status == "pass"
    assert report.overall_status == "warnings"


def test_health_report_is_healthy():
    """Test HealthReport.is_healthy() method."""
    # Healthy report
    healthy = HealthReport(
        structure=ValidationResult("pass", [], {}),
        config=ValidationResult("pass", [], {}),
        annotations=ValidationResult("pass", [], {}),
        models=ValidationResult("pass", [], {}),
        overall_status="healthy"
    )
    assert healthy.is_healthy() is True

    # Warning report (still healthy)
    warnings = HealthReport(
        structure=ValidationResult("pass", [], {}),
        config=ValidationResult("pass", [], {}),
        annotations=ValidationResult("warning", [], {}),
        models=ValidationResult("pass", [], {}),
        overall_status="warnings"
    )
    assert warnings.is_healthy() is True

    # Error report (not healthy)
    errors = HealthReport(
        structure=ValidationResult("error", [], {}),
        config=ValidationResult("pass", [], {}),
        annotations=ValidationResult("pass", [], {}),
        models=ValidationResult("pass", [], {}),
        overall_status="errors"
    )
    assert errors.is_healthy() is False


def test_validation_result_invalid_status():
    """Test ValidationResult rejects invalid status values."""
    with pytest.raises(ValueError, match="Invalid status"):
        ValidationResult(
            status="invalid",
            messages=[],
            details={}
        )


def test_validation_result_helper_methods():
    """Test ValidationResult helper methods."""
    pass_result = ValidationResult("pass", [], {})
    assert pass_result.is_pass() is True
    assert pass_result.is_warning() is False
    assert pass_result.is_error() is False

    warning_result = ValidationResult("warning", ["warning msg"], {})
    assert warning_result.is_pass() is False
    assert warning_result.is_warning() is True
    assert warning_result.is_error() is False

    error_result = ValidationResult("error", ["error msg"], {})
    assert error_result.is_pass() is False
    assert error_result.is_warning() is False
    assert error_result.is_error() is True


def test_health_report_invalid_overall_status():
    """Test HealthReport rejects invalid overall_status values."""
    with pytest.raises(ValueError, match="Invalid overall_status"):
        HealthReport(
            structure=ValidationResult("pass", [], {}),
            config=ValidationResult("pass", [], {}),
            annotations=ValidationResult("pass", [], {}),
            models=ValidationResult("pass", [], {}),
            overall_status="invalid"
        )


@pytest.fixture
def setup_test_structure(tmp_path):
    """Create a valid YOLO directory structure for testing."""
    # Create all required directories
    dirs = [
        "data/working/images",
        "data/working/labels",
        "data/verified/images",
        "data/verified/labels",
        "data/eval/images",
        "data/eval/labels",
        "data/test/images",
        "data/test/labels",
        "data/splits",
        "models/active",
        "models/checkpoints",
        "models/deployed",
        "logs",
        "configs",
    ]

    for dir_path in dirs:
        (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)

    return tmp_path


def test_validate_structure_pass(setup_test_structure):
    """Test structure validation passes with valid structure."""
    root = setup_test_structure
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

    paths = PathManager(root, config)
    validator = PipelineValidator(paths)

    result = validator.validate_structure()

    assert result.status == "pass"
    assert len(result.messages) > 0
    assert "All required directories exist" in " ".join(result.messages)


def test_validate_structure_missing_directory(tmp_path):
    """Test structure validation fails with missing directory."""
    # Create incomplete structure (missing working/images)
    (tmp_path / "data" / "working" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "verified" / "images").mkdir(parents=True)
    (tmp_path / "data" / "verified" / "labels").mkdir(parents=True)

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
    validator = PipelineValidator(paths)

    result = validator.validate_structure()

    assert result.status == "error"
    assert any("data/working/images" in msg for msg in result.messages)


def test_validate_structure_orphaned_files(setup_test_structure):
    """Test structure validation warns about orphaned files."""
    root = setup_test_structure

    # Create orphaned file in parent directory
    (root / "data" / "working" / "orphan.txt").write_text("test")

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

    paths = PathManager(root, config)
    validator = PipelineValidator(paths)

    result = validator.validate_structure()

    assert result.status == "warning"
    assert any("orphaned" in msg.lower() for msg in result.messages)


def test_validate_config_pass(setup_test_structure):
    """Test config validation passes with valid config files."""
    import yaml

    root = setup_test_structure

    # Create valid config files
    pipeline_cfg = {
        "project_name": "test",
        "classes": ["class1", "class2"],
        "trigger_threshold": 50,
        "early_trigger": 25,
        "min_train_images": 50,
        "eval_split_ratio": 0.15,
        "stratify": True,
        "uncertainty_weight": 0.4,
        "disagreement_weight": 0.35,
        "diversity_weight": 0.25,
        "desktop_notify": False,
        "slack_webhook": None,
        "keep_last_n_checkpoints": 10,
    }

    yolo_cfg = {
        "model": "yolo11n.pt",
        "epochs": 100,
        "batch_size": 16,
        "imgsz": 1280,
    }

    (root / "configs" / "pipeline_config.yaml").write_text(yaml.dump(pipeline_cfg))
    (root / "configs" / "yolo_config.yaml").write_text(yaml.dump(yolo_cfg))

    config = PipelineConfig.from_yaml(root / "configs" / "pipeline_config.yaml")
    paths = PathManager(root, config)
    validator = PipelineValidator(paths)

    result = validator.validate_config()

    assert result.status == "pass"


def test_validate_config_missing_file(setup_test_structure):
    """Test config validation fails with missing config file."""
    root = setup_test_structure

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

    paths = PathManager(root, config)
    validator = PipelineValidator(paths)

    result = validator.validate_config()

    assert result.status == "error"
    assert any("pipeline_config.yaml" in msg for msg in result.messages)


def test_validate_annotations_pass(setup_test_structure):
    """Test annotation validation passes with valid YOLO format."""
    root = setup_test_structure

    # Create valid annotation files
    labels_dir = root / "data" / "verified" / "labels"
    images_dir = root / "data" / "verified" / "images"
    verified_dir = root / "data" / "verified"

    # Create classes.txt
    (verified_dir / "classes.txt").write_text("class1\nclass2\n")

    # Create matching image and label
    (labels_dir / "img001.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.2 0.2")
    (images_dir / "img001.png").touch()

    config = PipelineConfig(
        project_name="test",
        classes=["class1", "class2"],
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

    paths = PathManager(root, config)
    validator = PipelineValidator(paths)

    result = validator.validate_annotations(root / "data" / "verified")

    assert result.status == "pass"
    assert result.details["label_count"] == 1


def test_validate_annotations_invalid_format(setup_test_structure):
    """Test annotation validation fails with invalid YOLO format."""
    root = setup_test_structure

    labels_dir = root / "data" / "verified" / "labels"

    # Create invalid annotation (not enough values)
    (labels_dir / "img001.txt").write_text("0 0.5 0.5")

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

    paths = PathManager(root, config)
    validator = PipelineValidator(paths)

    result = validator.validate_annotations(root / "data" / "verified")

    assert result.status == "error"
    assert any("invalid format" in msg.lower() for msg in result.messages)


def test_validate_model_pass(setup_test_structure):
    """Test model validation passes when no active model exists."""
    root = setup_test_structure

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

    paths = PathManager(root, config)
    validator = PipelineValidator(paths)

    result = validator.validate_model()

    # Should pass with info message (no model yet)
    assert result.status == "pass"
    assert any("no active model" in msg.lower() for msg in result.messages)


def test_validate_model_load_failure(setup_test_structure):
    """Test model validation fails gracefully with corrupted model file."""
    root = setup_test_structure

    # Create invalid model file (not a real .pt file)
    model_path = root / "models" / "active" / "best.pt"
    model_path.write_text("corrupted data")

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

    paths = PathManager(root, config)
    validator = PipelineValidator(paths)

    result = validator.validate_model()

    assert result.status == "error"
    assert any("failed to load" in msg.lower() for msg in result.messages)
    assert result.details.get("model_path") == str(model_path)


def test_full_health_check(setup_test_structure):
    """Test full_health_check aggregates all validation checks."""
    import yaml

    root = setup_test_structure

    # Create valid config
    pipeline_cfg = {
        "project_name": "test",
        "classes": ["class1"],
        "trigger_threshold": 50,
        "early_trigger": 25,
        "min_train_images": 50,
        "eval_split_ratio": 0.15,
        "stratify": True,
        "uncertainty_weight": 0.4,
        "disagreement_weight": 0.35,
        "diversity_weight": 0.25,
        "desktop_notify": False,
        "slack_webhook": None,
        "keep_last_n_checkpoints": 10,
    }

    yolo_cfg = {
        "model": "yolo11n.pt",
        "epochs": 100,
    }

    (root / "configs" / "pipeline_config.yaml").write_text(yaml.dump(pipeline_cfg))
    (root / "configs" / "yolo_config.yaml").write_text(yaml.dump(yolo_cfg))

    config = PipelineConfig.from_yaml(root / "configs" / "pipeline_config.yaml")
    paths = PathManager(root, config)
    validator = PipelineValidator(paths)

    report = validator.full_health_check()

    assert isinstance(report, HealthReport)
    assert report.structure.status == "pass"
    assert report.config.status == "pass"
    assert report.annotations.status == "pass"
    assert report.models.status == "pass"
    assert report.overall_status == "healthy"
    assert report.is_healthy() is True
