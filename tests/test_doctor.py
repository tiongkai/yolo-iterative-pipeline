"""Tests for doctor command."""

import pytest
from pathlib import Path
from io import StringIO
import sys
from pipeline.doctor import print_health_report, main
from pipeline.validation import ValidationResult, HealthReport
from pipeline.config import PipelineConfig


def test_print_health_report_healthy(capsys):
    """Test print_health_report with healthy status."""
    report = HealthReport(
        structure=ValidationResult(
            status="pass",
            messages=["✓ All directories exist"],
            details={}
        ),
        config=ValidationResult(
            status="pass",
            messages=["✓ pipeline_config.yaml exists and valid"],
            details={}
        ),
        annotations=ValidationResult(
            status="pass",
            messages=["Working: ℹ No annotations to validate", "Verified: ℹ No annotations to validate"],
            details={"working": {}, "verified": {}}
        ),
        models=ValidationResult(
            status="pass",
            messages=["ℹ No active model found (first run)"],
            details={}
        ),
        overall_status="healthy"
    )

    print_health_report(report)
    captured = capsys.readouterr()

    # Should show healthy status
    assert "✅ Pipeline is healthy" in captured.out or "healthy" in captured.out.lower()
    # Should show structure check
    assert "Structure" in captured.out or "structure" in captured.out.lower()


def test_print_health_report_with_errors(capsys):
    """Test print_health_report with errors."""
    report = HealthReport(
        structure=ValidationResult(
            status="error",
            messages=["✗ Missing required directory: data/verified/images/"],
            details={}
        ),
        config=ValidationResult(
            status="pass",
            messages=["✓ pipeline_config.yaml exists and valid"],
            details={}
        ),
        annotations=ValidationResult(
            status="pass",
            messages=["ℹ No annotations to validate"],
            details={}
        ),
        models=ValidationResult(
            status="pass",
            messages=["ℹ No active model found"],
            details={}
        ),
        overall_status="errors"
    )

    print_health_report(report)
    captured = capsys.readouterr()

    # Should show error status
    assert "❌" in captured.out or "error" in captured.out.lower()
    # Should show the missing directory error
    assert "verified/images" in captured.out


def test_print_health_report_with_warnings(capsys):
    """Test print_health_report with warnings."""
    report = HealthReport(
        structure=ValidationResult(
            status="pass",
            messages=["✓ All directories exist"],
            details={}
        ),
        config=ValidationResult(
            status="pass",
            messages=["✓ Configs valid"],
            details={}
        ),
        annotations=ValidationResult(
            status="warning",
            messages=["⚠ 3 labels missing matching images"],
            details={"missing_images": ["img001.txt", "img002.txt", "img003.txt"]}
        ),
        models=ValidationResult(
            status="pass",
            messages=["✓ Active model loads"],
            details={}
        ),
        overall_status="warnings"
    )

    print_health_report(report)
    captured = capsys.readouterr()

    # Should show warning status
    assert "⚠" in captured.out or "warning" in captured.out.lower()
    # Should show the missing images warning
    assert "missing" in captured.out.lower()


def test_main_healthy_exit_code(tmp_path, monkeypatch):
    """Test main() exits with 0 for healthy pipeline."""
    # Create minimal valid structure (PathManager expects data/ prefix)
    for subdir in ["data/working/images", "data/working/labels",
                   "data/verified/images", "data/verified/labels",
                   "data/eval/images", "data/eval/labels",
                   "data/test/images", "data/test/labels",
                   "data/splits",
                   "models/active", "models/checkpoints", "models/deployed",
                   "configs", "logs"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

    # Create valid config files
    import yaml
    pipeline_cfg = {
        "project_name": "test",
        "classes": ["class1"],
        "trigger_threshold": 50,
        "early_trigger": 25,
        "min_train_images": 10,
        "eval_split_ratio": 0.15,
        "stratify": False,
        "uncertainty_weight": 0.4,
        "disagreement_weight": 0.35,
        "diversity_weight": 0.25,
        "desktop_notify": False,
        "slack_webhook": None,
        "keep_last_n_checkpoints": 10,
    }
    (tmp_path / "configs" / "pipeline_config.yaml").write_text(yaml.dump(pipeline_cfg))
    (tmp_path / "configs" / "yolo_config.yaml").write_text(yaml.dump({"model": "yolo11n.pt"}))

    # Change to tmp directory
    monkeypatch.chdir(tmp_path)

    # Should exit with 0 (healthy)
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0


def test_main_error_exit_code(tmp_path, monkeypatch):
    """Test main() exits with 1 for unhealthy pipeline."""
    # Create incomplete structure (missing verified/images and other required dirs)
    for subdir in ["data/working/images", "data/working/labels"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

    # Change to tmp directory
    monkeypatch.chdir(tmp_path)

    # Should exit with 1 (errors)
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
