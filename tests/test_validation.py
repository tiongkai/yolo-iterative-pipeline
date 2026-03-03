"""Tests for PipelineValidator."""

import pytest
from pathlib import Path
from pipeline.validation import ValidationResult, HealthReport


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
