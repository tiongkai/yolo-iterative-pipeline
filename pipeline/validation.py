"""Pipeline validation and health checking.

This module provides comprehensive validation of pipeline setup,
directory structure, configuration files, annotations, and models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        status: One of "pass", "warning", "error"
        messages: Human-readable messages describing the result
        details: Additional structured information about the check
    """
    status: str  # "pass", "warning", "error"
    messages: List[str]
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate status value."""
        if self.status not in ["pass", "warning", "error"]:
            raise ValueError(
                f"Invalid status: {self.status}. "
                f"Must be 'pass', 'warning', or 'error'"
            )

    def is_error(self) -> bool:
        """Check if validation failed with errors."""
        return self.status == "error"

    def is_warning(self) -> bool:
        """Check if validation passed with warnings."""
        return self.status == "warning"

    def is_pass(self) -> bool:
        """Check if validation passed without issues."""
        return self.status == "pass"


@dataclass
class HealthReport:
    """Complete pipeline health check report.

    Attributes:
        structure: Directory structure validation result
        config: Configuration validation result
        annotations: Annotation format validation result
        models: Model validation result
        overall_status: One of "healthy", "warnings", "errors"
    """
    structure: ValidationResult
    config: ValidationResult
    annotations: ValidationResult
    models: ValidationResult
    overall_status: str  # "healthy", "warnings", "errors"

    def __post_init__(self):
        """Validate overall_status value."""
        if self.overall_status not in ["healthy", "warnings", "errors"]:
            raise ValueError(
                f"Invalid overall_status: {self.overall_status}. "
                f"Must be 'healthy', 'warnings', or 'errors'"
            )

    def is_healthy(self) -> bool:
        """Check if pipeline is healthy enough to run.

        Returns:
            True if overall_status is "healthy" or "warnings"
        """
        return self.overall_status in ["healthy", "warnings"]
