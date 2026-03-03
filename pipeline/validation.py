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

    def is_healthy(self) -> bool:
        """Check if pipeline is healthy enough to run.

        Returns:
            True if overall_status is "healthy" or "warnings"
        """
        return self.overall_status in ["healthy", "warnings"]
