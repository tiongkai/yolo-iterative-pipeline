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


class PipelineValidator:
    """Validate pipeline setup and state.

    Provides comprehensive validation of directory structure, configuration,
    annotations, and models. Used by doctor command and components at startup.
    """

    def __init__(self, paths):
        """Initialize validator.

        Args:
            paths: PathManager instance for accessing pipeline paths
        """
        self.paths = paths

    def validate_structure(self) -> ValidationResult:
        """Check all required directories exist with correct layout.

        Validates:
        - All data directories exist (working, verified, eval, test)
        - Each has images/ and labels/ subdirectories
        - No orphaned files in parent directories
        - Models and logs directories exist

        Returns:
            ValidationResult with status and messages
        """
        messages = []
        errors = []
        warnings = []
        details = {}

        # Check required data directories with subdirs
        required_dirs = [
            (self.paths.working_images(), "data/working/images/"),
            (self.paths.working_labels(), "data/working/labels/"),
            (self.paths.verified_images(), "data/verified/images/"),
            (self.paths.verified_labels(), "data/verified/labels/"),
            (self.paths.eval_images(), "data/eval/images/"),
            (self.paths.eval_labels(), "data/eval/labels/"),
            (self.paths.test_images(), "data/test/images/"),
            (self.paths.test_labels(), "data/test/labels/"),
            (self.paths.splits_dir(), "data/splits/"),
            (self.paths.active_model().parent, "models/active/"),
            (self.paths.checkpoint_dir(), "models/checkpoints/"),
            (self.paths.deployed_dir(), "models/deployed/"),
            (self.paths.pipeline_config().parent, "configs/"),
            (self.paths.logs_dir(), "logs/"),
        ]

        for dir_path, display_name in required_dirs:
            if not dir_path.exists():
                errors.append(f"✗ Missing required directory: {display_name}")
            else:
                messages.append(f"✓ {display_name} exists")

        # Check for orphaned files in parent directories
        data_parents = [
            self.paths.working_dir(),
            self.paths.verified_dir(),
            self.paths.eval_dir(),
            self.paths.test_dir(),
        ]

        orphaned_files = []
        for parent_dir in data_parents:
            if parent_dir.exists():
                # Look for .txt or image files in parent (should be in subdirs)
                for pattern in ["*.txt", "*.png", "*.jpg", "*.jpeg"]:
                    files = list(parent_dir.glob(pattern))
                    if files:
                        orphaned_files.extend(files)
                        warnings.append(
                            f"⚠ Found {len(files)} orphaned files in {parent_dir.relative_to(self.paths.root_dir)}"
                        )

        if orphaned_files:
            details["orphaned_files"] = [str(f) for f in orphaned_files]
            warnings.append("→ Move files into images/ and labels/ subdirectories")

        # Determine overall status
        if errors:
            status = "error"
            messages = errors + warnings
        elif warnings:
            status = "warning"
            messages = messages + warnings
        else:
            status = "pass"
            messages.append("All required directories exist with correct layout")

        return ValidationResult(
            status=status,
            messages=messages,
            details=details
        )
