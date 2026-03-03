"""Pipeline validation and health checking.

This module provides comprehensive validation of pipeline setup,
directory structure, configuration files, annotations, and models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import yaml


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

    def validate_config(self) -> ValidationResult:
        """Validate configuration files.

        Validates:
        - pipeline_config.yaml exists and parses
        - yolo_config.yaml exists and parses
        - Required fields present

        Returns:
            ValidationResult with status and messages
        """
        messages = []
        errors = []

        # Check pipeline config
        pipeline_cfg_path = self.paths.pipeline_config()
        if not pipeline_cfg_path.exists():
            errors.append(f"✗ Missing config file: {pipeline_cfg_path.name}")
        else:
            try:
                with open(pipeline_cfg_path) as f:
                    yaml.safe_load(f)
                messages.append(f"✓ {pipeline_cfg_path.name} exists and valid")
            except yaml.YAMLError as e:
                errors.append(f"✗ Invalid YAML in {pipeline_cfg_path.name}: {e}")

        # Check YOLO config
        yolo_cfg_path = self.paths.yolo_config()
        if not yolo_cfg_path.exists():
            errors.append(f"✗ Missing config file: {yolo_cfg_path.name}")
        else:
            try:
                with open(yolo_cfg_path) as f:
                    yaml.safe_load(f)
                messages.append(f"✓ {yolo_cfg_path.name} exists and valid")
            except yaml.YAMLError as e:
                errors.append(f"✗ Invalid YAML in {yolo_cfg_path.name}: {e}")

        if errors:
            return ValidationResult(
                status="error",
                messages=errors,
                details={}
            )
        else:
            return ValidationResult(
                status="pass",
                messages=messages,
                details={}
            )

    def validate_annotations(self, dir_path: Path) -> ValidationResult:
        """Validate YOLO annotation format in directory.

        Validates:
        - All .txt files parse as YOLO format
        - Class IDs within valid range
        - Coordinates normalized to [0, 1]
        - Matching images exist for labels

        Args:
            dir_path: Directory containing images/ and labels/ subdirs

        Returns:
            ValidationResult with status and messages
        """
        messages = []
        errors = []
        warnings = []
        details = {}

        labels_dir = dir_path / "labels"
        images_dir = dir_path / "images"

        if not labels_dir.exists():
            return ValidationResult(
                status="error",
                messages=[f"✗ Labels directory not found: {labels_dir}"],
                details={}
            )

        if not images_dir.exists():
            return ValidationResult(
                status="error",
                messages=[f"✗ Images directory not found: {images_dir}"],
                details={}
            )

        # Scan all label files
        label_files = list(labels_dir.glob("*.txt"))
        details["label_count"] = len(label_files)

        if len(label_files) == 0:
            return ValidationResult(
                status="pass",
                messages=["ℹ No annotations to validate"],
                details=details
            )

        invalid_format = []
        missing_images = []
        num_classes = len(self.paths.config.classes)

        for label_file in label_files:
            # Check format
            try:
                with open(label_file) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) != 5:
                            invalid_format.append(
                                f"{label_file.name}:{line_num} (expected 5 values, got {len(parts)})"
                            )
                            continue

                        # Validate class ID
                        try:
                            class_id = int(parts[0])
                            if class_id < 0 or class_id >= num_classes:
                                invalid_format.append(
                                    f"{label_file.name}:{line_num} (class_id {class_id} out of range [0-{num_classes-1}])"
                                )
                        except ValueError:
                            invalid_format.append(
                                f"{label_file.name}:{line_num} (invalid class_id: {parts[0]})"
                            )

                        # Validate coordinates
                        try:
                            coords = [float(x) for x in parts[1:]]
                            for i, coord in enumerate(coords):
                                if coord < 0 or coord > 1:
                                    invalid_format.append(
                                        f"{label_file.name}:{line_num} (coordinate {i+1} out of range [0,1]: {coord})"
                                    )
                        except ValueError:
                            invalid_format.append(
                                f"{label_file.name}:{line_num} (invalid coordinate values)"
                            )
            except Exception as e:
                errors.append(f"✗ Failed to read {label_file.name}: {e}")

            # Check for matching image
            image_found = False
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                image_path = images_dir / f"{label_file.stem}{ext}"
                if image_path.exists():
                    image_found = True
                    break

            if not image_found:
                missing_images.append(label_file.name)

        # Build messages
        if invalid_format:
            errors.extend([f"✗ Invalid format: {err}" for err in invalid_format[:5]])
            if len(invalid_format) > 5:
                errors.append(f"✗ ... and {len(invalid_format) - 5} more format errors")
            details["invalid_format"] = invalid_format
        else:
            messages.append(f"✓ All {len(label_files)} annotations valid YOLO format")

        if missing_images:
            warnings.append(f"⚠ {len(missing_images)} labels missing matching images")
            if len(missing_images) <= 3:
                warnings.extend([f"  → {name}" for name in missing_images])
            else:
                warnings.extend([f"  → {name}" for name in missing_images[:3]])
                warnings.append(f"  → ... and {len(missing_images) - 3} more")
            details["missing_images"] = missing_images
        else:
            messages.append(f"✓ All labels have matching images")

        # Determine status
        if errors:
            return ValidationResult(
                status="error",
                messages=errors + warnings,
                details=details
            )
        elif warnings:
            return ValidationResult(
                status="warning",
                messages=messages + warnings,
                details=details
            )
        else:
            return ValidationResult(
                status="pass",
                messages=messages,
                details=details
            )
