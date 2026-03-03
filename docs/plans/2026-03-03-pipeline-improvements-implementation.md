# Pipeline Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add robustness and usability improvements to YOLO pipeline including centralized path management, comprehensive validation, doctor command, single-command workflow, and atomic file moves.

**Architecture:** Three-layer design with Foundation (PathManager + Validator) → Components (refactored to use foundation) → Features (doctor, process manager). All components validate on startup, fail fast with clear errors.

**Tech Stack:** Python 3.9+, pathlib, dataclasses, subprocess, signal, ultralytics

---

## Phase 1: Foundation - PathManager

### Task 1: Create PathManager with Data Directory Methods

**Goal:** Implement centralized path management for data directories with YOLO structure enforcement.

**Files:**
- Create: `pipeline/paths.py`
- Create: `tests/test_paths.py`

**Step 1: Write the failing test**

Create `tests/test_paths.py`:

```python
"""Tests for PathManager."""

import pytest
from pathlib import Path
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


def test_path_manager_working_paths(tmp_path):
    """Test PathManager returns correct working directory paths."""
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

    # Test working directory paths
    assert paths.working_dir() == tmp_path / "data" / "working"
    assert paths.working_images() == tmp_path / "data" / "working" / "images"
    assert paths.working_labels() == tmp_path / "data" / "working" / "labels"


def test_path_manager_verified_paths(tmp_path):
    """Test PathManager returns correct verified directory paths."""
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

    # Test verified directory paths
    assert paths.verified_dir() == tmp_path / "data" / "verified"
    assert paths.verified_images() == tmp_path / "data" / "verified" / "images"
    assert paths.verified_labels() == tmp_path / "data" / "verified" / "labels"


def test_path_manager_eval_and_test_paths(tmp_path):
    """Test PathManager returns correct eval and test directory paths."""
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

    # Test eval paths
    assert paths.eval_dir() == tmp_path / "data" / "eval"
    assert paths.eval_images() == tmp_path / "data" / "eval" / "images"
    assert paths.eval_labels() == tmp_path / "data" / "eval" / "labels"

    # Test test paths
    assert paths.test_dir() == tmp_path / "data" / "test"
    assert paths.test_images() == tmp_path / "data" / "test" / "images"
    assert paths.test_labels() == tmp_path / "data" / "test" / "labels"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_paths.py::test_path_manager_working_paths -v
```

Expected output: `ModuleNotFoundError: No module named 'pipeline.paths'`

**Step 3: Write minimal implementation**

Create `pipeline/paths.py`:

```python
"""Centralized path management for YOLO pipeline.

This module provides a single source of truth for all directory paths
in the pipeline, enforcing YOLO directory structure (images/ and labels/
subdirectories) everywhere.
"""

from pathlib import Path
from typing import Optional
from pipeline.config import PipelineConfig


class PathManager:
    """Manage all pipeline paths with YOLO structure enforcement.

    All data directories follow YOLO format:
    - data/working/images/ and data/working/labels/
    - data/verified/images/ and data/verified/labels/
    - data/eval/images/ and data/eval/labels/
    - data/test/images/ and data/test/labels/

    No flat directory structures are supported.
    """

    def __init__(self, root: Path, config: PipelineConfig):
        """Initialize path manager.

        Args:
            root: Pipeline root directory
            config: Pipeline configuration
        """
        self.root = Path(root)
        self.config = config

    # Data directory paths
    def working_dir(self) -> Path:
        """Get working directory path."""
        return self.root / "data" / "working"

    def working_images(self) -> Path:
        """Get working images directory path."""
        return self.working_dir() / "images"

    def working_labels(self) -> Path:
        """Get working labels directory path."""
        return self.working_dir() / "labels"

    def verified_dir(self) -> Path:
        """Get verified directory path."""
        return self.root / "data" / "verified"

    def verified_images(self) -> Path:
        """Get verified images directory path."""
        return self.verified_dir() / "images"

    def verified_labels(self) -> Path:
        """Get verified labels directory path."""
        return self.verified_dir() / "labels"

    def eval_dir(self) -> Path:
        """Get eval directory path."""
        return self.root / "data" / "eval"

    def eval_images(self) -> Path:
        """Get eval images directory path."""
        return self.eval_dir() / "images"

    def eval_labels(self) -> Path:
        """Get eval labels directory path."""
        return self.eval_dir() / "labels"

    def test_dir(self) -> Path:
        """Get test directory path."""
        return self.root / "data" / "test"

    def test_images(self) -> Path:
        """Get test images directory path."""
        return self.test_dir() / "images"

    def test_labels(self) -> Path:
        """Get test labels directory path."""
        return self.test_dir() / "labels"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_paths.py::test_path_manager_working_paths -v
pytest tests/test_paths.py::test_path_manager_verified_paths -v
pytest tests/test_paths.py::test_path_manager_eval_and_test_paths -v
```

Expected output: All tests PASS

**Step 5: Commit**

```bash
git add pipeline/paths.py tests/test_paths.py
git commit -m "feat: add PathManager for data directory paths

- Centralized path management with YOLO structure enforcement
- All data dirs return images/ and labels/ subdirectories
- Tests verify correct path construction"
```

---

### Task 2: Add Manifest and Model Path Methods to PathManager

**Goal:** Add methods for manifest files, model paths, and config paths.

**Files:**
- Modify: `pipeline/paths.py`
- Modify: `tests/test_paths.py`

**Step 1: Write the failing test**

Add to `tests/test_paths.py`:

```python
def test_path_manager_manifest_paths(tmp_path):
    """Test PathManager returns correct manifest paths."""
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

    # Test manifest paths
    assert paths.splits_dir() == tmp_path / "data" / "splits"
    assert paths.train_manifest() == tmp_path / "data" / "splits" / "train.txt"
    assert paths.eval_manifest() == tmp_path / "data" / "splits" / "eval.txt"


def test_path_manager_model_paths(tmp_path):
    """Test PathManager returns correct model paths."""
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

    # Test model paths
    assert paths.active_model() == tmp_path / "models" / "active" / "best.pt"
    assert paths.checkpoint_dir() == tmp_path / "models" / "checkpoints"
    assert paths.deployed_dir() == tmp_path / "models" / "deployed"


def test_path_manager_config_and_log_paths(tmp_path):
    """Test PathManager returns correct config and log paths."""
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

    # Test config paths
    assert paths.data_yaml() == tmp_path / "data" / "data.yaml"
    assert paths.pipeline_config() == tmp_path / "configs" / "pipeline_config.yaml"
    assert paths.yolo_config() == tmp_path / "configs" / "yolo_config.yaml"

    # Test log paths
    assert paths.logs_dir() == tmp_path / "logs"
    assert paths.training_history() == tmp_path / "logs" / "training_history.json"
    assert paths.watcher_log() == tmp_path / "logs" / "watcher.log"
    assert paths.auto_move_log() == tmp_path / "logs" / "auto_move.log"
    assert paths.priority_queue() == tmp_path / "logs" / "priority_queue.txt"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_paths.py::test_path_manager_manifest_paths -v
```

Expected output: `AttributeError: 'PathManager' object has no attribute 'splits_dir'`

**Step 3: Write minimal implementation**

Add to `pipeline/paths.py`:

```python
    # Manifest paths
    def splits_dir(self) -> Path:
        """Get splits directory path."""
        return self.root / "data" / "splits"

    def train_manifest(self) -> Path:
        """Get train manifest file path."""
        return self.splits_dir() / "train.txt"

    def eval_manifest(self) -> Path:
        """Get eval manifest file path."""
        return self.splits_dir() / "eval.txt"

    # Model paths
    def active_model(self) -> Path:
        """Get active model file path."""
        return self.root / "models" / "active" / "best.pt"

    def checkpoint_dir(self) -> Path:
        """Get checkpoints directory path."""
        return self.root / "models" / "checkpoints"

    def deployed_dir(self) -> Path:
        """Get deployed models directory path."""
        return self.root / "models" / "deployed"

    # Config paths
    def data_yaml(self) -> Path:
        """Get data.yaml file path."""
        return self.root / "data" / "data.yaml"

    def pipeline_config(self) -> Path:
        """Get pipeline config file path."""
        return self.root / "configs" / "pipeline_config.yaml"

    def yolo_config(self) -> Path:
        """Get YOLO config file path."""
        return self.root / "configs" / "yolo_config.yaml"

    # Log paths
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        return self.root / "logs"

    def training_history(self) -> Path:
        """Get training history file path."""
        return self.logs_dir() / "training_history.json"

    def watcher_log(self) -> Path:
        """Get watcher log file path."""
        return self.logs_dir() / "watcher.log"

    def auto_move_log(self) -> Path:
        """Get auto-move log file path."""
        return self.logs_dir() / "auto_move.log"

    def priority_queue(self) -> Path:
        """Get priority queue file path."""
        return self.logs_dir() / "priority_queue.txt"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_paths.py::test_path_manager_manifest_paths -v
pytest tests/test_paths.py::test_path_manager_model_paths -v
pytest tests/test_paths.py::test_path_manager_config_and_log_paths -v
```

Expected output: All tests PASS

**Step 5: Commit**

```bash
git add pipeline/paths.py tests/test_paths.py
git commit -m "feat: add manifest, model, config, and log paths to PathManager

- Add splits_dir(), train_manifest(), eval_manifest()
- Add active_model(), checkpoint_dir(), deployed_dir()
- Add data_yaml(), pipeline_config(), yolo_config()
- Add logs_dir() and individual log file paths
- Tests verify all path construction"
```

---

## Phase 2: Foundation - PipelineValidator

### Task 3: Create ValidationResult and HealthReport Dataclasses

**Goal:** Define data structures for validation results.

**Files:**
- Create: `pipeline/validation.py`
- Create: `tests/test_validation.py`

**Step 1: Write the failing test**

Create `tests/test_validation.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_validation.py::test_validation_result_creation -v
```

Expected output: `ModuleNotFoundError: No module named 'pipeline.validation'`

**Step 3: Write minimal implementation**

Create `pipeline/validation.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_validation.py -v
```

Expected output: All tests PASS

**Step 5: Commit**

```bash
git add pipeline/validation.py tests/test_validation.py
git commit -m "feat: add ValidationResult and HealthReport dataclasses

- ValidationResult stores status, messages, and details
- HealthReport aggregates all validation checks
- is_healthy() method determines if pipeline can run
- Tests verify dataclass creation and behavior"
```

---

### Task 4: Implement Structure Validation in PipelineValidator

**Goal:** Implement directory structure validation checking for YOLO layout.

**Files:**
- Modify: `pipeline/validation.py`
- Modify: `tests/test_validation.py`

**Step 1: Write the failing test**

Add to `tests/test_validation.py`:

```python
from pipeline.validation import PipelineValidator
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_validation.py::test_validate_structure_pass -v
```

Expected output: `AttributeError: 'PipelineValidator' object has no attribute 'validate_structure'`

**Step 3: Write minimal implementation**

Add to `pipeline/validation.py`:

```python
from pipeline.paths import PathManager


class PipelineValidator:
    """Validate pipeline setup and state.

    Provides comprehensive validation of directory structure, configuration,
    annotations, and models. Used by doctor command and components at startup.
    """

    def __init__(self, paths: PathManager):
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
            (self.paths.checkpoint_dir(), "models/checkpoints/"),
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
                            f"⚠ Found {len(files)} orphaned files in {parent_dir.relative_to(self.paths.root)}"
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
        else:
            status = "pass"
            messages.append("All required directories exist with correct layout")

        return ValidationResult(
            status=status,
            messages=messages,
            details=details
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_validation.py::test_validate_structure_pass -v
pytest tests/test_validation.py::test_validate_structure_missing_directory -v
pytest tests/test_validation.py::test_validate_structure_orphaned_files -v
```

Expected output: All tests PASS

**Step 5: Commit**

```bash
git add pipeline/validation.py tests/test_validation.py
git commit -m "feat: implement structure validation in PipelineValidator

- validate_structure() checks all required directories
- Detects missing directories (error status)
- Detects orphaned files in parent dirs (warning status)
- Returns ValidationResult with detailed messages
- Tests cover pass, error, and warning cases"
```

---

### Task 5: Implement Config and Annotation Validation

**Goal:** Add configuration and annotation format validation methods.

**Files:**
- Modify: `pipeline/validation.py`
- Modify: `tests/test_validation.py`

**Step 1: Write the failing test**

Add to `tests/test_validation.py`:

```python
import yaml


def test_validate_config_pass(setup_test_structure):
    """Test config validation passes with valid config files."""
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_validation.py::test_validate_config_pass -v
```

Expected output: `AttributeError: 'PipelineValidator' object has no attribute 'validate_config'`

**Step 3: Write minimal implementation**

Add to `pipeline/validation.py`:

```python
import yaml


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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_validation.py::test_validate_config_pass -v
pytest tests/test_validation.py::test_validate_config_missing_file -v
pytest tests/test_validation.py::test_validate_annotations_pass -v
pytest tests/test_validation.py::test_validate_annotations_invalid_format -v
```

Expected output: All tests PASS

**Step 5: Commit**

```bash
git add pipeline/validation.py tests/test_validation.py
git commit -m "feat: add config and annotation validation

- validate_config() checks YAML files exist and parse
- validate_annotations() validates YOLO format
- Checks class IDs, coordinates, matching images
- Returns detailed error messages with line numbers
- Tests cover valid and invalid cases"
```

---

### Task 6: Implement Model Validation and Full Health Check

**Goal:** Add model validation and aggregate health check method.

**Files:**
- Modify: `pipeline/validation.py`
- Modify: `tests/test_validation.py`

**Step 1: Write the failing test**

Add to `tests/test_validation.py`:

```python
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


def test_full_health_check(setup_test_structure):
    """Test full_health_check aggregates all validation checks."""
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_validation.py::test_validate_model_pass -v
```

Expected output: `AttributeError: 'PipelineValidator' object has no attribute 'validate_model'`

**Step 3: Write minimal implementation**

Add to `pipeline/validation.py`:

```python
    def validate_model(self) -> ValidationResult:
        """Validate YOLO model file.

        Validates:
        - .pt file exists (if present)
        - File loads with YOLO()

        Returns:
            ValidationResult with status and messages
        """
        messages = []

        model_path = self.paths.active_model()

        if not model_path.exists():
            return ValidationResult(
                status="pass",
                messages=["ℹ No active model found (first run)"],
                details={}
            )

        # Try to load model
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            messages.append(f"✓ Active model loads successfully: {model_path.name}")

            return ValidationResult(
                status="pass",
                messages=messages,
                details={"model_path": str(model_path)}
            )
        except Exception as e:
            return ValidationResult(
                status="error",
                messages=[f"✗ Active model failed to load: {e}"],
                details={"model_path": str(model_path)}
            )

    def full_health_check(self) -> HealthReport:
        """Run all validation checks and generate health report.

        Runs:
        - Structure validation
        - Config validation
        - Annotation validation (working + verified)
        - Model validation

        Returns:
            HealthReport with all check results
        """
        # Run all validations
        structure = self.validate_structure()
        config = self.validate_config()

        # Validate annotations in working and verified
        working_annotations = self.validate_annotations(self.paths.working_dir())
        verified_annotations = self.validate_annotations(self.paths.verified_dir())

        # Combine annotation results
        ann_messages = []
        ann_details = {}
        ann_status = "pass"

        if working_annotations.status == "pass":
            ann_messages.extend([f"Working: {msg}" for msg in working_annotations.messages])
        elif working_annotations.status == "warning":
            ann_messages.extend([f"Working: {msg}" for msg in working_annotations.messages])
            ann_status = "warning"
        else:
            ann_messages.extend([f"Working: {msg}" for msg in working_annotations.messages])
            ann_status = "error"

        if verified_annotations.status == "pass":
            ann_messages.extend([f"Verified: {msg}" for msg in verified_annotations.messages])
        elif verified_annotations.status == "warning":
            ann_messages.extend([f"Verified: {msg}" for msg in verified_annotations.messages])
            if ann_status == "pass":
                ann_status = "warning"
        else:
            ann_messages.extend([f"Verified: {msg}" for msg in verified_annotations.messages])
            ann_status = "error"

        ann_details.update(working_annotations.details)
        ann_details.update(verified_annotations.details)

        annotations = ValidationResult(
            status=ann_status,
            messages=ann_messages,
            details=ann_details
        )

        models = self.validate_model()

        # Determine overall status
        statuses = [structure.status, config.status, annotations.status, models.status]

        if "error" in statuses:
            overall_status = "errors"
        elif "warning" in statuses:
            overall_status = "warnings"
        else:
            overall_status = "healthy"

        return HealthReport(
            structure=structure,
            config=config,
            annotations=annotations,
            models=models,
            overall_status=overall_status
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_validation.py::test_validate_model_pass -v
pytest tests/test_validation.py::test_full_health_check -v
```

Expected output: All tests PASS

**Step 5: Commit**

```bash
git add pipeline/validation.py tests/test_validation.py
git commit -m "feat: add model validation and full health check

- validate_model() checks if active model loads
- full_health_check() aggregates all validations
- Returns HealthReport with overall status
- Combines working and verified annotation checks
- Tests verify model validation and health report"
```

---

## Phase 3: Manifest-Based Splits

### Task 7: Implement generate_manifests() Function

**Goal:** Generate train.txt and eval.txt manifest files from verified dataset.

**Files:**
- Modify: `pipeline/data_utils.py`
- Create: `tests/test_manifests.py`

**Step 1: Write the failing test**

Create `tests/test_manifests.py`:

```python
"""Tests for manifest generation."""

import pytest
from pathlib import Path
from pipeline.data_utils import generate_manifests
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


@pytest.fixture
def setup_verified_data(tmp_path):
    """Create verified dataset with images and labels."""
    verified_labels = tmp_path / "data" / "verified" / "labels"
    verified_images = tmp_path / "data" / "verified" / "images"
    verified_labels.mkdir(parents=True)
    verified_images.mkdir(parents=True)

    # Create 10 label/image pairs
    for i in range(10):
        (verified_labels / f"img{i:03d}.txt").write_text(f"0 0.5 0.5 0.1 0.1")
        (verified_images / f"img{i:03d}.png").touch()

    return tmp_path


def test_generate_manifests_creates_files(setup_verified_data):
    """Test generate_manifests creates train.txt and eval.txt."""
    root = setup_verified_data

    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=10,
        eval_split_ratio=0.2,  # 80/20 split
        stratify=False,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(root, config)

    train_count, eval_count = generate_manifests(paths, config)

    # Check manifests exist
    assert paths.train_manifest().exists()
    assert paths.eval_manifest().exists()

    # Check counts (8 train, 2 eval for 80/20 split)
    assert train_count == 8
    assert eval_count == 2


def test_generate_manifests_correct_split_ratio(setup_verified_data):
    """Test generate_manifests maintains correct split ratio."""
    root = setup_verified_data

    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=10,
        eval_split_ratio=0.3,  # 70/30 split
        stratify=False,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(root, config)

    train_count, eval_count = generate_manifests(paths, config)

    # 10 images: 70% = 7 train, 30% = 3 eval
    assert train_count == 7
    assert eval_count == 3


def test_generate_manifests_content_format(setup_verified_data):
    """Test manifest files contain correct image paths."""
    root = setup_verified_data

    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=10,
        eval_split_ratio=0.2,
        stratify=False,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(root, config)

    generate_manifests(paths, config)

    # Read manifests
    with open(paths.train_manifest()) as f:
        train_paths = [line.strip() for line in f]

    with open(paths.eval_manifest()) as f:
        eval_paths = [line.strip() for line in f]

    # Check format: should be paths to images
    for path_str in train_paths + eval_paths:
        path = Path(path_str)
        assert path.suffix in ['.png', '.jpg', '.jpeg']
        assert "verified/images" in str(path)

    # Check no overlap
    assert len(set(train_paths) & set(eval_paths)) == 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_manifests.py::test_generate_manifests_creates_files -v
```

Expected output: `ImportError: cannot import name 'generate_manifests' from 'pipeline.data_utils'`

**Step 3: Write minimal implementation**

Add to `pipeline/data_utils.py`:

```python
import random
from typing import Tuple
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


def generate_manifests(
    paths: PathManager,
    config: PipelineConfig
) -> Tuple[int, int]:
    """Generate train and eval manifests for current verified dataset.

    Re-splits the entire verified dataset every time this is called.
    Maintains config.eval_split_ratio (e.g., 85% train, 15% eval).

    All files remain in verified/ directory - manifests just reference them.

    Args:
        paths: PathManager instance
        config: PipelineConfig with split settings

    Returns:
        (train_count, eval_count) tuple

    Raises:
        ValueError: If not enough labels in verified/
    """
    # Find all labels in verified/labels/
    all_labels = list(paths.verified_labels().glob("*.txt"))

    if len(all_labels) < config.min_train_images:
        raise ValueError(
            f"Need at least {config.min_train_images} labels, "
            f"found {len(all_labels)}"
        )

    # Simple random split (stratified not implemented yet)
    random.shuffle(all_labels)
    split_idx = int(len(all_labels) * (1 - config.eval_split_ratio))
    train_labels = all_labels[:split_idx]
    eval_labels = all_labels[split_idx:]

    # Convert label paths to image paths
    def label_to_image_path(label_path: Path) -> Path:
        """Convert label path to corresponding image path."""
        images_dir = paths.verified_images()

        # Try different image extensions
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            image_path = images_dir / f"{label_path.stem}{ext}"
            if image_path.exists():
                return image_path

        # Fallback: assume .png
        return images_dir / f"{label_path.stem}.png"

    train_images = [label_to_image_path(lbl) for lbl in train_labels]
    eval_images = [label_to_image_path(lbl) for lbl in eval_labels]

    # Write manifests (absolute paths)
    paths.splits_dir().mkdir(parents=True, exist_ok=True)

    with open(paths.train_manifest(), 'w') as f:
        f.write('\n'.join(str(img) for img in train_images))

    with open(paths.eval_manifest(), 'w') as f:
        f.write('\n'.join(str(img) for img in eval_images))

    return len(train_images), len(eval_images)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_manifests.py -v
```

Expected output: All tests PASS

**Step 5: Commit**

```bash
git add pipeline/data_utils.py tests/test_manifests.py
git commit -m "feat: implement generate_manifests for train/eval splits

- Generates train.txt and eval.txt from verified/ dataset
- Maintains configured split ratio
- Files stay in verified/, manifests reference them
- Simple random split (no stratification yet)
- Tests verify manifest creation and format"
```

---

Due to length constraints, I'll create the implementation plan up to this point. The plan would continue with:

**Phase 4: Smart Resume Training** (Tasks 8-10)
**Phase 5: Atomic File Moves** (Tasks 11-13)
**Phase 6: Doctor Command** (Tasks 14-16)
**Phase 7: Process Manager** (Tasks 17-21)
**Phase 8: Component Refactoring** (Tasks 22-26)
**Phase 9: Migration Script** (Task 27)
**Phase 10: Integration Testing** (Tasks 28-30)

Each following the same TDD pattern: write test → run (fail) → implement → run (pass) → commit.

Would you like me to:
1. Continue with the remaining phases in this document?
2. Split into multiple documents?
3. Or is this level of detail sufficient for you to understand the pattern and continue?

---

## Phase 4: Smart Resume Training

### Task 8: Implement init_model() with Smart Resume

**Goal:** Create model initialization that resumes from active model when available.

**Files:**
- Modify: `pipeline/train.py`
- Create: `tests/test_train_resume.py`

**Implementation:** Add `init_model()` function that checks for `paths.active_model()`, validates it loads, and falls back to pretrained model if not available or corrupted. Add `--from-scratch` CLI flag to force fresh start.

**Commit:** "feat: add smart resume training from active model"

---

## Phase 5: Atomic File Moves

### Task 9: Implement atomic_move_pair() Function

**Goal:** Implement copy-then-rename pattern for atomic file moves.

**Files:**
- Modify: `scripts/auto_move_verified.py`
- Create: `tests/test_atomic_moves.py`

**Step 1: Write failing test** - Test successful move and rollback on failure
**Step 2: Implement** - Copy both files with `.tmp` extension, verify, rename atomically, delete originals
**Step 3: Add cleanup** - `cleanup_tmp_files()` removes stale `.tmp` files on startup

**Commit:** "feat: implement atomic file moves for data integrity"

---

## Phase 6: Doctor Command

### Task 10: Create Doctor Command CLI

**Goal:** Implement `yolo-pipeline doctor` command for comprehensive health checks.

**Files:**
- Create: `pipeline/doctor.py`
- Modify: `setup.py`

**Implementation:**
```python
def doctor_command():
    """Run comprehensive pipeline health check."""
    paths = PathManager(Path.cwd(), config)
    validator = PipelineValidator(paths)
    report = validator.full_health_check()
    print_health_report(report)
    sys.exit(0 if report.is_healthy() else 1)
```

Add entry point to `setup.py`:
```python
"yolo-pipeline-doctor=pipeline.doctor:main",
```

**Commit:** "feat: add doctor command for pipeline health checks"

---

## Phase 7: Process Manager

### Task 11: Create ProcessManager Class

**Goal:** Implement process manager for running all services with graceful lifecycle.

**Files:**
- Create: `pipeline/process_manager.py`
- Create: `tests/test_process_manager.py`

**Implementation:**
```python
class ProcessManager:
    def __init__(self, paths: PathManager):
        self.paths = paths
        self.processes = []
    
    def run(self, debug=False, no_doctor=False):
        # 1. Run doctor check
        # 2. Register signal handlers
        # 3. Launch subprocesses (auto_move, watcher, monitor)
        # 4. Stream monitor output
        # 5. Handle shutdown gracefully
```

**Commit:** "feat: add ProcessManager for single-command workflow"

### Task 12: Create yolo-pipeline run Command

**Goal:** Add CLI entry point for process manager.

**Files:**
- Modify: `pipeline/process_manager.py`
- Modify: `setup.py`

Add `main()` function and entry point:
```python
"yolo-pipeline-run=pipeline.process_manager:main",
```

**Commit:** "feat: add yolo-pipeline run command"

---

## Phase 8: Component Refactoring

### Task 13: Update train.py to Use PathManager and Manifests

**Goal:** Refactor training pipeline to use PathManager and generate manifests.

**Files:**
- Modify: `pipeline/train.py`

**Changes:**
1. Add `paths` parameter to `train_model()`
2. Replace hardcoded paths with `paths.verified_labels()` etc.
3. Call `generate_manifests()` before training
4. Update `create_data_yaml()` to use manifest paths
5. Use `init_model()` for smart resume

**Commit:** "refactor: update train.py to use PathManager and manifests"

### Task 14: Update watcher.py to Use PathManager

**Goal:** Refactor watcher to use PathManager and validate on startup.

**Files:**
- Modify: `pipeline/watcher.py`

**Changes:**
1. Add `paths` parameter to `__init__()`
2. Replace `self.verified_dir` with `paths.verified_dir()`
3. Add structure validation at startup
4. Pass `paths` to `train_model()`

**Commit:** "refactor: update watcher.py to use PathManager"

### Task 15: Update monitor.py and data_utils.py

**Goal:** Refactor monitor and data_utils to use PathManager.

**Files:**
- Modify: `pipeline/monitor.py`
- Modify: `pipeline/data_utils.py`

Apply same pattern - replace hardcoded paths with PathManager methods.

**Commits:**
- "refactor: update monitor.py to use PathManager"
- "refactor: update data_utils.py to use PathManager"

---

## Phase 9: Migration Helper

### Task 16: Create Migration Script

**Goal:** Create script to migrate from flat to YOLO layout.

**Files:**
- Create: `scripts/migrate_to_yolo_layout.py`

**Implementation:**
```python
def migrate_to_yolo_layout(root: Path):
    for data_dir in ["working", "verified", "eval", "test"]:
        # Create images/ and labels/ subdirs
        # Move *.txt to labels/
        # Move *.png/*.jpg to images/
```

**Commit:** "feat: add migration script for YOLO layout"

---

## Phase 10: Integration Testing

### Task 17: End-to-End Doctor Test

**Goal:** Test doctor command detects and reports issues correctly.

**Files:**
- Create: `tests/integration/test_doctor_integration.py`

Test scenarios:
- Valid structure → healthy report
- Missing directories → error report
- Invalid annotations → error report
- Orphaned files → warning report

**Commit:** "test: add integration tests for doctor command"

### Task 18: End-to-End Process Manager Test

**Goal:** Test process manager launches and shuts down cleanly.

**Files:**
- Create: `tests/integration/test_process_manager_integration.py`

Mock subprocesses, test:
- Services launch
- Graceful shutdown on SIGINT
- Doctor check runs first

**Commit:** "test: add integration tests for process manager"

### Task 19: End-to-End Manifest Training Test

**Goal:** Test full training flow with manifests.

**Files:**
- Create: `tests/integration/test_manifest_training.py`

Test:
1. Generate manifests from verified/
2. Create data.yaml
3. Verify YOLO can read manifests
4. Files remain in verified/

**Commit:** "test: add integration tests for manifest-based training"

---

## Summary

**Total Tasks:** 19
**Estimated Time:** 8-10 hours
**Lines of Code:** ~2000-2500 (including tests)

**Key Milestones:**
1. Foundation complete (Tasks 1-6) - PathManager + Validator working
2. Manifests working (Task 7) - Can generate train/eval splits
3. Features complete (Tasks 8-12) - Doctor, Process Manager, Atomic Moves
4. Refactoring done (Tasks 13-15) - All components use foundation
5. Testing done (Tasks 16-19) - Integration tests pass

**Testing Strategy:**
- Unit tests per task (TDD)
- Integration tests at end
- Manual testing with real dataset
- Run full test suite: `pytest tests/ -v --cov=pipeline`

**Rollout Plan:**
1. Merge to feature branch
2. Test with real 1332-image dataset
3. Update documentation (README, QUICKSTART)
4. Merge to main
5. Tag release v0.2.0

---

## Execution Options

Plan complete and saved to `docs/plans/2026-03-03-pipeline-improvements-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach would you like?
