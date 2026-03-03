# YOLO Pipeline Improvements - Design Document

**Date:** 2026-03-03
**Status:** Approved
**Implementation Approach:** Core Refactor + Features

## Overview

This design addresses critical robustness and usability improvements to the YOLO iterative training pipeline. After fixing directory structure bugs, we're standardizing on YOLO layout enforcement, adding comprehensive validation, simplifying the user experience, and improving data integrity.

## Goals

1. **Unify dataset layout** - Enforce YOLO images/labels structure everywhere
2. **Add preflight validation** - Catch setup errors before wasting GPU time
3. **Simplify workflow** - Replace 4-terminal setup with single command
4. **Improve data integrity** - Atomic file moves prevent inconsistent state
5. **Optimize training** - Resume from active model, use manifest-based splits

## Non-Goals

- Priority queue integration (deferred)
- Systemd service installation (process manager only)
- Backward compatibility with flat directory structure

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│                    Features Layer                        │
│  - Doctor Command                                        │
│  - Process Manager (yolo-pipeline run)                   │
│  - Manifest-based Splits                                 │
│  - Smart Resume Training                                 │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ uses
                          │
┌─────────────────────────────────────────────────────────┐
│                  Component Layer                         │
│  - watcher.py, train.py, monitor.py, data_utils.py      │
│  - auto_move_verified.py                                 │
│  - All refactored to use PathManager                     │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ uses
                          │
┌─────────────────────────────────────────────────────────┐
│                  Foundation Layer                        │
│  - PathManager (paths.py)                                │
│  - PipelineValidator (validation.py)                     │
│  - Single source of truth for paths and validation      │
└─────────────────────────────────────────────────────────┘
```

**Dependency flow:** Foundation → Components → Features
**Validation:** Each layer validates inputs, fails fast with clear messages

## Foundation Layer

### PathManager (`pipeline/paths.py`)

**Purpose:** Single source of truth for all directory paths with YOLO structure enforcement.

**Class Interface:**
```python
class PathManager:
    """Manage all pipeline paths with YOLO structure enforcement."""

    def __init__(self, root: Path, config: PipelineConfig):
        """Initialize path manager.

        Args:
            root: Pipeline root directory
            config: Pipeline configuration
        """
        self.root = root
        self.config = config

    # Core data paths
    def working_dir(self) -> Path:
        return self.root / "data" / "working"

    def working_images(self) -> Path:
        return self.working_dir() / "images"

    def working_labels(self) -> Path:
        return self.working_dir() / "labels"

    def verified_dir(self) -> Path:
        return self.root / "data" / "verified"

    def verified_images(self) -> Path:
        return self.verified_dir() / "images"

    def verified_labels(self) -> Path:
        return self.verified_dir() / "labels"

    def eval_dir(self) -> Path:
        return self.root / "data" / "eval"

    def eval_images(self) -> Path:
        return self.eval_dir() / "images"

    def eval_labels(self) -> Path:
        return self.eval_dir() / "labels"

    def test_dir(self) -> Path:
        return self.root / "data" / "test"

    def test_images(self) -> Path:
        return self.test_dir() / "images"

    def test_labels(self) -> Path:
        return self.test_dir() / "labels"

    # Manifest paths
    def splits_dir(self) -> Path:
        return self.root / "data" / "splits"

    def train_manifest(self) -> Path:
        return self.splits_dir() / "train.txt"

    def eval_manifest(self) -> Path:
        return self.splits_dir() / "eval.txt"

    # Model paths
    def active_model(self) -> Path:
        return self.root / "models" / "active" / "best.pt"

    def checkpoint_dir(self) -> Path:
        return self.root / "models" / "checkpoints"

    def deployed_dir(self) -> Path:
        return self.root / "models" / "deployed"

    # Config paths
    def data_yaml(self) -> Path:
        return self.root / "data" / "data.yaml"

    def pipeline_config(self) -> Path:
        return self.root / "configs" / "pipeline_config.yaml"

    def yolo_config(self) -> Path:
        return self.root / "configs" / "yolo_config.yaml"

    # Log paths
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def training_history(self) -> Path:
        return self.logs_dir() / "training_history.json"

    def watcher_log(self) -> Path:
        return self.logs_dir() / "watcher.log"

    def auto_move_log(self) -> Path:
        return self.logs_dir() / "auto_move.log"

    def priority_queue(self) -> Path:
        return self.logs_dir() / "priority_queue.txt"
```

**Design Decisions:**
- All methods return `Path` objects, never strings
- Enforces `images/` and `labels/` subdirectories everywhere
- No support for flat directory structure (hard enforcement)
- Configuration-driven (reads root from config or uses cwd)
- Methods named for clarity: `working_images()` not `working()/images/`

### PipelineValidator (`pipeline/validation.py`)

**Purpose:** Comprehensive validation of pipeline state with detailed error reporting.

**Class Interface:**
```python
@dataclass
class ValidationResult:
    """Result of a validation check."""
    status: str  # "pass", "warning", "error"
    messages: List[str]
    details: Dict[str, Any]

@dataclass
class HealthReport:
    """Complete pipeline health check report."""
    structure: ValidationResult
    config: ValidationResult
    annotations: ValidationResult
    models: ValidationResult
    overall_status: str  # "healthy", "warnings", "errors"

    def is_healthy(self) -> bool:
        return self.overall_status in ["healthy", "warnings"]

class PipelineValidator:
    """Validate pipeline setup and state."""

    def __init__(self, paths: PathManager):
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

    def validate_config(self) -> ValidationResult:
        """Validate configuration files.

        Validates:
        - pipeline_config.yaml exists and parses
        - yolo_config.yaml exists and parses
        - Required fields present
        - Class count consistency

        Returns:
            ValidationResult with status and messages
        """

    def validate_annotations(self, dir_path: Path) -> ValidationResult:
        """Validate YOLO annotation format in directory.

        Validates:
        - All .txt files parse as YOLO format
        - Class IDs within valid range
        - Coordinates normalized to [0, 1]
        - Matching images exist for labels
        - No duplicate filenames

        Args:
            dir_path: Directory containing images/ and labels/ subdirs

        Returns:
            ValidationResult with status and messages
        """

    def validate_model(self, model_path: Path) -> ValidationResult:
        """Validate YOLO model file.

        Validates:
        - .pt file exists
        - File loads with YOLO()
        - Model architecture matches expected

        Args:
            model_path: Path to .pt model file

        Returns:
            ValidationResult with status and messages
        """

    def full_health_check(self) -> HealthReport:
        """Run all validation checks and generate health report.

        Runs:
        - Structure validation
        - Config validation
        - Annotation validation (working + verified)
        - Model validation (if active model exists)

        Returns:
            HealthReport with all check results
        """
```

**Design Decisions:**
- `ValidationResult` separates status from messages (machine + human readable)
- Three status levels: "pass" (all good), "warning" (can continue), "error" (must fix)
- `full_health_check()` aggregates all checks into single report
- Validation is read-only (no side effects, no auto-fix)
- Clear, actionable error messages guide users to fixes

## Component Layer Updates

### Updated Components

All existing components refactored to use `PathManager`:

**`pipeline/watcher.py`**
- Use `paths.verified_labels()` instead of hardcoded paths
- Call `validator.validate_structure()` at startup
- Pass `PathManager` instance to training

**`pipeline/train.py`**
- Use `paths` for all directory access
- Generate manifests instead of moving files
- Smart resume: check `paths.active_model()` before init
- Call validator before training starts

**`pipeline/monitor.py`**
- Use `paths` for file counting
- Display validation status in dashboard

**`pipeline/data_utils.py`**
- Replace hardcoded paths with `PathManager`
- Update `sample_eval_set()` to generate manifests
- Remove file moving logic (keep everything in verified/)

**`scripts/auto_move_verified.py`**
- Use `PathManager` for source/destination paths
- Implement atomic move pattern
- Validate structure at startup

### Constructor Pattern

All components follow same pattern:
```python
class Component:
    def __init__(self, paths: PathManager, config: Config):
        self.paths = paths
        self.config = config

        # Validate structure at initialization
        validator = PipelineValidator(paths)
        result = validator.validate_structure()
        if result.status == "error":
            raise ValueError("\n".join(result.messages))

        # Component initialization continues...
```

This ensures every component operates on valid structure.

## Manifest-Based Splits

### Motivation

**Current problem:** `sample_eval_set()` moves files from `verified/` to `eval/`, causing:
- Watcher gets confused (file counts change unexpectedly)
- `verified/` shrinks, loses training data
- Can't easily regenerate eval set

**Solution:** Keep all files in `verified/`, use manifests to define splits.

### Implementation

**Manifest generation** (`pipeline/data_utils.py`):
```python
def generate_manifests(
    paths: PathManager,
    config: PipelineConfig
) -> Tuple[int, int]:
    """Generate train and eval manifests for current verified dataset.

    Re-splits the entire verified dataset every time this is called.
    Maintains config.eval_split_ratio (e.g., 85% train, 15% eval).

    Args:
        paths: PathManager instance
        config: PipelineConfig with split settings

    Returns:
        (train_count, eval_count) tuple
    """
    # 1. Find all labels in verified/labels/
    all_labels = list(paths.verified_labels().glob("*.txt"))

    if len(all_labels) < config.min_train_images:
        raise ValueError(f"Need at least {config.min_train_images} labels")

    # 2. Stratified split if enabled
    if config.stratify:
        train_labels, eval_labels = stratified_split(
            all_labels,
            ratio=config.eval_split_ratio,
            num_classes=len(config.classes)
        )
    else:
        # Simple random split
        random.shuffle(all_labels)
        split_idx = int(len(all_labels) * (1 - config.eval_split_ratio))
        train_labels = all_labels[:split_idx]
        eval_labels = all_labels[split_idx:]

    # 3. Convert to image paths
    train_images = [label_to_image_path(lbl, paths.verified_images())
                    for lbl in train_labels]
    eval_images = [label_to_image_path(lbl, paths.verified_images())
                   for lbl in eval_labels]

    # 4. Write manifests (absolute paths)
    paths.splits_dir().mkdir(parents=True, exist_ok=True)

    with open(paths.train_manifest(), 'w') as f:
        f.write('\n'.join(str(img) for img in train_images))

    with open(paths.eval_manifest(), 'w') as f:
        f.write('\n'.join(str(img) for img in eval_images))

    return len(train_images), len(eval_images)
```

**Updated data.yaml** (`pipeline/train.py`):
```python
def create_data_yaml(paths: PathManager, config: PipelineConfig):
    """Create data.yaml pointing to manifest files."""
    data = {
        "path": str(paths.root),
        "train": str(paths.train_manifest()),  # Manifest file
        "val": str(paths.eval_manifest()),     # Manifest file
        "test": str(paths.test_dir()),          # Directory (fixed test set)
        "names": {i: name for i, name in enumerate(config.classes)}
    }

    with open(paths.data_yaml(), 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
```

### Training Flow

1. **Count files:** `len(list(paths.verified_labels().glob("*.txt")))`
2. **Generate manifests:** Call `generate_manifests()` before training
3. **Create data.yaml:** Point to manifest files
4. **Train:** YOLO reads images from paths in manifests
5. **Next iteration:** Regenerate manifests with new verified files

**Key insight:** Every training run gets fresh 85/15 split of current dataset. Eval set grows with dataset, stays representative.

## Smart Resume Training

### Current Behavior

`train.py` always initializes from `yolo_config.model` (e.g., `"yolo11n.pt"`), starting from scratch each iteration.

### New Behavior

**Smart resume logic**:
```python
def init_model(
    paths: PathManager,
    config: YOLOConfig,
    from_scratch: bool = False
) -> YOLO:
    """Initialize YOLO model with smart resume.

    Args:
        paths: PathManager instance
        config: YOLO configuration
        from_scratch: If True, ignore active model and start fresh

    Returns:
        Initialized YOLO model
    """
    active_model = paths.active_model()

    # Check if we should resume
    if from_scratch:
        logger.info(f"Starting from scratch: {config.model}")
        return YOLO(config.model)

    if active_model.exists():
        # Validate model loads
        try:
            model = YOLO(str(active_model))
            logger.info(f"Resuming from active model: {active_model}")
            return model
        except Exception as e:
            logger.warning(f"Active model failed to load: {e}")
            logger.info(f"Falling back to: {config.model}")
            return YOLO(config.model)
    else:
        logger.info(f"No active model found, starting from: {config.model}")
        return YOLO(config.model)
```

**Integration:**
```python
def train_model(
    paths: PathManager,
    pipeline_config: PipelineConfig,
    yolo_config: YOLOConfig,
    from_scratch: bool = False
) -> Tuple[str, Path]:
    """Train YOLO model with smart resume."""

    # Initialize model (resume or from scratch)
    model = init_model(paths, yolo_config, from_scratch)

    # Generate manifests
    train_count, eval_count = generate_manifests(paths, pipeline_config)

    # Create data.yaml
    create_data_yaml(paths, pipeline_config)

    # Train (rest of training logic unchanged)
    results = model.train(
        data=str(paths.data_yaml()),
        epochs=yolo_config.epochs,
        # ... other params
    )

    # ... evaluation and promotion logic
```

### Command-Line Interface

- `yolo-pipeline-train` → resume from active model (default)
- `yolo-pipeline-train --from-scratch` → start fresh
- `yolo-pipeline-watch` → auto-triggers use resume by default

### Benefits

1. **Faster convergence:** Each iteration builds on previous model
2. **Better continuity:** Model evolves with dataset, not restarted
3. **Flexibility:** Can still reset with `--from-scratch`
4. **Fail-safe:** Falls back to pretrained if active model corrupted

## Doctor Command

### Purpose

Comprehensive preflight validation that catches 90% of setup errors before wasting GPU time.

### Command: `yolo-pipeline doctor`

**Implementation:**
```python
def doctor_command():
    """Run comprehensive pipeline health check."""

    # Initialize PathManager and Validator
    root = Path.cwd()
    config = PipelineConfig.from_yaml(root / "configs" / "pipeline_config.yaml")
    paths = PathManager(root, config)
    validator = PipelineValidator(paths)

    # Run full health check
    print("Running pipeline health check...\n")
    report = validator.full_health_check()

    # Display results
    print_health_report(report)

    # Exit with appropriate code
    if report.overall_status == "errors":
        print("\n❌ Pipeline has errors - fix issues before running")
        sys.exit(1)
    elif report.overall_status == "warnings":
        print("\n⚠️  Pipeline has warnings but can run")
        sys.exit(0)
    else:
        print("\n✅ Pipeline is healthy and ready to run")
        sys.exit(0)
```

### Output Format

```
Running pipeline health check...

Directory Structure
  ✓ data/working/images/ exists
  ✓ data/working/labels/ exists
  ✓ data/verified/images/ exists
  ✓ data/verified/labels/ exists
  ✓ data/test/images/ exists
  ✓ data/test/labels/ exists
  ✓ data/splits/ exists
  ✓ models/active/ exists
  ✓ logs/ exists
  ✗ Found orphaned files in data/working/
    → Move files into images/ and labels/ subdirectories

Configuration
  ✓ configs/pipeline_config.yaml exists and valid
  ✓ configs/yolo_config.yaml exists and valid
  ✓ Class count matches between configs (3 classes)

Annotations
  ✓ Scanned 236 labels in data/working/labels/
  ✓ All YOLO format valid
  ⚠ Warning: 3 labels missing matching images
    → img045.txt, img102.txt, img203.txt
  ✓ Scanned 1332 labels in data/verified/labels/
  ✓ All normalized coordinates in range [0,1]
  ✓ Class IDs within valid range [0-2]

Models
  ✓ Active model exists: models/active/best.pt
  ✓ Model loads successfully
  ℹ Training history: 3 iterations completed

System
  ✓ Python 3.9.12
  ✓ ultralytics 8.3.0
  ✓ torch 2.0.1+cu118
  ✓ CUDA available: 2x NVIDIA RTX A5500

Ready to Run
  ✗ Pipeline has errors - fix issues before running
```

### Integration

**All pipeline commands run doctor check:**
```python
def run_with_doctor_check(func):
    """Decorator to run doctor check before command."""
    def wrapper(*args, **kwargs):
        if not args.no_doctor:  # Allow skipping with --no-doctor
            report = run_doctor_check()
            if not report.is_healthy():
                print("❌ Doctor check failed, fix issues first")
                sys.exit(1)
        return func(*args, **kwargs)
    return wrapper

@run_with_doctor_check
def main():
    # Command implementation
    pass
```

## Process Manager

### Purpose

Replace 4-terminal workflow with single command that manages all services.

### Command: `yolo-pipeline run`

**Implementation:**
```python
import subprocess
import signal
import sys
from typing import List

class ProcessManager:
    """Manage pipeline subprocesses with graceful lifecycle."""

    def __init__(self, paths: PathManager):
        self.paths = paths
        self.processes: List[subprocess.Popen] = []

    def launch_subprocess(self, cmd: List[str], name: str) -> subprocess.Popen:
        """Launch subprocess with logging."""
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.processes.append(proc)
        print(f"  ✓ {name} started (PID {proc.pid})")
        return proc

    def run(self, debug: bool = False, no_doctor: bool = False):
        """Run all pipeline services."""

        # 1. Doctor check
        if not no_doctor:
            print("Running doctor check...")
            report = run_doctor_check()
            if not report.is_healthy():
                print("❌ Doctor check failed")
                sys.exit(1)
            print("✓ All checks passed\n")

        # 2. Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # 3. Launch services
        print("Starting pipeline services...")

        auto_move = self.launch_subprocess(
            ["python", "scripts/auto_move_verified.py"],
            "Auto-move watcher"
        )

        watcher = self.launch_subprocess(
            ["yolo-pipeline-watch"],
            "Training watcher"
        )

        monitor = self.launch_subprocess(
            ["yolo-pipeline-monitor", "--loop"],
            "Monitor"
        )

        print()

        # 4. Display monitor output
        if debug:
            # Debug mode: tail all logs
            self._tail_all_logs()
        else:
            # Normal mode: show monitor output
            self._stream_monitor(monitor)

    def _stream_monitor(self, monitor_proc: subprocess.Popen):
        """Stream monitor output to console."""
        print("📊 Pipeline Status (refreshes every 5s)")
        print("Press Ctrl+C to stop all services\n")

        try:
            for line in monitor_proc.stdout:
                print(line, end='')
        except KeyboardInterrupt:
            pass  # Handled by signal handler

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        print("\n\n🛑 Shutting down pipeline...")

        for proc in self.processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"  ✓ Process {proc.pid} stopped")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  ⚠ Process {proc.pid} force killed")

        print("\n✓ All services stopped")
        sys.exit(0)

def run_command(debug: bool = False, no_doctor: bool = False):
    """Entry point for yolo-pipeline run."""
    root = Path.cwd()
    config = PipelineConfig.from_yaml(root / "configs" / "pipeline_config.yaml")
    paths = PathManager(root, config)

    manager = ProcessManager(paths)
    manager.run(debug=debug, no_doctor=no_doctor)
```

### User Experience

```bash
$ yolo-pipeline run
Running doctor check...
✓ All checks passed

Starting pipeline services...
  ✓ Auto-move watcher started (PID 12345)
  ✓ Training watcher started (PID 12346)
  ✓ Monitor started (PID 12347)

📊 Pipeline Status (refreshes every 5s)
Press Ctrl+C to stop all services

YOLO Iterative Pipeline Status
Active Model: v003 (2026-03-03)
Eval Metrics: mAP50: 0.851 (+0.023)  F1: 0.817 (+0.015)
Test Metrics: mAP50: 0.823  F1: 0.795

Data Progress:
  Verified: 1332 / 1500 images  [████████░░] 88.8%
  Working: 236 images

Pipeline Status: HEALTHY
  File Watcher: Running ✓
  Last Training: 2.3 hours ago

^C

🛑 Shutting down pipeline...
  ✓ Process 12345 stopped
  ✓ Process 12346 stopped
  ✓ Process 12347 stopped

✓ All services stopped
```

### Command Options

- `yolo-pipeline run` - Standard mode (monitor output only)
- `yolo-pipeline run --debug` - Debug mode (all subprocess logs)
- `yolo-pipeline run --no-doctor` - Skip doctor check
- `yolo-pipeline run --no-monitor` - Run services without monitor display

### Integration with Existing Commands

Existing commands still work independently:
- `yolo-pipeline-watch` - Run watcher standalone
- `yolo-pipeline-train` - Manual training trigger
- `yolo-pipeline-monitor` - View status

Process manager is convenience wrapper, not replacement.

## Atomic File Moves

### Problem

Current implementation in `auto_move_verified.py`:
```python
# Move label
shutil.move(label_src, label_dst)
# Move image (if this fails, inconsistent state!)
shutil.move(image_src, image_dst)
```

If image move fails (disk full, permissions, crash), you have:
- Label in `verified/labels/` but image in `working/images/`
- Training fails (missing image for label)
- No automatic recovery

### Solution: Copy-Then-Rename

**Implementation:**
```python
def atomic_move_pair(
    label_src: Path,
    image_src: Path,
    label_dst: Path,
    image_dst: Path
) -> None:
    """Atomically move label and image pair.

    Uses copy-then-rename pattern:
    1. Copy both files with .tmp extension
    2. Verify both copies exist
    3. Atomic rename both (filesystem guarantees atomicity)
    4. Delete originals only after successful renames

    If any step fails, no files are moved (consistent state).

    Args:
        label_src: Source label path
        image_src: Source image path
        label_dst: Destination label path
        image_dst: Destination image path

    Raises:
        IOError: If atomic move fails
    """
    label_tmp = label_dst.with_suffix('.txt.tmp')
    image_tmp = image_dst.with_suffix(image_dst.suffix + '.tmp')

    try:
        # Step 1: Copy both files
        shutil.copy2(label_src, label_tmp)
        shutil.copy2(image_src, image_tmp)

        # Step 2: Verify both copies exist
        if not (label_tmp.exists() and image_tmp.exists()):
            raise IOError("Copy verification failed")

        # Step 3: Atomic renames (filesystem operation)
        # If either fails, nothing has moved yet
        label_tmp.rename(label_dst)
        image_tmp.rename(image_dst)

        # Step 4: Delete originals (only after successful moves)
        label_src.unlink()
        image_src.unlink()

    except Exception as e:
        # Cleanup: remove .tmp files if they exist
        if label_tmp.exists():
            label_tmp.unlink()
        if image_tmp.exists():
            image_tmp.unlink()

        # Re-raise with context
        raise IOError(f"Atomic move failed for {label_src.name}: {e}")
```

### Error Handling Matrix

| Failure Point | State After Failure | Recovery |
|--------------|-------------------|----------|
| Copy label fails | Both files in `working/` | Automatic (next iteration) |
| Copy image fails | Both files in `working/` | Automatic (next iteration) |
| Rename label fails | Both files in `working/`, .tmp cleaned | Automatic (next iteration) |
| Rename image fails | Label in `verified/`, image in `working/`, .tmp cleaned | Manual cleanup needed |
| Delete original fails | Duplicates in both dirs | Warning logged, manual cleanup |
| Process crash | .tmp files remain | Cleanup on next run |

**Critical insight:** Rename is atomic filesystem operation. If label rename succeeds but image rename fails, we still have consistent state (label moved back via exception handling).

### Cleanup Strategy

**On startup:**
```python
def cleanup_tmp_files(paths: PathManager):
    """Remove stale .tmp files from previous failed moves."""
    for tmp_file in paths.verified_dir().rglob("*.tmp"):
        age = time.time() - tmp_file.stat().st_mtime
        if age > 3600:  # Older than 1 hour
            tmp_file.unlink()
            logger.info(f"Cleaned stale tmp file: {tmp_file}")
```

**Doctor command checks:**
```python
def validate_structure(self) -> ValidationResult:
    # ... existing checks ...

    # Check for .tmp files
    tmp_files = list(self.paths.root.rglob("*.tmp"))
    if tmp_files:
        return ValidationResult(
            status="warning",
            messages=[f"Found {len(tmp_files)} .tmp files - possible incomplete moves"],
            details={"tmp_files": [str(f) for f in tmp_files]}
        )
```

### Performance Impact

**Disk I/O comparison:**
- Current: 1 move operation per file (2 total)
- New: 1 copy + 1 rename per file (4 total)

**Actual cost:**
- Annotation file: ~10KB, negligible
- Image file: ~500KB average
- For 1000 images: ~500MB copy takes ~2 seconds on SSD
- Trade-off: 2 seconds vs data integrity → worth it

## Migration Strategy

### For Existing Users

**Manual migration required:**
1. Run `yolo-pipeline doctor` - will show structure errors
2. Run migration script: `python scripts/migrate_to_yolo_layout.py`
3. Script moves files from flat structure to images/labels subdirs
4. Re-run `yolo-pipeline doctor` - should pass

**Migration script:**
```python
def migrate_to_yolo_layout(root: Path):
    """Migrate flat directory structure to YOLO layout."""

    for data_dir in ["working", "verified", "eval", "test"]:
        dir_path = root / "data" / data_dir
        if not dir_path.exists():
            continue

        # Create subdirectories
        images_dir = dir_path / "images"
        labels_dir = dir_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        # Move files
        for txt_file in dir_path.glob("*.txt"):
            shutil.move(txt_file, labels_dir / txt_file.name)

        for img_file in dir_path.glob("*.png"):
            shutil.move(img_file, images_dir / img_file.name)

        for img_file in dir_path.glob("*.jpg"):
            shutil.move(img_file, images_dir / img_file.name)

    print("✓ Migration complete")
```

### Documentation Updates

- README: Add "Breaking Change" notice with migration instructions
- QUICKSTART: Update all paths to show images/labels structure
- CLAUDE.md: Document hard enforcement policy

## Testing Strategy

### Unit Tests

**`tests/test_paths.py`** - PathManager
- Verify correct path construction
- Test all methods return expected paths
- Ensure images/labels subdirs included

**`tests/test_validation.py`** - PipelineValidator
- Test structure validation (pass/warning/error cases)
- Test config validation (missing files, invalid YAML)
- Test annotation validation (format errors, missing images)
- Test model validation (corrupt files, wrong architecture)

**`tests/test_atomic_moves.py`** - Atomic move operations
- Test successful move
- Test copy failure (rollback)
- Test rename failure (rollback)
- Test cleanup of .tmp files

### Integration Tests

**`tests/integration/test_doctor.py`** - Doctor command
- Test on valid structure (healthy report)
- Test on broken structure (error report)
- Test on warnings (orphaned files)

**`tests/integration/test_process_manager.py`** - Process manager
- Test service launch
- Test graceful shutdown
- Test crash recovery

**`tests/integration/test_manifests.py`** - Manifest generation
- Test split ratio maintained
- Test stratified splitting
- Test manifest file format
- Test YOLO reads manifests correctly

### End-to-End Tests

**Full pipeline workflow:**
1. Initialize clean structure
2. Add test images to working/
3. Run `yolo-pipeline doctor` → healthy
4. Run `yolo-pipeline run` (mock training)
5. Verify files moved correctly
6. Verify manifests generated
7. Verify model trained
8. Shutdown cleanly

## Implementation Plan

The implementation will follow these phases:

1. **Foundation** (PathManager + Validator)
2. **Component Refactoring** (Update all components)
3. **Manifest System** (Splits + data.yaml)
4. **Smart Resume** (Training from active model)
5. **Doctor Command** (Health checks)
6. **Process Manager** (Single-command workflow)
7. **Atomic Moves** (Copy-then-rename)
8. **Testing** (Unit + integration + e2e)

Detailed tasks and dependencies will be defined in implementation plan.

## Success Criteria

### Robustness
- ✓ Doctor command catches 90%+ of setup errors
- ✓ No inconsistent states from failed file moves
- ✓ All components validate structure on startup

### Usability
- ✓ Single command replaces 4-terminal workflow
- ✓ Clear error messages guide users to fixes
- ✓ Existing commands still work independently

### Performance
- ✓ Training resumes from active model (faster convergence)
- ✓ No file moves during splits (simpler watcher logic)
- ✓ Atomic moves <5s overhead for 1000 images

### Maintainability
- ✓ Single source of truth for paths (PathManager)
- ✓ Centralized validation (no scattered checks)
- ✓ 80%+ test coverage on new code

## Risks and Mitigations

### Risk: Breaking change for existing users
**Mitigation:**
- Clear migration guide in README
- Migration script provided
- Doctor command catches old structure

### Risk: Atomic moves performance impact
**Mitigation:**
- Measured overhead: ~2s for 1000 images
- Acceptable trade-off for data integrity
- Can be optimized later if needed

### Risk: Process manager complexity
**Mitigation:**
- Keep simple (just subprocess management)
- Existing commands still work independently
- Graceful degradation (can Ctrl+C individual processes)

### Risk: Manifest regeneration changes eval metrics
**Mitigation:**
- This is by design (representative eval set)
- Test set provides stable metrics for comparison
- Document clearly in training history logs

## Future Enhancements

Not in this design, but natural next steps:

1. **Priority queue integration** - Auto-populate working/ from queue
2. **Systemd services** - Production deployment option
3. **Remote monitoring** - Web dashboard for pipeline status
4. **Distributed training** - Multi-GPU support
5. **Auto-scaling** - Adjust batch size based on GPU memory

## Appendix: File Structure

```
yolo-iterative-pipeline/
├── data/
│   ├── working/
│   │   ├── images/          # Enforced
│   │   └── labels/          # Enforced
│   ├── verified/
│   │   ├── images/          # Enforced
│   │   └── labels/          # Enforced
│   ├── eval/
│   │   ├── images/          # Enforced (but empty with manifests)
│   │   └── labels/          # Enforced (but empty with manifests)
│   ├── test/
│   │   ├── images/          # Enforced
│   │   └── labels/          # Enforced
│   ├── splits/              # New
│   │   ├── train.txt        # New: manifest file
│   │   └── eval.txt         # New: manifest file
│   └── data.yaml
├── pipeline/
│   ├── paths.py             # New: PathManager
│   ├── validation.py        # New: PipelineValidator
│   ├── process_manager.py   # New: ProcessManager
│   ├── watcher.py           # Updated: use PathManager
│   ├── train.py             # Updated: manifests + smart resume
│   ├── monitor.py           # Updated: use PathManager
│   ├── data_utils.py        # Updated: generate_manifests()
│   └── ...
├── scripts/
│   ├── auto_move_verified.py      # Updated: atomic moves
│   ├── migrate_to_yolo_layout.py  # New: migration helper
│   └── ...
└── tests/
    ├── test_paths.py          # New
    ├── test_validation.py     # New
    ├── test_atomic_moves.py   # New
    └── integration/
        ├── test_doctor.py     # New
        ├── test_process_manager.py  # New
        └── test_manifests.py  # New
```

## References

- Original implementation plan: `docs/plans/2026-02-26-yolo-iterative-pipeline-implementation.md`
- Directory structure fixes: `docs/PIPELINE_FIXES.md`
- YOLO dataset format: https://docs.ultralytics.com/datasets/detect/
