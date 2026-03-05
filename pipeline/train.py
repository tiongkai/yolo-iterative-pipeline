"""YOLO training pipeline with dual evaluation."""

from pathlib import Path
from typing import Optional, Dict, Tuple, TYPE_CHECKING
import time
import shutil
from datetime import datetime
from ultralytics import YOLO
import yaml

from pipeline.config import PipelineConfig, YOLOConfig
from pipeline.data_utils import generate_manifests
from pipeline.metrics import (
    calculate_f1_score,
    format_metrics,
    append_training_history,
    load_training_history
)

if TYPE_CHECKING:
    from pipeline.paths import PathManager


def init_model(
    paths: 'PathManager',
    yolo_config: YOLOConfig,
    from_scratch: bool = False
) -> Tuple[YOLO, str]:
    """Initialize YOLO model with smart resume.

    Checks for active model and resumes from it if available.
    Falls back to pretrained model if:
    - No active model exists
    - Active model fails to load (corrupted)
    - from_scratch flag is set

    Args:
        paths: PathManager instance
        yolo_config: YOLO configuration
        from_scratch: If True, ignore active model and use pretrained

    Returns:
        (model, source) where source is "active" or "pretrained"
    """
    active_model_path = paths.active_model()

    # Check if we should use active model
    if not from_scratch and active_model_path.exists():
        try:
            print(f"Loading active model from {active_model_path.name}...")
            model = YOLO(str(active_model_path))

            # Verify model is valid
            if not hasattr(model, 'model') or model.model is None:
                raise ValueError("Loaded model is invalid or uninitialized")

            print(f"✓ Resuming from active model (version {active_model_path.parent.parent.name})")
            return model, "active"
        except Exception as e:
            print(f"⚠ Failed to load active model ({type(e).__name__}): {e}")
            print(f"  Falling back to pretrained model...")
    elif from_scratch:
        print(f"🔄 Training from scratch (--from-scratch flag set)")
    else:
        print(f"ℹ No active model found, using pretrained model")

    # Use pretrained model
    print(f"Loading pretrained model {yolo_config.model}...")
    model = YOLO(yolo_config.model)
    return model, "pretrained"


def _make_relative_safe(path: Path, root: Path) -> str:
    """Make path relative to root, fallback to absolute if not possible.

    Args:
        path: Path to make relative
        root: Root directory to make relative to

    Returns:
        Relative path string if possible, absolute path otherwise
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        # Path is not relative to root (symlink, mount, etc)
        return str(path.absolute())


def load_classes_from_file(classes_file: Path) -> list:
    """Load class names from classes.txt file.

    Args:
        classes_file: Path to classes.txt file

    Returns:
        List of class names (non-empty lines)

    Raises:
        FileNotFoundError: If classes.txt doesn't exist
    """
    if not classes_file.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_file}")

    classes = []
    with open(classes_file, 'r') as f:
        for line in f:
            class_name = line.strip()
            if class_name:  # Skip empty lines
                classes.append(class_name)

    if not classes:
        raise ValueError(f"No classes found in {classes_file}")

    return classes


def create_data_yaml(
    train_manifest: Path,
    eval_manifest: Path,
    test_dir: Path,
    classes: list,
    output_path: Path,
    root_dir: Path
):
    """Create data.yaml for YOLO training.

    Args:
        train_manifest: Path to train.txt manifest file
        eval_manifest: Path to eval.txt manifest file
        test_dir: Test data directory (still uses directory structure)
        classes: List of class names
        output_path: Where to save data.yaml
        root_dir: Root directory for the project (to make relative paths)
    """
    data = {
        "path": str(root_dir),
        "train": _make_relative_safe(train_manifest, root_dir),
        "val": _make_relative_safe(eval_manifest, root_dir),
        "test": _make_relative_safe(test_dir, root_dir),
        "names": {i: name for i, name in enumerate(classes)}
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def get_next_version(log_path: Path) -> str:
    """Get next model version number.

    Args:
        log_path: Path to training_history.json

    Returns:
        Next version string (e.g., "v003")
    """
    history = load_training_history(log_path)
    if not history:
        return "v001"

    last_version = history[-1]["version"]
    version_num = int(last_version.replace("v", ""))
    return f"v{version_num + 1:03d}"


def evaluate_model(
    model: YOLO,
    data_yaml: Path,
    split: str = "val"
) -> Dict[str, float]:
    """Evaluate model on a dataset split.

    Args:
        model: Trained YOLO model
        data_yaml: Path to data.yaml
        split: Split to evaluate on ("val" or "test")

    Returns:
        Dictionary of metrics
    """
    results = model.val(data=str(data_yaml), split=split, verbose=False)

    # Extract metrics
    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    # Add F1
    metrics = format_metrics(metrics, add_f1=True)

    return metrics


def train_model(
    pipeline_config: PipelineConfig,
    yolo_config: YOLOConfig,
    paths: 'PathManager',
    bootstrap: bool = False,
    from_scratch: bool = False
) -> Tuple[str, Path]:
    """Train YOLO model.

    Args:
        pipeline_config: Pipeline configuration
        yolo_config: YOLO training configuration
        paths: PathManager instance for path resolution
        bootstrap: If True, train on SAM3 annotations (noisy baseline)
        from_scratch: If True, ignore active model and use pretrained

    Returns:
        (model_version, checkpoint_dir)
    """
    start_time = time.time()

    # Setup paths using PathManager
    data_yaml = paths.data_yaml()
    log_path = paths.training_history()

    # Determine source for training
    if bootstrap:
        print("Bootstrap mode: training on SAM3 annotations")
        print("NOTE: Bootstrap not recommended - manually verify data instead")
        print("Skipping bootstrap - requires manual setup of verified/ directory")

    # Check minimum images (expects verified/labels/*.txt structure)
    labels_dir = paths.verified_labels()
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")

    train_files = list(labels_dir.glob("*.txt"))
    if len(train_files) < pipeline_config.min_train_images:
        raise ValueError(
            f"Need at least {pipeline_config.min_train_images} images, "
            f"found {len(train_files)}"
        )

    # Generate manifests (replaces eval set sampling)
    print(f"Generating train/eval manifests ({pipeline_config.eval_split_ratio * 100:.0f}% eval)...")
    train_count, eval_count = generate_manifests(
        paths=paths,
        config=pipeline_config,
        random_seed=42
    )
    print(f"  Train: {train_count} images")
    print(f"  Eval: {eval_count} images")

    # Load classes from verified/classes.txt (source of truth)
    classes_file = paths.verified_dir() / "classes.txt"
    classes = load_classes_from_file(classes_file)
    print(f"  Classes: {len(classes)} classes from {classes_file.name}")

    # Create data.yaml
    create_data_yaml(
        train_manifest=paths.train_manifest(),
        eval_manifest=paths.eval_manifest(),
        test_dir=paths.test_dir(),
        classes=classes,
        output_path=data_yaml,
        root_dir=paths.root_dir
    )

    # Initialize model (with smart resume)
    model, source = init_model(paths, yolo_config, from_scratch=from_scratch)

    # Get version before training
    version = get_next_version(log_path)
    run_name = f"model_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Train
    print(f"Training on {train_count} images...")
    checkpoint_dir = paths.checkpoint_dir() / run_name  # Known path before training

    results = model.train(
        data=str(data_yaml),
        epochs=yolo_config.epochs,
        batch=yolo_config.batch_size,
        imgsz=yolo_config.imgsz,
        device=yolo_config.device,
        patience=yolo_config.patience,
        close_mosaic=yolo_config.close_mosaic,
        copy_paste=yolo_config.copy_paste,
        mixup=yolo_config.mixup,
        scale=yolo_config.scale,
        fliplr=yolo_config.fliplr,
        hsv_h=yolo_config.hsv_h,
        hsv_s=yolo_config.hsv_s,
        hsv_v=yolo_config.hsv_v,
        degrees=yolo_config.degrees,
        translate=yolo_config.translate,
        mosaic=yolo_config.mosaic,
        project=str(paths.checkpoint_dir()),
        name=run_name,
        exist_ok=True,
    )

    # Verify training completed successfully
    if not checkpoint_dir.exists():
        raise RuntimeError(f"Training failed: checkpoint directory not created at {checkpoint_dir}")

    weights_dir = checkpoint_dir / "weights"
    if not (weights_dir / "best.pt").exists():
        raise RuntimeError(f"Training failed: best.pt not found at {weights_dir}")

    # Evaluate
    print("Evaluating on eval set...")
    eval_metrics = evaluate_model(model, data_yaml, split="val")

    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, data_yaml, split="test")

    # Save training info
    training_time = (time.time() - start_time) / 60

    append_training_history(
        log_path=log_path,
        version=version,
        train_images=train_count,
        eval_metrics=eval_metrics,
        test_metrics=test_metrics,
        training_time_minutes=training_time,
        notes="Bootstrap training on SAM3" if bootstrap else ""
    )

    print(f"\n✓ Training complete ({training_time:.1f} min)")
    print(f"  Eval mAP50: {eval_metrics['mAP50']:.3f}, F1: {eval_metrics['f1']:.3f}")
    print(f"  Test mAP50: {test_metrics['mAP50']:.3f}, F1: {test_metrics['f1']:.3f}")

    return version, checkpoint_dir


def export_to_onnx(model_path: Path, output_path: Path, imgsz: int = 1280) -> bool:
    """Export YOLO model to ONNX format.

    Args:
        model_path: Path to .pt model file
        output_path: Path to save .onnx file
        imgsz: Input image size

    Returns:
        True if export succeeded, False otherwise
    """
    try:
        print("  Exporting to ONNX format...")
        model = YOLO(str(model_path))
        
        # Export to ONNX (saves to same directory as model)
        onnx_path = model.export(format='onnx', imgsz=imgsz, opset=21, verbose=False)
        
        # Copy to desired output location
        if Path(onnx_path).exists():
            shutil.copy2(onnx_path, output_path)
            print(f"  ✓ ONNX model saved to {output_path.name}")
            return True
        else:
            print(f"  ⚠️  ONNX export failed")
            return False
    except Exception as e:
        print(f"  ⚠️  ONNX export error: {e}")
        return False


def promote_model(
    checkpoint_dir: Path,
    active_dir: Path,
    paths: 'PathManager'
,
    export_onnx: bool = True
) -> bool:
    """Promote model to active if it improved.

    Args:
        checkpoint_dir: Directory with new checkpoint
        active_dir: Active model directory
        paths: PathManager instance

        export_onnx: Whether to export to ONNX format (default: True)
    Returns:
        True if promoted, False otherwise
    """
    log_path = paths.training_history()
    history = load_training_history(log_path)

    if len(history) < 2:
        # First model, always promote
        should_promote = True
    else:
        # Check if eval mAP improved
        current = history[-1]
        previous = history[-2]
        should_promote = current["eval_mAP50"] > previous["eval_mAP50"]

    if should_promote:
        active_dir.mkdir(parents=True, exist_ok=True)
        best_pt = checkpoint_dir / "weights" / "best.pt"

        if best_pt.exists():
            # Create symlink
            active_link = active_dir / "best.pt"
            if active_link.exists() or active_link.is_symlink():
                active_link.unlink()
            active_link.symlink_to(best_pt.absolute())

            print(f"✓ Model promoted to active")
            
            # Export to ONNX for X-AnyLabeling
            if export_onnx:
                onnx_output = active_dir / "best.onnx"
                export_to_onnx(best_pt, onnx_output)
            return True
        else:
            print(f"⚠️  Warning: best.pt not found in {checkpoint_dir}")
            return False
    else:
        print(f"✗ Model not promoted (no improvement)")
        return False


def main():
    """CLI entry point for training."""
    import argparse
    from pipeline.paths import PathManager

    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--bootstrap", action="store_true",
                       help="Train on SAM3 annotations (bootstrap)")
    parser.add_argument("--from-scratch", action="store_true",
                       help="Train from pretrained model (ignore active model)")
    parser.add_argument("--pipeline-config", type=Path,
                       default="configs/pipeline_config.yaml")
    parser.add_argument("--yolo-config", type=Path,
                       default="configs/yolo_config.yaml")

    args = parser.parse_args()

    # Load configs
    pipeline_config = PipelineConfig.from_yaml(args.pipeline_config)
    yolo_config = YOLOConfig.from_yaml(args.yolo_config)

    # Create PathManager
    paths = PathManager(Path.cwd(), pipeline_config)

    # Train
    version, checkpoint_dir = train_model(
        pipeline_config,
        yolo_config,
        paths,
        bootstrap=args.bootstrap,
        from_scratch=args.from_scratch
    )

    # Promote if improved
    promote_model(checkpoint_dir, paths.active_model().parent, paths)

    # Re-score priority queue
    print("\nRe-scoring priority queue...")
    from pipeline.active_learning import score_all_images, save_priority_queue

    model_path = paths.active_model()
    if model_path.exists():
        scores = score_all_images(
            working_dir=paths.working_dir(),
            sam3_dir=paths.sam3_dir(),
            model_path=model_path
        )
        save_priority_queue(scores, paths.priority_queue(), version)
        print("✓ Priority queue updated")
    else:
        print("⚠️  Skipping priority queue update (no active model)")


if __name__ == "__main__":
    main()
