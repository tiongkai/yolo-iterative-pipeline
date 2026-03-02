"""YOLO training pipeline with dual evaluation."""

from pathlib import Path
from typing import Optional, Dict, Tuple
import time
import shutil
from datetime import datetime
from ultralytics import YOLO
import yaml

from pipeline.config import PipelineConfig, YOLOConfig
from pipeline.data_utils import get_image_label_pairs, sample_eval_set
from pipeline.metrics import (
    calculate_f1_score,
    format_metrics,
    append_training_history,
    load_training_history
)


def create_data_yaml(
    train_dir: Path,
    eval_dir: Path,
    test_dir: Path,
    classes: list,
    output_path: Path
):
    """Create data.yaml for YOLO training.

    Args:
        train_dir: Training data directory
        eval_dir: Evaluation data directory
        test_dir: Test data directory
        classes: List of class names
        output_path: Where to save data.yaml
    """
    data = {
        "path": str(Path.cwd()),
        "train": str(train_dir.relative_to(Path.cwd())),
        "val": str(eval_dir.relative_to(Path.cwd())),
        "test": str(test_dir.relative_to(Path.cwd())),
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
    bootstrap: bool = False
) -> Tuple[str, Path]:
    """Train YOLO model.

    Args:
        pipeline_config: Pipeline configuration
        yolo_config: YOLO training configuration
        bootstrap: If True, train on SAM3 annotations (noisy baseline)

    Returns:
        (model_version, checkpoint_dir)
    """
    start_time = time.time()

    # Setup paths
    verified_dir = Path("data/verified")
    eval_dir = Path("data/eval")
    test_dir = Path("data/test")
    data_yaml = Path("data/data.yaml")
    log_path = Path("logs/training_history.json")

    # Determine source for training
    if bootstrap:
        print("Bootstrap mode: training on SAM3 annotations")
        train_source = Path("data/sam3_annotations")
        # Copy SAM3 to verified for bootstrap
        verified_dir.mkdir(parents=True, exist_ok=True)
        for file in train_source.glob("*"):
            if file.suffix.lower() in {".txt", ".jpg", ".jpeg", ".png", ".bmp"}:
                dest = verified_dir / file.name
                if not dest.exists():
                    shutil.copy(file, dest)
    else:
        train_source = verified_dir

    # Check minimum images
    train_files = list(verified_dir.glob("*.txt"))
    if len(train_files) < pipeline_config.min_train_images:
        raise ValueError(
            f"Need at least {pipeline_config.min_train_images} images, "
            f"found {len(train_files)}"
        )

    # Sample eval set if enough images
    if len(train_files) >= 100 and not bootstrap:
        print(f"Sampling eval set ({pipeline_config.eval_split_ratio * 100:.0f}%)...")
        sample_eval_set(
            verified_dir=verified_dir,
            eval_dir=eval_dir,
            split_ratio=pipeline_config.eval_split_ratio,
            stratify=pipeline_config.stratify,
            num_classes=len(pipeline_config.classes)
        )
        val_dir = eval_dir
    else:
        val_dir = verified_dir
        print(f"Using all {len(train_files)} images for training (eval set not sampled)")

    # Create data.yaml
    create_data_yaml(
        train_dir=verified_dir,
        eval_dir=val_dir,
        test_dir=test_dir,
        classes=pipeline_config.classes,
        output_path=data_yaml
    )

    # Initialize model
    print(f"Initializing {yolo_config.model}...")
    model = YOLO(yolo_config.model)

    # Get version before training
    version = get_next_version(log_path)
    run_name = f"model_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Train
    print(f"Training on {len(list(verified_dir.glob('*.txt')))} images...")
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
        project="models/checkpoints",
        name=run_name,
        exist_ok=True,
    )

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
        train_images=len(list(verified_dir.glob("*.txt"))),
        eval_metrics=eval_metrics,
        test_metrics=test_metrics,
        training_time_minutes=training_time,
        notes="Bootstrap training on SAM3" if bootstrap else ""
    )

    # Get checkpoint directory
    checkpoint_dir = Path(results.save_dir)

    print(f"\n✓ Training complete ({training_time:.1f} min)")
    print(f"  Eval mAP50: {eval_metrics['mAP50']:.3f}, F1: {eval_metrics['f1']:.3f}")
    print(f"  Test mAP50: {test_metrics['mAP50']:.3f}, F1: {test_metrics['f1']:.3f}")

    return version, checkpoint_dir


def promote_model(checkpoint_dir: Path, active_dir: Path) -> bool:
    """Promote model to active if it improved.

    Args:
        checkpoint_dir: Directory with new checkpoint
        active_dir: Active model directory

    Returns:
        True if promoted, False otherwise
    """
    log_path = Path("logs/training_history.json")
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

    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--bootstrap", action="store_true",
                       help="Train on SAM3 annotations (bootstrap)")
    parser.add_argument("--pipeline-config", type=Path,
                       default="configs/pipeline_config.yaml")
    parser.add_argument("--yolo-config", type=Path,
                       default="configs/yolo_config.yaml")

    args = parser.parse_args()

    # Load configs
    pipeline_config = PipelineConfig.from_yaml(args.pipeline_config)
    yolo_config = YOLOConfig.from_yaml(args.yolo_config)

    # Train
    try:
        version, checkpoint_dir = train_model(pipeline_config, yolo_config, args.bootstrap)

        # Promote if improved
        promote_model(checkpoint_dir, Path("models/active"))

        # Re-score priority queue
        print("\nRe-scoring priority queue...")
        from pipeline.active_learning import score_all_images, save_priority_queue

        model_path = Path("models/active/best.pt")
        if model_path.exists():
            scores = score_all_images(
                working_dir=Path("data/working"),
                sam3_dir=Path("data/sam3_annotations"),
                model_path=model_path
            )
            save_priority_queue(scores, Path("logs/priority_queue.txt"), version)
            print("✓ Priority queue updated")
        else:
            print("⚠️  Skipping priority queue update (no active model)")

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
