"""Model export utilities for production deployment."""

from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime


def export_model(
    checkpoint_dir: Path,
    formats: list,
    output_dir: Path
):
    """Export model to deployment formats.

    Args:
        checkpoint_dir: Directory with model checkpoint
        formats: List of formats to export to (onnx, tensorrt, torchscript)
        output_dir: Output directory for exports
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = checkpoint_dir / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    exports = {}

    for fmt in formats:
        print(f"Exporting to {fmt}...")

        if fmt == "onnx":
            export_path = model.export(format="onnx")
            exports[fmt] = str(export_path)

        elif fmt == "tensorrt" or fmt == "engine":
            export_path = model.export(format="engine", device=0)
            exports[fmt] = str(export_path)

        elif fmt == "torchscript":
            export_path = model.export(format="torchscript")
            exports[fmt] = str(export_path)

        else:
            print(f"Unknown format: {fmt}")

    # Save deployment info
    deployment_info = {
        "timestamp": datetime.now().isoformat(),
        "source_model": str(model_path),
        "exports": exports,
    }

    info_path = output_dir / "deployment_info.json"
    with open(info_path, "w") as f:
        json.dump(deployment_info, f, indent=2)

    print(f"✓ Exports saved to {output_dir}")
    return exports


def main():
    """CLI entry point for export."""
    import argparse

    parser = argparse.ArgumentParser(description="Export model for deployment")
    parser.add_argument("--version", required=True,
                       help="Model version to export (e.g., v003)")
    parser.add_argument("--formats", nargs="+",
                       default=["onnx"],
                       choices=["onnx", "tensorrt", "engine", "torchscript"],
                       help="Export formats")
    parser.add_argument("--output", type=Path,
                       help="Output directory (default: models/deployed/<version>)")

    args = parser.parse_args()

    # Find checkpoint directory
    checkpoint_dirs = list(Path("models/checkpoints").glob(f"model_{args.version}*"))
    if not checkpoint_dirs:
        print(f"✗ Model {args.version} not found")
        exit(1)

    checkpoint_dir = checkpoint_dirs[0]

    output_dir = args.output or Path(f"models/deployed/model_{args.version}")

    export_model(checkpoint_dir, args.formats, output_dir)


if __name__ == "__main__":
    main()
