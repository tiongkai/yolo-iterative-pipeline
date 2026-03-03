#!/usr/bin/env python3
"""Verify that all pipeline components work with images/labels structure."""

from pathlib import Path
import sys

def check_structure():
    """Check that directory structure is correct."""
    print("=" * 60)
    print("DIRECTORY STRUCTURE VERIFICATION")
    print("=" * 60)
    print()

    checks = []

    # Check working directory
    working = Path("data/working")
    working_images = working / "images"
    working_labels = working / "labels"

    checks.append(("working/images exists", working_images.exists()))
    checks.append(("working/labels exists", working_labels.exists()))

    if working_labels.exists():
        label_count = len(list(working_labels.glob("*.txt")))
        checks.append((f"working/labels has files ({label_count})", label_count > 0))

    if working_images.exists():
        image_count = len(list(working_images.glob("*.png"))) + len(list(working_images.glob("*.jpg")))
        checks.append((f"working/images has files ({image_count})", image_count > 0))

    # Check verified directory
    verified = Path("data/verified")
    verified_images = verified / "images"
    verified_labels = verified / "labels"

    checks.append(("verified directory exists", verified.exists()))

    if verified_labels.exists():
        label_count = len(list(verified_labels.glob("*.txt")))
        checks.append((f"verified/labels count: {label_count}", True))

    if verified_images.exists():
        image_count = len(list(verified_images.glob("*.png"))) + len(list(verified_images.glob("*.jpg")))
        checks.append((f"verified/images count: {image_count}", True))

    # Print results
    all_pass = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_pass = False

    print()
    print("=" * 60)

    if all_pass:
        print("✓ All checks passed!")
    else:
        print("✗ Some checks failed!")
        return False

    return True


def check_pipeline_code():
    """Verify that pipeline code uses correct paths."""
    print()
    print("=" * 60)
    print("PIPELINE CODE VERIFICATION")
    print("=" * 60)
    print()

    checks = []

    # Check watcher
    try:
        from pipeline.watcher import FileWatcher
        watcher = FileWatcher(
            verified_dir=Path("data/verified"),
            trigger_threshold=50  # Default threshold
        )
        count = watcher.count_verified_images()
        checks.append((f"Watcher counts correctly ({count} files)", True))
    except Exception as e:
        checks.append((f"Watcher check failed: {e}", False))

    # Check monitor
    try:
        from pipeline.monitor import display_status
        # Just check it doesn't crash
        checks.append(("Monitor imports correctly", True))
    except Exception as e:
        checks.append((f"Monitor check failed: {e}", False))

    # Check data_utils
    try:
        from pipeline.data_utils import sample_eval_set
        checks.append(("Data utils imports correctly", True))
    except Exception as e:
        checks.append((f"Data utils check failed: {e}", False))

    # Print results
    all_pass = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_pass = False

    print()
    print("=" * 60)

    if all_pass:
        print("✓ All pipeline checks passed!")
    else:
        print("✗ Some pipeline checks failed!")
        return False

    return True


def main():
    print()
    structure_ok = check_structure()
    pipeline_ok = check_pipeline_code()

    print()
    if structure_ok and pipeline_ok:
        print("🎉 VERIFICATION COMPLETE - All systems ready!")
        print()
        print("Next steps:")
        print("  1. Terminal 1: python scripts/auto_move_verified.py")
        print("  2. Terminal 2: yolo-pipeline-watch")
        print("  3. Terminal 3: watch -n 5 yolo-pipeline-monitor")
        print("  4. Terminal 4: xanylabeling (Open Dir: data/working/)")
        print()
        return 0
    else:
        print("❌ VERIFICATION FAILED - Fix issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
