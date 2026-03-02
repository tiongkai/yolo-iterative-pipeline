#!/usr/bin/env python3
"""
Automatic file movement from working to verified directory.

Monitors data/working/ and automatically moves annotation files to data/verified/
when they meet verification criteria:
- File has not been modified for N seconds (default: 60)
- File has valid YOLO annotations
- Corresponding image exists

Usage:
    python scripts/auto_move_verified.py [--interval 60] [--stability 60]
"""

import time
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_move.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_valid_yolo_annotation(label_path: Path) -> bool:
    """Check if file contains valid YOLO annotations."""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            return False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                return False

            # Check format: class_id cx cy w h
            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                # Check if normalized (0-1)
                if not all(0 <= c <= 1 for c in coords):
                    return False
            except ValueError:
                return False

        return True
    except Exception as e:
        logger.warning(f"Error validating {label_path}: {e}")
        return False


def find_image_for_label(label_path: Path) -> Path:
    """Find corresponding image file for a label."""
    base_name = label_path.stem
    working_dir = label_path.parent

    # Try exact match first
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        img_path = working_dir / f"{base_name}{ext}"
        if img_path.exists():
            return img_path

    # Try with "_detected" suffix (common in detection outputs)
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        img_path = working_dir / f"{base_name}_detected{ext}"
        if img_path.exists():
            return img_path

    return None


def move_verified_file(label_path: Path, verified_dir: Path) -> bool:
    """Move label and corresponding image to verified directory."""
    try:
        # Find image
        img_path = find_image_for_label(label_path)
        if not img_path:
            logger.warning(f"No image found for {label_path.name}")
            return False

        # Validate annotation
        if not is_valid_yolo_annotation(label_path):
            logger.warning(f"Invalid YOLO format: {label_path.name}")
            return False

        # Move both files
        verified_dir.mkdir(parents=True, exist_ok=True)

        shutil.move(str(label_path), str(verified_dir / label_path.name))
        shutil.move(str(img_path), str(verified_dir / img_path.name))

        logger.info(f"✓ Moved: {label_path.stem}")
        return True

    except Exception as e:
        logger.error(f"Error moving {label_path.name}: {e}")
        return False


def get_file_age(file_path: Path) -> float:
    """Get time since file was last modified (in seconds)."""
    mtime = file_path.stat().st_mtime
    return time.time() - mtime


def auto_move_loop(
    working_dir: Path,
    verified_dir: Path,
    check_interval: int = 60,
    stability_threshold: int = 60
):
    """
    Main loop that monitors working directory and moves stable files.

    Args:
        working_dir: Directory to monitor (data/working/)
        verified_dir: Destination directory (data/verified/)
        check_interval: Seconds between checks
        stability_threshold: Seconds of no modification before moving
    """
    logger.info(f"Starting auto-move watcher")
    logger.info(f"  Working dir: {working_dir}")
    logger.info(f"  Verified dir: {verified_dir}")
    logger.info(f"  Check interval: {check_interval}s")
    logger.info(f"  Stability threshold: {stability_threshold}s")

    try:
        while True:
            # Find all label files in working directory
            label_files = list(working_dir.glob("*.txt"))

            moved_count = 0
            for label_path in label_files:
                # Check if file is stable (not being modified)
                file_age = get_file_age(label_path)

                if file_age >= stability_threshold:
                    if move_verified_file(label_path, verified_dir):
                        moved_count += 1

            if moved_count > 0:
                verified_count = len(list(verified_dir.glob("*.txt")))
                logger.info(f"Moved {moved_count} files. Total verified: {verified_count}")

            # Wait before next check
            time.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info("Auto-move watcher stopped")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-move verified annotations from working to verified directory"
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Check interval in seconds (default: 60)'
    )
    parser.add_argument(
        '--stability',
        type=int,
        default=60,
        help='File stability threshold in seconds (default: 60)'
    )
    parser.add_argument(
        '--working-dir',
        type=Path,
        default=Path('data/working'),
        help='Working directory to monitor (default: data/working)'
    )
    parser.add_argument(
        '--verified-dir',
        type=Path,
        default=Path('data/verified'),
        help='Verified directory destination (default: data/verified)'
    )

    args = parser.parse_args()

    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)

    auto_move_loop(
        working_dir=args.working_dir,
        verified_dir=args.verified_dir,
        check_interval=args.interval,
        stability_threshold=args.stability
    )


if __name__ == '__main__':
    main()
