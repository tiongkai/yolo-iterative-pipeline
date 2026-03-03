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
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path to import verification tracker
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.track_verification import VerificationTracker

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
    """
    Find corresponding image file for a label.

    Expects structure:
        data/working/labels/image.txt
        data/working/images/image.png
    """
    base_name = label_path.stem
    # Get parent of labels/ directory (working/)
    working_dir = label_path.parent.parent
    images_dir = working_dir / 'images'

    # Try exact match first
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        img_path = images_dir / f"{base_name}{ext}"
        if img_path.exists():
            return img_path

    # Try with "_detected" suffix (common in detection outputs)
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        img_path = images_dir / f"{base_name}_detected{ext}"
        if img_path.exists():
            return img_path

    return None


def atomic_move_pair(
    label_src: Path,
    image_src: Path,
    label_dst: Path,
    image_dst: Path
) -> bool:
    """Atomically move label and image pair using copy-then-rename.

    Steps:
    1. Copy both files with .tmp extension
    2. Verify both copies exist
    3. Rename atomically (os.rename is atomic on POSIX)
    4. Delete originals only after both renames succeed

    If any step fails, rolls back any partial operations.

    Args:
        label_src: Source label file path
        image_src: Source image file path
        label_dst: Destination label file path
        image_dst: Destination image file path

    Returns:
        True if successful, False if failed
    """
    label_tmp = label_dst.parent / f"{label_dst.name}.tmp"
    image_tmp = image_dst.parent / f"{image_dst.name}.tmp"

    try:
        # Step 1: Copy both files with .tmp extension
        shutil.copy2(str(label_src), str(label_tmp))
        shutil.copy2(str(image_src), str(image_tmp))

        # Step 2: Verify both copies exist
        if not label_tmp.exists() or not image_tmp.exists():
            raise IOError("Copy verification failed")

        # Step 3: Rename atomically (atomic on POSIX systems)
        os.rename(str(label_tmp), str(label_dst))
        os.rename(str(image_tmp), str(image_dst))

        # Step 4: Delete originals only after both renames succeed
        label_src.unlink()
        image_src.unlink()

        return True

    except Exception as e:
        logger.error(f"Atomic move failed: {e}")

        # Rollback: remove any .tmp files created
        if label_tmp.exists():
            label_tmp.unlink()
        if image_tmp.exists():
            image_tmp.unlink()

        # Rollback: remove any partial destination files
        if label_dst.exists():
            label_dst.unlink()
        if image_dst.exists():
            image_dst.unlink()

        return False


def cleanup_tmp_files(verified_dir: Path) -> int:
    """Remove stale .tmp files from verified directory.

    Should be called on startup to clean up after crashes.

    Args:
        verified_dir: Verified directory (data/verified/)

    Returns:
        Number of files removed
    """
    count = 0

    for subdir in ['labels', 'images']:
        dir_path = verified_dir / subdir
        if not dir_path.exists():
            continue

        for tmp_file in dir_path.glob("*.tmp"):
            try:
                tmp_file.unlink()
                count += 1
                logger.info(f"Removed stale temp file: {tmp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {tmp_file.name}: {e}")

    return count


def move_verified_file(label_path: Path, verified_dir: Path, tracker: VerificationTracker = None) -> bool:
    """
    Move label and corresponding image to verified directory.

    Moves:
        data/working/labels/image.txt → data/verified/labels/image.txt
        data/working/images/image.png → data/verified/images/image.png
    """
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

        # Create verified subdirectories
        verified_labels_dir = verified_dir / 'labels'
        verified_images_dir = verified_dir / 'images'
        verified_labels_dir.mkdir(parents=True, exist_ok=True)
        verified_images_dir.mkdir(parents=True, exist_ok=True)

        # Atomically move both files
        success = atomic_move_pair(
            label_path,
            img_path,
            verified_labels_dir / label_path.name,
            verified_images_dir / img_path.name
        )

        if not success:
            logger.error(f"Failed to move {label_path.name}")
            return False

        # Log as verified in tracker
        if tracker:
            tracker.mark_verified(img_path.name)

        logger.info(f"✓ Moved: {label_path.stem} → verified/")
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
    Main loop that monitors working/labels/ directory and moves stable files.

    IMPORTANT: Only moves files that were modified AFTER this script started.
    This ensures pre-labeled files don't auto-move without manual review.

    Expected structure:
        working_dir/labels/*.txt  → verified_dir/labels/*.txt
        working_dir/images/*.png  → verified_dir/images/*.png

    Args:
        working_dir: Directory to monitor (data/working/)
        verified_dir: Destination directory (data/verified/)
        check_interval: Seconds between checks
        stability_threshold: Seconds of no modification before moving
    """
    # Initialize verification tracker
    tracker = VerificationTracker()

    # Clean up stale .tmp files from previous crashes
    tmp_count = cleanup_tmp_files(verified_dir)
    if tmp_count > 0:
        logger.info(f"Cleaned up {tmp_count} stale temp files")

    # Record start time - only files modified after this will be moved
    script_start_time = time.time()

    # Monitor labels subdirectory
    labels_dir = working_dir / 'labels'
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        logger.error(f"Expected structure: {working_dir}/labels/ and {working_dir}/images/")
        return

    logger.info(f"Starting auto-move watcher with verification tracking")
    logger.info(f"  Working dir: {working_dir}")
    logger.info(f"  Labels dir: {labels_dir}")
    logger.info(f"  Verified dir: {verified_dir}")
    logger.info(f"  Check interval: {check_interval}s")
    logger.info(f"  Stability threshold: {stability_threshold}s")
    logger.info(f"  Script start time: {datetime.fromtimestamp(script_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"")
    logger.info(f"⚠️  IMPORTANT: Only files modified AFTER script start will be moved")
    logger.info(f"⚠️  Pre-labeled files will NOT auto-move until you open/save them in X-AnyLabeling")
    logger.info(f"")

    # Initial scan of working directory
    tracker.scan_working_dir(working_dir / 'images')
    stats = tracker.get_stats()
    logger.info(f"  Initial status: {stats['total_verified']} verified, {stats['total_unverified']} unverified")

    try:
        while True:
            # Find all label files in working/labels/ directory
            label_files = list(labels_dir.glob("*.txt"))

            moved_count = 0
            skipped_count = 0

            for label_path in label_files:
                # Get file modification time
                file_mtime = label_path.stat().st_mtime

                # Skip if file was NOT modified after script started
                if file_mtime < script_start_time:
                    skipped_count += 1
                    continue

                # Check if file is stable (not being modified)
                file_age = get_file_age(label_path)

                if file_age >= stability_threshold:
                    if move_verified_file(label_path, verified_dir, tracker):
                        moved_count += 1

            if moved_count > 0:
                verified_labels_dir = verified_dir / 'labels'
                verified_count = len(list(verified_labels_dir.glob("*.txt"))) if verified_labels_dir.exists() else 0
                stats = tracker.get_stats()
                logger.info(f"Moved {moved_count} files. Total verified: {verified_count}")
                logger.info(f"  Progress: {stats['total_verified']} verified / {stats['total']} total ({stats['verification_rate']:.1f}%)")

            # Log status every 10 checks (10 minutes by default)
            if int(time.time() - script_start_time) % (check_interval * 10) < check_interval:
                logger.info(f"Status: {skipped_count} pre-labeled files waiting for review, {len(label_files) - skipped_count} files recently modified")

            # Wait before next check
            time.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info("Auto-move watcher stopped")
        stats = tracker.get_stats()
        logger.info(f"Final status: {stats['total_verified']} verified, {stats['total_unverified']} remaining")


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
