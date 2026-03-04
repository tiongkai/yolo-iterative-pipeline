#!/usr/bin/env python3
"""
Automatic file movement from working to verified directory.

Monitors data/working/images/ for X-AnyLabeling JSON files with verified flag.
When a JSON file has flags.verified=true:
- Converts JSON annotations to YOLO format
- Moves image to data/verified/images/
- Creates YOLO label in data/verified/labels/
- Leaves JSON in working/images/ (for X-AnyLabeling)

Usage:
    python scripts/auto_move_verified.py [--interval 60] [--stability 60]
"""

import time
import shutil
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

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


def load_class_mapping(classes_file: Path) -> Dict[str, int]:
    """Load classes.txt and return mapping of class_name -> class_id.

    Args:
        classes_file: Path to classes.txt file

    Returns:
        Dictionary mapping class names to IDs (0-indexed)

    Example:
        {'boat': 0, 'human': 1, 'outboard motor': 2}
    """
    if not classes_file.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_file}")

    class_map = {}
    class_id = 0
    with open(classes_file, 'r') as f:
        for line in f:
            class_name = line.strip()
            if class_name:  # Skip empty lines
                class_map[class_name] = class_id
                class_id += 1

    return class_map


def validate_bbox_coordinates(
    points: List[List[float]],
    img_width: int,
    img_height: int
) -> bool:
    """Validate that bbox coordinates are within image bounds.

    Args:
        points: List of [x, y] coordinates (4 corners of rectangle)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        True if all coordinates are valid, False otherwise
    """
    if len(points) != 4:
        return False

    for point in points:
        if len(point) != 2:
            return False
        x, y = point
        if x < 0 or x > img_width or y < 0 or y > img_height:
            return False

    return True


def parse_xanylabeling_json(
    json_path: Path,
    class_map: Dict[str, int]
) -> Tuple[List[str], List[str]]:
    """Parse X-AnyLabeling JSON and convert to YOLO format.

    Args:
        json_path: Path to X-AnyLabeling JSON file
        class_map: Dictionary mapping class names to IDs

    Returns:
        Tuple of (yolo_lines, warnings):
            - yolo_lines: List of YOLO format strings
            - warnings: List of warning messages for invalid shapes

    YOLO format: class_id center_x center_y width height (normalized 0-1)
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return [], [f"Failed to parse JSON: {e}"]

    img_width = data.get('imageWidth')
    img_height = data.get('imageHeight')
    shapes = data.get('shapes', [])

    if not img_width or not img_height:
        return [], ["Missing imageWidth or imageHeight in JSON"]

    yolo_lines = []
    warnings = []

    for shape in shapes:
        label = shape.get('label')
        points = shape.get('points', [])
        shape_type = shape.get('shape_type', 'rectangle')

        # Only process rectangles
        if shape_type != 'rectangle':
            warnings.append(f"Skipping non-rectangle shape: {shape_type}")
            continue

        # Validate label exists in class mapping
        if label not in class_map:
            warnings.append(f"Unknown class label: {label}")
            continue

        # Validate coordinates
        if not validate_bbox_coordinates(points, img_width, img_height):
            warnings.append(f"Invalid bbox coordinates for label: {label}")
            continue

        # Convert pixel coordinates to YOLO format
        # X-AnyLabeling provides 4 corners: [top-left, top-right, bottom-right, bottom-left]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Calculate center and dimensions (normalized)
        center_x = ((x_min + x_max) / 2) / img_width
        center_y = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # Validate normalized coordinates are in [0, 1]
        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and
                0 <= width <= 1 and 0 <= height <= 1):
            warnings.append(f"Normalized coordinates out of range for label: {label}")
            continue

        # Format as YOLO: class_id center_x center_y width height
        class_id = class_map[label]
        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)

    return yolo_lines, warnings


def is_verified(json_path: Path) -> bool:
    """Check if JSON file has verified flag set to true.

    Args:
        json_path: Path to X-AnyLabeling JSON file

    Returns:
        True if flags.verified == true, False otherwise
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        flags = data.get('flags', {})
        return flags.get('verified', False) is True
    except Exception as e:
        logger.warning(f"Error checking verified flag in {json_path.name}: {e}")
        return False


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
    1. Validate source files exist
    2. Check destination files don't already exist
    3. Copy both files with .tmp extension
    4. Verify both copies exist
    5. Rename atomically (os.rename is atomic on POSIX)
    6. Delete originals only after both renames succeed

    If any step fails, rolls back any partial operations.

    Args:
        label_src: Source label file path
        image_src: Source image file path
        label_dst: Destination label file path
        image_dst: Destination image file path

    Returns:
        True if successful, False if failed
    """
    # Validate source files exist
    if not label_src.exists():
        logger.error(f"Source label does not exist: {label_src}")
        return False
    if not image_src.exists():
        logger.error(f"Source image does not exist: {image_src}")
        return False

    # Check if destination already exists
    if label_dst.exists() or image_dst.exists():
        logger.warning(f"Destination already exists: {label_dst.name}")
        return False

    label_tmp = label_dst.parent / f"{label_dst.name}.tmp"
    image_tmp = image_dst.parent / f"{image_dst.name}.tmp"

    # Track which renames succeeded for proper rollback
    label_renamed = False
    image_renamed = False

    try:
        # Step 1: Copy both files with .tmp extension
        shutil.copy2(str(label_src), str(label_tmp))
        shutil.copy2(str(image_src), str(image_tmp))

        # Step 2: Verify both copies exist
        if not label_tmp.exists() or not image_tmp.exists():
            raise IOError("Copy verification failed")

        # Step 3: Rename atomically (atomic on POSIX systems)
        os.rename(str(label_tmp), str(label_dst))
        label_renamed = True  # Track success

        os.rename(str(image_tmp), str(image_dst))
        image_renamed = True  # Track success

        # Step 4: Delete originals only after both renames succeed
        label_src.unlink()
        image_src.unlink()

        return True

    except Exception as e:
        logger.error(f"Atomic move failed: {e}")

        # Rollback based on what succeeded
        if label_renamed:
            # Label was moved, move it back
            try:
                if label_dst.exists():
                    os.rename(str(label_dst), str(label_tmp))
            except Exception as rollback_err:
                logger.warning(f"Failed to rollback label: {rollback_err}")

        if image_renamed:
            # Image was moved, move it back
            try:
                if image_dst.exists():
                    os.rename(str(image_dst), str(image_tmp))
            except Exception as rollback_err:
                logger.warning(f"Failed to rollback image: {rollback_err}")

        # Clean up .tmp files
        try:
            if label_tmp.exists():
                label_tmp.unlink()
        except Exception as cleanup_err:
            logger.warning(f"Failed to cleanup {label_tmp.name}: {cleanup_err}")

        try:
            if image_tmp.exists():
                image_tmp.unlink()
        except Exception as cleanup_err:
            logger.warning(f"Failed to cleanup {image_tmp.name}: {cleanup_err}")

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
    Main loop that monitors working/images/ for verified JSON files.

    Checks X-AnyLabeling JSON files for flags.verified=true.
    When found:
    - Converts JSON annotations to YOLO format
    - Moves image to verified/images/
    - Creates YOLO label in verified/labels/
    - Leaves JSON in working/images/

    Expected structure:
        working_dir/images/*.json  → check for verified flag
        working_dir/images/*.png   → verified_dir/images/*.png
        (generated) → verified_dir/labels/*.txt

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

    # Load class mapping from verified/classes.txt
    classes_file = verified_dir / 'classes.txt'
    try:
        class_map = load_class_mapping(classes_file)
        logger.info(f"Loaded {len(class_map)} classes from {classes_file}")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Cannot proceed without classes.txt")
        return

    # Monitor images subdirectory (contains JSONs)
    images_dir = working_dir / 'images'
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        logger.error(f"Expected structure: {working_dir}/images/")
        return

    logger.info(f"Starting auto-move watcher (JSON-based verification)")
    logger.info(f"  Working dir: {working_dir}")
    logger.info(f"  Images dir: {images_dir}")
    logger.info(f"  Verified dir: {verified_dir}")
    logger.info(f"  Check interval: {check_interval}s")
    logger.info(f"  Stability threshold: {stability_threshold}s")
    logger.info(f"")
    logger.info(f"✓  Monitoring: {images_dir}/*.json")
    logger.info(f"✓  Checking: flags.verified == true")
    logger.info(f"✓  Converting: JSON → YOLO format")
    logger.info(f"")

    # Initial scan of working directory
    tracker.scan_working_dir(images_dir)
    stats = tracker.get_stats()
    logger.info(f"  Initial status: {stats['total_verified']} verified, {stats['total_unverified']} unverified")

    try:
        while True:
            # Find all JSON files in working/images/ directory
            json_files = list(images_dir.glob("*.json"))

            moved_count = 0
            verified_count_checked = 0

            for json_path in json_files:
                # Check if file is stable (not being modified)
                file_age = get_file_age(json_path)

                if file_age < stability_threshold:
                    continue  # Still being modified

                # Check if verified flag is set
                if not is_verified(json_path):
                    continue

                verified_count_checked += 1

                # Get image path (same name, different extension)
                image_name = json_path.stem
                image_path = None
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                    candidate = images_dir / f"{image_name}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break

                if not image_path:
                    logger.warning(f"No image found for {json_path.name}")
                    continue

                # Parse JSON and convert to YOLO format
                yolo_lines, warnings = parse_xanylabeling_json(json_path, class_map)

                if warnings:
                    for warning in warnings:
                        logger.warning(f"{json_path.name}: {warning}")

                if not yolo_lines:
                    logger.warning(f"No valid annotations in {json_path.name}, skipping")
                    continue

                # Create verified subdirectories
                verified_images_dir = verified_dir / 'images'
                verified_labels_dir = verified_dir / 'labels'
                verified_images_dir.mkdir(parents=True, exist_ok=True)
                verified_labels_dir.mkdir(parents=True, exist_ok=True)

                # Prepare paths
                label_path = verified_labels_dir / f"{image_name}.txt"
                image_dst = verified_images_dir / image_path.name
                label_tmp = verified_labels_dir / f"{image_name}.txt.tmp"
                image_tmp = verified_images_dir / f"{image_path.name}.tmp"

                # Atomically move image and create label
                try:
                    # Check if destination already exists
                    if label_path.exists() or image_dst.exists():
                        logger.warning(f"Destination already exists for {image_name}, skipping")
                        continue

                    # Step 1: Write label file with .tmp extension
                    with open(label_tmp, 'w') as f:
                        f.write('\n'.join(yolo_lines) + '\n')

                    # Step 2: Copy image with .tmp extension
                    shutil.copy2(str(image_path), str(image_tmp))

                    # Step 3: Verify both temp files exist
                    if not label_tmp.exists() or not image_tmp.exists():
                        raise IOError("Temp file creation failed")

                    # Step 4: Rename atomically (atomic on POSIX)
                    os.rename(str(label_tmp), str(label_path))
                    os.rename(str(image_tmp), str(image_dst))

                    # Step 5: Delete original image only after both renames succeed
                    image_path.unlink()

                    # Log as verified in tracker
                    tracker.mark_verified(image_path.name)
                    moved_count += 1
                    logger.info(f"✓ Moved: {image_name} → verified/ ({len(yolo_lines)} annotations)")

                except Exception as e:
                    logger.error(f"Error processing {json_path.name}: {e}")

                    # Rollback: remove any files that were created
                    try:
                        if label_path.exists():
                            label_path.unlink()
                        if image_dst.exists():
                            image_dst.unlink()
                        if label_tmp.exists():
                            label_tmp.unlink()
                        if image_tmp.exists():
                            image_tmp.unlink()
                    except Exception as cleanup_err:
                        logger.warning(f"Cleanup error for {image_name}: {cleanup_err}")

            if moved_count > 0:
                verified_labels_dir = verified_dir / 'labels'
                verified_count = len(list(verified_labels_dir.glob("*.txt"))) if verified_labels_dir.exists() else 0
                stats = tracker.get_stats()
                logger.info(f"Moved {moved_count} files. Total verified: {verified_count}")
                logger.info(f"  Progress: {stats['total_verified']} verified / {stats['total']} total ({stats['verification_rate']:.1f}%)")

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
