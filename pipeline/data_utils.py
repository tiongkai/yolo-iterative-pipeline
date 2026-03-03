from pathlib import Path
from typing import List, Tuple, Optional, Dict
import shutil
import random
from collections import defaultdict

def validate_bbox_coords(x: float, y: float, w: float, h: float) -> Tuple[bool, Optional[str]]:
    """Validate YOLO bounding box coordinates.

    Args:
        x, y: Center coordinates (normalized 0-1, inclusive)
        w, h: Width and height (normalized 0-1, exclusive of 0)

    Returns:
        (is_valid, error_message)

    Note:
        - Boxes can touch image edges (x=0, y=0, x=1, y=1 are valid)
        - Zero-width or zero-height boxes are invalid (w > 0, h > 0)
    """
    if not (0 <= x <= 1):
        return False, f"Center x coordinate={x} out of range [0, 1]"
    if not (0 <= y <= 1):
        return False, f"Center y coordinate={y} out of range [0, 1]"
    if not (0 < w <= 1):
        return False, f"Width coordinate={w} out of range (0, 1]"
    if not (0 < h <= 1):
        return False, f"Height coordinate={h} out of range (0, 1]"
    return True, None


def validate_yolo_annotation(line: str, num_classes: int) -> Tuple[bool, Optional[str]]:
    """Validate a single YOLO annotation line.

    Args:
        line: YOLO format line "class_id x y w h"
        num_classes: Number of valid classes

    Returns:
        (is_valid, error_message)
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return False, f"Expected 5 values, got {len(parts)}"

    try:
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])
    except ValueError as e:
        return False, f"Parse error: {e}"

    if not (0 <= class_id < num_classes):
        return False, f"Class ID {class_id} out of range [0, {num_classes})"

    return validate_bbox_coords(x, y, w, h)


def validate_annotation_file(
    label_path: Path,
    num_classes: int
) -> Tuple[bool, List[str]]:
    """Validate an entire YOLO annotation file.

    Args:
        label_path: Path to .txt annotation file
        num_classes: Number of valid classes

    Returns:
        (is_valid, list_of_errors)
    """
    if not label_path.exists():
        return False, ["File does not exist"]

    errors = []
    with open(label_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():  # Skip empty lines
                is_valid, error = validate_yolo_annotation(line, num_classes)
                if not is_valid:
                    errors.append(f"Line {line_num}: {error}")

    return len(errors) == 0, errors


def get_image_label_pairs(
    image_dir: Path,
    label_dir: Optional[Path] = None
) -> List[Tuple[Path, Path]]:
    """Get matching image-label file pairs.

    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels (defaults to image_dir)

    Returns:
        List of (image_path, label_path) tuples
    """
    if label_dir is None:
        label_dir = image_dir

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs = []

    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() in image_extensions:
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                pairs.append((img_path, label_path))

    return pairs


def get_class_distribution(label_files: List[Path]) -> Dict[int, int]:
    """Get class distribution from label files.

    Args:
        label_files: List of YOLO annotation files

    Returns:
        Dictionary mapping class_id to count
    """
    class_counts = defaultdict(int)

    for label_file in label_files:
        try:
            with open(label_file, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            parts = line.split()
                            if len(parts) >= 1:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                        except (ValueError, IndexError):
                            continue  # Skip invalid lines
        except (IOError, OSError):
            continue  # Skip corrupted files

    return dict(class_counts)


def sample_eval_set(
    verified_dir: Path,
    eval_dir: Path,
    split_ratio: float = 0.15,
    stratify: bool = True,
    num_classes: Optional[int] = None,
    random_seed: int = 42
) -> List[Path]:
    """Sample eval set from verified annotations.

    Args:
        verified_dir: Directory with verified annotations
        eval_dir: Destination directory for eval set
        split_ratio: Fraction to sample (0.15 = 15%)
        stratify: Whether to stratify by class
        num_classes: Number of classes (for stratification)
        random_seed: Random seed for reproducibility

    Returns:
        List of sampled label file paths
    """
    # Input validation
    if not 0 < split_ratio < 1:
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")
    if not verified_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {verified_dir}")

    random.seed(random_seed)

    # Get all label files (expects verified/labels/*.txt structure)
    labels_dir = verified_dir / 'labels'
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    label_files = list(labels_dir.glob("*.txt"))
    n_samples = int(len(label_files) * split_ratio)

    if n_samples == 0:
        return []

    if stratify and num_classes:
        # Group files by primary class (most frequent in file)
        class_groups = defaultdict(list)
        for label_file in label_files:
            class_counts = defaultdict(int)
            try:
                with open(label_file, encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                parts = line.split()
                                if len(parts) >= 1:
                                    class_id = int(parts[0])
                                    class_counts[class_id] += 1
                            except (ValueError, IndexError):
                                continue  # Skip invalid lines
            except (IOError, OSError):
                continue  # Skip corrupted files

            if class_counts:
                primary_class = max(class_counts, key=class_counts.get)
                class_groups[primary_class].append(label_file)

        # Sample proportionally from each class
        sampled = []
        remaining_files = []
        for class_id in sorted(class_groups.keys()):
            group = class_groups[class_id]
            n_class_samples = max(1, int(len(group) * split_ratio))
            class_sampled = random.sample(group, min(n_class_samples, len(group)))
            sampled.extend(class_sampled)
            # Keep track of files not sampled for padding if needed
            remaining_files.extend([f for f in group if f not in class_sampled])

        # If we didn't get enough samples due to rounding, pad from remaining
        if len(sampled) < n_samples and remaining_files:
            additional = random.sample(remaining_files, min(n_samples - len(sampled), len(remaining_files)))
            sampled.extend(additional)

        # Trim to exact sample size (in case we went over)
        sampled = sampled[:n_samples]
    else:
        # Simple random sampling
        sampled = random.sample(label_files, n_samples)

    # Move sampled files to eval directory (with images/labels structure)
    eval_labels_dir = eval_dir / 'labels'
    eval_images_dir = eval_dir / 'images'
    eval_labels_dir.mkdir(parents=True, exist_ok=True)
    eval_images_dir.mkdir(parents=True, exist_ok=True)

    images_dir = verified_dir / 'images'
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    for label_file in sampled:
        # Move label file
        dest = eval_labels_dir / label_file.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(label_file), str(dest))

        # Find and move corresponding image file (case-insensitive)
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        found = False
        for ext in image_extensions:
            if found:
                break
            # Try both lowercase and uppercase
            for case_ext in [ext, ext.upper()]:
                img_file = images_dir / f"{label_file.stem}{case_ext}"
                if img_file.exists():
                    dest_img = eval_images_dir / img_file.name
                    if dest_img.exists():
                        dest_img.unlink()
                    shutil.move(str(img_file), str(dest_img))
                    found = True
                    break

    return sampled
