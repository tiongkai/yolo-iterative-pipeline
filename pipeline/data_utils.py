from pathlib import Path
from typing import List, Tuple, Optional, Dict
import shutil
import random
from collections import defaultdict

def validate_bbox_coords(x: float, y: float, w: float, h: float) -> Tuple[bool, Optional[str]]:
    """Validate YOLO bounding box coordinates.

    Args:
        x, y: Center coordinates (normalized 0-1)
        w, h: Width and height (normalized 0-1)

    Returns:
        (is_valid, error_message)
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
    with open(label_path) as f:
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
        with open(label_file) as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

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
    random.seed(random_seed)

    # Get all label files
    label_files = list(verified_dir.glob("*.txt"))
    n_samples = int(len(label_files) * split_ratio)

    if n_samples == 0:
        return []

    if stratify and num_classes:
        # Group files by primary class (most frequent in file)
        class_groups = defaultdict(list)
        for label_file in label_files:
            class_counts = defaultdict(int)
            with open(label_file) as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1

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

    # Move sampled files to eval directory
    eval_dir.mkdir(parents=True, exist_ok=True)
    for label_file in sampled:
        dest = eval_dir / label_file.name
        shutil.move(str(label_file), str(dest))

    return sampled
