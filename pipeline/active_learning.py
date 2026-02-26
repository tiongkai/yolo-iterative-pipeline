# pipeline/active_learning.py
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter
import numpy as np
from ultralytics import YOLO

def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """Calculate IoU between two boxes in YOLO format.

    Args:
        box1, box2: (class_id, x, y, w, h) tuples

    Returns:
        IoU score
    """
    _, x1, y1, w1, h1 = box1
    _, x2, y2, w2, h2 = box2

    # Convert to corner coordinates
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2

    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_uncertainty_score(confidences: List[float]) -> float:
    """Calculate uncertainty score from model confidence values.

    Args:
        confidences: List of detection confidence scores

    Returns:
        Uncertainty score [0, 1] where higher = more uncertain
    """
    if not confidences:
        return 1.0  # Maximum uncertainty if no detections

    return 1.0 - np.mean(confidences)


def calculate_disagreement_score(
    model_boxes: List[Tuple],
    sam3_boxes: List[Tuple],
    iou_threshold: float = 0.5
) -> float:
    """Calculate disagreement between model and SAM3 predictions.

    Args:
        model_boxes: Model predictions [(class, x, y, w, h), ...]
        sam3_boxes: SAM3 annotations [(class, x, y, w, h), ...]
        iou_threshold: IoU threshold for matching boxes

    Returns:
        Disagreement score [0, 1] where higher = more disagreement
    """
    if not model_boxes and not sam3_boxes:
        return 0.0  # Both empty, no disagreement

    if not model_boxes or not sam3_boxes:
        return 1.0  # One empty, maximum disagreement

    # Find matches using Hungarian algorithm (simplified greedy approach)
    matched_model = set()
    matched_sam3 = set()
    low_iou_matches = 0

    for i, model_box in enumerate(model_boxes):
        best_iou = 0
        best_j = -1

        for j, sam3_box in enumerate(sam3_boxes):
            if j in matched_sam3:
                continue

            # Only match if same class
            if model_box[0] != sam3_box[0]:
                continue

            iou = calculate_iou(model_box, sam3_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0:
            matched_model.add(i)
            matched_sam3.add(best_j)
            if best_iou < iou_threshold:
                low_iou_matches += 1

    num_missed = len(sam3_boxes) - len(matched_sam3)  # SAM3 boxes not detected
    num_extra = len(model_boxes) - len(matched_model)  # Model extra detections

    total_disagreement = num_missed + num_extra + low_iou_matches
    max_boxes = max(len(model_boxes), len(sam3_boxes))

    return total_disagreement / max_boxes if max_boxes > 0 else 0.0


def calculate_diversity_score(
    detection_count: int,
    count_distribution: Dict[int, int]
) -> float:
    """Calculate diversity score based on detection count rarity.

    Args:
        detection_count: Number of detections in this image
        count_distribution: Histogram of detection counts across dataset

    Returns:
        Diversity score [0, 1] where higher = more diverse/rare
    """
    if not count_distribution:
        return 0.5

    # Find closest bin
    closest_bin = min(count_distribution.keys(), key=lambda k: abs(k - detection_count))
    frequency = count_distribution[closest_bin]

    # Inverse frequency
    total_images = sum(count_distribution.values())
    return 1.0 - (frequency / total_images)


def calculate_priority_score(
    uncertainty: float,
    disagreement: float,
    diversity: float,
    weights: Tuple[float, float, float] = (0.4, 0.35, 0.25)
) -> float:
    """Calculate combined priority score.

    Args:
        uncertainty: Uncertainty score [0, 1]
        disagreement: Disagreement score [0, 1]
        diversity: Diversity score [0, 1]
        weights: (uncertainty_w, disagreement_w, diversity_w)

    Returns:
        Combined priority score [0, 1]
    """
    w_u, w_d, w_v = weights
    return w_u * uncertainty + w_d * disagreement + w_v * diversity


def load_yolo_annotations(label_path: Path) -> List[Tuple]:
    """Load YOLO annotations from file.

    Args:
        label_path: Path to .txt annotation file

    Returns:
        List of (class_id, x, y, w, h) tuples
    """
    boxes = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    boxes.append((class_id, x, y, w, h))
    return boxes


def score_all_images(
    working_dir: Path,
    sam3_dir: Path,
    model_path: Optional[Path],
    weights: Tuple[float, float, float] = (0.4, 0.35, 0.25)
) -> List[Tuple[str, float, float, float, float]]:
    """Score all images in working directory.

    Args:
        working_dir: Directory with images to score
        sam3_dir: Directory with SAM3 annotations
        model_path: Path to trained model (None for iteration 0)
        weights: Active learning weights

    Returns:
        List of (filename, priority, uncertainty, disagreement, diversity)
        sorted by priority (descending)
    """
    # Get detection count distribution
    count_distribution = Counter()
    for label_file in working_dir.glob("*.txt"):
        num_detections = sum(1 for line in open(label_file) if line.strip())
        count_distribution[num_detections] += 1

    # Load model if available
    model = YOLO(str(model_path)) if model_path and model_path.exists() else None

    scores = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_file in working_dir.iterdir():
        if img_file.suffix.lower() not in image_extensions:
            continue

        label_file = working_dir / f"{img_file.stem}.txt"
        sam3_file = sam3_dir / f"{img_file.stem}.txt"

        # Load SAM3 annotations
        sam3_boxes = load_yolo_annotations(sam3_file)

        # Get model predictions
        if model:
            results = model.predict(str(img_file), verbose=False)
            model_boxes = []
            confidences = []

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Convert to YOLO format
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        # Convert xyxy to xywh (normalized)
                        img_h, img_w = result.orig_shape
                        x = ((xyxy[0] + xyxy[2]) / 2) / img_w
                        y = ((xyxy[1] + xyxy[3]) / 2) / img_h
                        w = (xyxy[2] - xyxy[0]) / img_w
                        h = (xyxy[3] - xyxy[1]) / img_h

                        model_boxes.append((cls, x, y, w, h))
                        confidences.append(conf)

            uncertainty = calculate_uncertainty_score(confidences)
            disagreement = calculate_disagreement_score(model_boxes, sam3_boxes)
        else:
            # No model yet, use defaults
            uncertainty = 0.5
            disagreement = calculate_disagreement_score([], sam3_boxes) if sam3_boxes else 0.0

        # Calculate diversity
        num_detections = len(sam3_boxes)
        diversity = calculate_diversity_score(num_detections, count_distribution)

        # Calculate priority
        priority = calculate_priority_score(uncertainty, disagreement, diversity, weights)

        scores.append((img_file.name, priority, uncertainty, disagreement, diversity))

    # Sort by priority (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores


def save_priority_queue(
    scores: List[Tuple[str, float, float, float, float]],
    output_path: Path,
    model_version: Optional[str] = None
):
    """Save priority queue to file.

    Args:
        scores: List of (filename, priority, uncertainty, disagreement, diversity)
        output_path: Path to save priority queue
        model_version: Model version used for scoring
    """
    from datetime import datetime

    with open(output_path, "w") as f:
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        if model_version:
            f.write(f"# Model: {model_version}\n")
        f.write("# Format: filename | priority | uncertainty | disagreement | diversity\n")

        for filename, priority, uncertainty, disagreement, diversity in scores:
            f.write(f"{filename} | {priority:.3f} | {uncertainty:.3f} | "
                   f"{disagreement:.3f} | {diversity:.3f}\n")


def main():
    """CLI entry point for scoring."""
    import argparse

    parser = argparse.ArgumentParser(description="Score images for active learning")
    parser.add_argument("--working-dir", type=Path, default="data/working")
    parser.add_argument("--sam3-dir", type=Path, default="data/sam3_annotations")
    parser.add_argument("--model", type=Path, default="models/active/best.pt")
    parser.add_argument("--output", type=Path, default="logs/priority_queue.txt")
    parser.add_argument("--rescore", action="store_true", help="Force re-scoring")

    args = parser.parse_args()

    print("Scoring images for active learning...")

    model_path = args.model if args.model.exists() else None
    if not model_path:
        print("⚠️  No trained model found, using SAM3 disagreement only")

    scores = score_all_images(
        working_dir=args.working_dir,
        sam3_dir=args.sam3_dir,
        model_path=model_path
    )

    save_priority_queue(scores, args.output)

    print(f"✓ Scored {len(scores)} images")
    print(f"✓ Priority queue saved to {args.output}")

    if scores:
        print(f"\nTop 5 priority images:")
        for filename, priority, _, _, _ in scores[:5]:
            print(f"  {filename}: {priority:.3f}")


if __name__ == "__main__":
    main()
