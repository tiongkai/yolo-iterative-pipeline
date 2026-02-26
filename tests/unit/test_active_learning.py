# tests/unit/test_active_learning.py
import pytest
import numpy as np
from pathlib import Path
from pipeline.active_learning import (
    calculate_iou,
    calculate_uncertainty_score,
    calculate_disagreement_score,
    calculate_diversity_score,
    calculate_priority_score,
    load_yolo_annotations,
    score_all_images,
)

def test_calculate_uncertainty_score():
    """Test uncertainty score calculation."""
    confidences = [0.9, 0.8, 0.7]
    score = calculate_uncertainty_score(confidences)
    # uncertainty = 1 - mean(confidences) = 1 - 0.8 = 0.2
    assert abs(score - 0.2) < 0.01

def test_calculate_uncertainty_score_no_detections():
    """Test uncertainty score with no detections."""
    score = calculate_uncertainty_score([])
    assert score == 1.0  # Maximum uncertainty

def test_calculate_disagreement_score():
    """Test disagreement score between model and SAM3."""
    model_boxes = [(0, 0.5, 0.5, 0.2, 0.3), (1, 0.3, 0.3, 0.1, 0.1)]
    sam3_boxes = [(0, 0.5, 0.5, 0.2, 0.3), (2, 0.8, 0.8, 0.1, 0.1)]

    score = calculate_disagreement_score(model_boxes, sam3_boxes, iou_threshold=0.5)
    # 1 match, 1 model-only, 1 sam3-only → disagreement
    assert score > 0

def test_calculate_diversity_score():
    """Test diversity score based on detection count."""
    detection_count = 5
    count_distribution = {0: 10, 1: 20, 5: 2, 10: 15}  # 5 is rare

    score = calculate_diversity_score(detection_count, count_distribution)
    assert score > 0.5  # Rare count should have high diversity

def test_calculate_priority_score():
    """Test combined priority score calculation."""
    score = calculate_priority_score(
        uncertainty=0.8,
        disagreement=0.6,
        diversity=0.4,
        weights=(0.4, 0.35, 0.25)
    )
    expected = 0.4 * 0.8 + 0.35 * 0.6 + 0.25 * 0.4
    assert abs(score - expected) < 0.01

def test_calculate_iou_edge_cases():
    """Test IoU calculation edge cases."""
    # Non-overlapping boxes
    assert calculate_iou((0, 0.2, 0.2, 0.1, 0.1), (0, 0.8, 0.8, 0.1, 0.1)) == 0.0
    # Identical boxes
    assert abs(calculate_iou((0, 0.5, 0.5, 0.2, 0.2), (0, 0.5, 0.5, 0.2, 0.2)) - 1.0) < 0.0001
    # Partial overlap
    iou = calculate_iou((0, 0.5, 0.5, 0.4, 0.4), (0, 0.6, 0.6, 0.4, 0.4))
    assert 0 < iou < 1

def test_load_yolo_annotations_malformed(tmp_path):
    """Test annotation loading with malformed data."""
    label_file = tmp_path / "test.txt"
    label_file.write_text("invalid data here\n", encoding='utf-8')
    boxes = load_yolo_annotations(label_file)
    assert boxes == []  # Should handle gracefully

def test_load_yolo_annotations_valid(tmp_path):
    """Test annotation loading with valid data."""
    label_file = tmp_path / "test.txt"
    label_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.7 0.7 0.1 0.1\n", encoding='utf-8')
    boxes = load_yolo_annotations(label_file)
    assert len(boxes) == 2
    assert boxes[0] == (0, 0.5, 0.5, 0.2, 0.3)
    assert boxes[1] == (1, 0.7, 0.7, 0.1, 0.1)
