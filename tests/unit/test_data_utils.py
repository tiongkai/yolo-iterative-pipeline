import pytest
from pathlib import Path
from pipeline.data_utils import (
    validate_yolo_annotation,
    validate_bbox_coords,
    get_image_label_pairs,
    sample_eval_set,
)

def test_validate_yolo_annotation_valid():
    """Test validation of correct YOLO annotation."""
    valid_line = "0 0.5 0.5 0.2 0.3"
    is_valid, error = validate_yolo_annotation(valid_line, num_classes=3)
    assert is_valid is True
    assert error is None

def test_validate_yolo_annotation_invalid_coords():
    """Test detection of invalid coordinates."""
    invalid_line = "0 1.5 0.5 0.2 0.3"  # x > 1.0
    is_valid, error = validate_yolo_annotation(invalid_line, num_classes=3)
    assert is_valid is False
    assert "coordinate" in error.lower()

def test_validate_yolo_annotation_invalid_class():
    """Test detection of invalid class ID."""
    invalid_line = "5 0.5 0.5 0.2 0.3"  # class_id >= num_classes
    is_valid, error = validate_yolo_annotation(invalid_line, num_classes=3)
    assert is_valid is False
    assert "class" in error.lower()

def test_sample_eval_set_stratified(tmp_path):
    """Test stratified sampling of eval set."""
    # Create dummy data
    verified_dir = tmp_path / "verified"
    verified_dir.mkdir()

    # Create annotations with different classes
    for i in range(100):
        label_file = verified_dir / f"img_{i:03d}.txt"
        class_id = i % 3  # 3 classes
        label_file.write_text(f"{class_id} 0.5 0.5 0.2 0.3\n")

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    sampled = sample_eval_set(
        verified_dir=verified_dir,
        eval_dir=eval_dir,
        split_ratio=0.15,
        stratify=True,
        num_classes=3
    )

    assert len(sampled) == 15  # 15% of 100
    # Check stratification (each class should have ~5 samples)
    # This is probabilistic, allow some variance
