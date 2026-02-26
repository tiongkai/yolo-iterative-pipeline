import pytest
from pathlib import Path
from pipeline.data_utils import (
    validate_yolo_annotation,
    validate_bbox_coords,
    validate_annotation_file,
    get_image_label_pairs,
    get_class_distribution,
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

def test_validate_yolo_annotation_zero_width():
    """Test detection of zero-width box."""
    invalid_line = "0 0.5 0.5 0.0 0.3"  # w = 0
    is_valid, error = validate_yolo_annotation(invalid_line, num_classes=3)
    assert is_valid is False
    assert "width" in error.lower()

def test_validate_annotation_file(tmp_path):
    """Test annotation file validation."""
    label_file = tmp_path / "test.txt"
    label_file.write_text("0 0.5 0.5 0.2 0.3\n5 1.5 0.5 0.2 0.3\n", encoding='utf-8')
    is_valid, errors = validate_annotation_file(label_file, num_classes=3)
    assert not is_valid
    assert len(errors) >= 1

def test_get_class_distribution(tmp_path):
    """Test class distribution calculation."""
    label_file = tmp_path / "test.txt"
    label_file.write_text("0 0.5 0.5 0.2 0.3\n0 0.3 0.3 0.1 0.1\n1 0.7 0.7 0.2 0.2\n", encoding='utf-8')
    dist = get_class_distribution([label_file])
    assert dist[0] == 2
    assert dist[1] == 1

def test_get_image_label_pairs(tmp_path):
    """Test image-label pair matching."""
    img_file = tmp_path / "img1.jpg"
    label_file = tmp_path / "img1.txt"
    img_file.touch()
    label_file.touch()
    pairs = get_image_label_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0] == (img_file, label_file)

def test_sample_eval_set_stratified(tmp_path):
    """Test stratified sampling of eval set."""
    # Create dummy data
    verified_dir = tmp_path / "verified"
    verified_dir.mkdir()

    # Create annotations with different classes
    for i in range(100):
        label_file = verified_dir / f"img_{i:03d}.txt"
        class_id = i % 3  # 3 classes
        label_file.write_text(f"{class_id} 0.5 0.5 0.2 0.3\n", encoding='utf-8')

    # Create corresponding image files
    for i in range(100):
        img_file = verified_dir / f"img_{i:03d}.jpg"
        img_file.touch()  # Create dummy image file

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

    # Verify stratification - count classes in eval set
    class_dist = {}
    for lf in eval_dir.glob("*.txt"):
        cid = int(lf.read_text().split()[0])
        class_dist[cid] = class_dist.get(cid, 0) + 1

    # Each class should have approximately 5 samples (15 / 3 classes)
    # Allow variance of ±2 samples
    for cid in range(3):
        assert 3 <= class_dist.get(cid, 0) <= 7

    # Verify image files were also moved
    for lf in eval_dir.glob("*.txt"):
        img_file = eval_dir / f"{lf.stem}.jpg"
        assert img_file.exists(), f"Image file not moved for {lf.name}"

def test_sample_eval_set_invalid_split_ratio():
    """Test validation of invalid split_ratio."""
    with pytest.raises(ValueError, match="split_ratio must be between 0 and 1"):
        sample_eval_set(
            verified_dir=Path("dummy"),
            eval_dir=Path("dummy"),
            split_ratio=1.5
        )

def test_sample_eval_set_nonexistent_dir():
    """Test validation of non-existent directory."""
    with pytest.raises(FileNotFoundError, match="Directory does not exist"):
        sample_eval_set(
            verified_dir=Path("/nonexistent/path"),
            eval_dir=Path("dummy"),
            split_ratio=0.15
        )
