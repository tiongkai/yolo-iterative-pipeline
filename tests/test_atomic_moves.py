"""Tests for atomic file move operations and JSON-based verification."""

import pytest
import json
from pathlib import Path
import time
import os
from scripts.auto_move_verified import (
    atomic_move_pair,
    cleanup_tmp_files,
    load_class_mapping,
    validate_bbox_coordinates,
    parse_xanylabeling_json,
    is_verified
)


@pytest.fixture
def setup_files(tmp_path):
    """Create test file structure."""
    working_labels = tmp_path / "working" / "labels"
    working_images = tmp_path / "working" / "images"
    verified_labels = tmp_path / "verified" / "labels"
    verified_images = tmp_path / "verified" / "images"

    working_labels.mkdir(parents=True)
    working_images.mkdir(parents=True)
    verified_labels.mkdir(parents=True)
    verified_images.mkdir(parents=True)

    # Create test files
    label_file = working_labels / "test001.txt"
    image_file = working_images / "test001.png"

    label_file.write_text("0 0.5 0.5 0.1 0.1")
    image_file.write_bytes(b"fake image data")

    return {
        "label_src": label_file,
        "image_src": image_file,
        "label_dst": verified_labels / "test001.txt",
        "image_dst": verified_images / "test001.png",
        "verified_labels": verified_labels,
        "verified_images": verified_images,
    }


def test_atomic_move_pair_success(setup_files):
    """Test atomic_move_pair successfully moves both files."""
    f = setup_files

    result = atomic_move_pair(
        f["label_src"],
        f["image_src"],
        f["label_dst"],
        f["image_dst"]
    )

    assert result is True
    assert f["label_dst"].exists()
    assert f["image_dst"].exists()
    assert not f["label_src"].exists()
    assert not f["image_src"].exists()

    # Verify content preserved
    assert f["label_dst"].read_text() == "0 0.5 0.5 0.1 0.1"
    assert f["image_dst"].read_bytes() == b"fake image data"


def test_atomic_move_pair_no_tmp_files_left(setup_files):
    """Test atomic_move_pair doesn't leave .tmp files behind."""
    f = setup_files

    atomic_move_pair(
        f["label_src"],
        f["image_src"],
        f["label_dst"],
        f["image_dst"]
    )

    # Check for .tmp files
    tmp_files = list(f["verified_labels"].glob("*.tmp"))
    tmp_files.extend(f["verified_images"].glob("*.tmp"))

    assert len(tmp_files) == 0


def test_atomic_move_pair_rollback_on_failure(setup_files):
    """Test atomic_move_pair rolls back if second file missing."""
    f = setup_files

    # Delete image to cause failure
    f["image_src"].unlink()

    result = atomic_move_pair(
        f["label_src"],
        f["image_src"],
        f["label_dst"],
        f["image_dst"]
    )

    assert result is False
    # Original label should still exist (rollback)
    assert f["label_src"].exists()
    # Destination should not have partial files
    assert not f["label_dst"].exists()
    assert not f["image_dst"].exists()


def test_cleanup_tmp_files(tmp_path):
    """Test cleanup_tmp_files removes stale .tmp files."""
    verified_labels = tmp_path / "verified" / "labels"
    verified_images = tmp_path / "verified" / "images"
    verified_labels.mkdir(parents=True)
    verified_images.mkdir(parents=True)

    # Create stale .tmp files
    (verified_labels / "old1.txt.tmp").write_text("stale")
    (verified_images / "old1.png.tmp").write_bytes(b"stale")
    (verified_labels / "old2.txt.tmp").write_text("stale")

    # Create normal files (should not be deleted)
    (verified_labels / "good.txt").write_text("keep")

    cleanup_tmp_files(tmp_path / "verified")

    # Check .tmp files removed
    assert not (verified_labels / "old1.txt.tmp").exists()
    assert not (verified_images / "old1.png.tmp").exists()
    assert not (verified_labels / "old2.txt.tmp").exists()

    # Check normal files kept
    assert (verified_labels / "good.txt").exists()


def test_atomic_move_pair_partial_rename_failure(setup_files, monkeypatch):
    """Test rollback when second rename fails after first succeeds."""
    f = setup_files

    # Mock os.rename to fail on second call
    rename_count = 0
    original_rename = os.rename

    def mock_rename(src, dst):
        nonlocal rename_count
        rename_count += 1
        if rename_count == 2:
            raise OSError("Simulated disk full")
        return original_rename(src, dst)

    monkeypatch.setattr("os.rename", mock_rename)

    result = atomic_move_pair(
        f["label_src"],
        f["image_src"],
        f["label_dst"],
        f["image_dst"]
    )

    # Should fail
    assert result is False

    # Source files should still exist (rollback successful)
    assert f["label_src"].exists()
    assert f["image_src"].exists()

    # Destination should be clean (no partial files)
    assert not f["label_dst"].exists()
    assert not f["image_dst"].exists()


def test_atomic_move_pair_destination_exists(setup_files):
    """Test atomic_move_pair fails when destination already exists."""
    f = setup_files

    # Create existing destination file
    f["label_dst"].write_text("existing content")

    result = atomic_move_pair(
        f["label_src"],
        f["image_src"],
        f["label_dst"],
        f["image_dst"]
    )

    # Should fail
    assert result is False

    # Source files should still exist
    assert f["label_src"].exists()
    assert f["image_src"].exists()

    # Existing destination should be unchanged
    assert f["label_dst"].read_text() == "existing content"


# ============================================================================
# JSON-based verification tests
# ============================================================================

def test_load_class_mapping(tmp_path):
    """Test load_class_mapping parses classes.txt correctly."""
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("boat\nhuman\noutboard motor\n")

    class_map = load_class_mapping(classes_file)

    assert class_map == {
        "boat": 0,
        "human": 1,
        "outboard motor": 2
    }


def test_load_class_mapping_skips_empty_lines(tmp_path):
    """Test load_class_mapping skips empty lines."""
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("boat\n\nhuman\n\n\noutboard motor\n")

    class_map = load_class_mapping(classes_file)

    assert class_map == {
        "boat": 0,
        "human": 1,
        "outboard motor": 2
    }


def test_load_class_mapping_file_not_found(tmp_path):
    """Test load_class_mapping raises error when file missing."""
    classes_file = tmp_path / "nonexistent.txt"

    with pytest.raises(FileNotFoundError):
        load_class_mapping(classes_file)


def test_validate_bbox_coordinates_valid():
    """Test validate_bbox_coordinates accepts valid coordinates."""
    points = [[10, 20], [50, 20], [50, 60], [10, 60]]
    assert validate_bbox_coordinates(points, 100, 100) is True


def test_validate_bbox_coordinates_out_of_bounds():
    """Test validate_bbox_coordinates rejects out-of-bounds coordinates."""
    points = [[10, 20], [150, 20], [150, 60], [10, 60]]
    assert validate_bbox_coordinates(points, 100, 100) is False


def test_validate_bbox_coordinates_invalid_structure():
    """Test validate_bbox_coordinates rejects invalid structure."""
    # Wrong number of points
    points = [[10, 20], [50, 20]]
    assert validate_bbox_coordinates(points, 100, 100) is False

    # Wrong point structure
    points = [[10], [50, 20], [50, 60], [10, 60]]
    assert validate_bbox_coordinates(points, 100, 100) is False


def test_is_verified_true(tmp_path):
    """Test is_verified returns True when flag is set."""
    json_file = tmp_path / "test.json"
    json_data = {
        "flags": {"verified": True},
        "shapes": []
    }
    json_file.write_text(json.dumps(json_data))

    assert is_verified(json_file) is True


def test_is_verified_false(tmp_path):
    """Test is_verified returns False when flag is not set."""
    json_file = tmp_path / "test.json"
    json_data = {
        "flags": {"verified": False},
        "shapes": []
    }
    json_file.write_text(json.dumps(json_data))

    assert is_verified(json_file) is False


def test_is_verified_missing_flag(tmp_path):
    """Test is_verified returns False when flag is missing."""
    json_file = tmp_path / "test.json"
    json_data = {
        "flags": {},
        "shapes": []
    }
    json_file.write_text(json.dumps(json_data))

    assert is_verified(json_file) is False


def test_parse_xanylabeling_json_success(tmp_path):
    """Test parse_xanylabeling_json converts JSON to YOLO format."""
    json_file = tmp_path / "test.json"
    json_data = {
        "imageWidth": 640,
        "imageHeight": 480,
        "shapes": [
            {
                "label": "boat",
                "shape_type": "rectangle",
                "points": [[100, 100], [200, 100], [200, 200], [100, 200]]
            }
        ]
    }
    json_file.write_text(json.dumps(json_data))

    class_map = {"boat": 0, "human": 1}
    yolo_lines, warnings = parse_xanylabeling_json(json_file, class_map)

    assert len(yolo_lines) == 1
    assert len(warnings) == 0

    # Parse YOLO line: class_id center_x center_y width height
    parts = yolo_lines[0].split()
    assert parts[0] == "0"  # boat class_id
    assert float(parts[1]) == pytest.approx(0.234375, rel=1e-5)  # center_x = 150/640
    assert float(parts[2]) == pytest.approx(0.3125, rel=1e-5)    # center_y = 150/480
    assert float(parts[3]) == pytest.approx(0.15625, rel=1e-5)   # width = 100/640
    assert float(parts[4]) == pytest.approx(0.208333, rel=1e-5)  # height = 100/480


def test_parse_xanylabeling_json_unknown_class(tmp_path):
    """Test parse_xanylabeling_json warns on unknown class."""
    json_file = tmp_path / "test.json"
    json_data = {
        "imageWidth": 640,
        "imageHeight": 480,
        "shapes": [
            {
                "label": "unknown_class",
                "shape_type": "rectangle",
                "points": [[100, 100], [200, 100], [200, 200], [100, 200]]
            }
        ]
    }
    json_file.write_text(json.dumps(json_data))

    class_map = {"boat": 0, "human": 1}
    yolo_lines, warnings = parse_xanylabeling_json(json_file, class_map)

    assert len(yolo_lines) == 0
    assert len(warnings) == 1
    assert "Unknown class label" in warnings[0]


def test_parse_xanylabeling_json_invalid_coordinates(tmp_path):
    """Test parse_xanylabeling_json warns on invalid coordinates."""
    json_file = tmp_path / "test.json"
    json_data = {
        "imageWidth": 640,
        "imageHeight": 480,
        "shapes": [
            {
                "label": "boat",
                "shape_type": "rectangle",
                "points": [[100, 100], [700, 100], [700, 200], [100, 200]]  # x > width
            }
        ]
    }
    json_file.write_text(json.dumps(json_data))

    class_map = {"boat": 0}
    yolo_lines, warnings = parse_xanylabeling_json(json_file, class_map)

    assert len(yolo_lines) == 0
    assert len(warnings) == 1
    assert "Invalid bbox coordinates" in warnings[0]


def test_parse_xanylabeling_json_skips_non_rectangle(tmp_path):
    """Test parse_xanylabeling_json skips non-rectangle shapes."""
    json_file = tmp_path / "test.json"
    json_data = {
        "imageWidth": 640,
        "imageHeight": 480,
        "shapes": [
            {
                "label": "boat",
                "shape_type": "polygon",
                "points": [[100, 100], [200, 100], [150, 200]]
            }
        ]
    }
    json_file.write_text(json.dumps(json_data))

    class_map = {"boat": 0}
    yolo_lines, warnings = parse_xanylabeling_json(json_file, class_map)

    assert len(yolo_lines) == 0
    assert len(warnings) == 1
    assert "non-rectangle" in warnings[0]


def test_parse_xanylabeling_json_multiple_shapes(tmp_path):
    """Test parse_xanylabeling_json handles multiple shapes."""
    json_file = tmp_path / "test.json"
    json_data = {
        "imageWidth": 640,
        "imageHeight": 480,
        "shapes": [
            {
                "label": "boat",
                "shape_type": "rectangle",
                "points": [[100, 100], [200, 100], [200, 200], [100, 200]]
            },
            {
                "label": "human",
                "shape_type": "rectangle",
                "points": [[300, 150], [350, 150], [350, 250], [300, 250]]
            }
        ]
    }
    json_file.write_text(json.dumps(json_data))

    class_map = {"boat": 0, "human": 1}
    yolo_lines, warnings = parse_xanylabeling_json(json_file, class_map)

    assert len(yolo_lines) == 2
    assert len(warnings) == 0
    assert yolo_lines[0].startswith("0")  # boat
    assert yolo_lines[1].startswith("1")  # human
