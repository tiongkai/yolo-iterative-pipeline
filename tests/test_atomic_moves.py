"""Tests for atomic file move operations."""

import pytest
from pathlib import Path
import time
import os
from scripts.auto_move_verified import atomic_move_pair, cleanup_tmp_files


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
