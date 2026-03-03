"""
Tests for migration script (migrate_to_yolo_layout.py).
"""

import pytest
from pathlib import Path
import shutil
import sys

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from migrate_to_yolo_layout import (
    is_yolo_layout,
    get_file_pairs,
    validate_migration,
    migrate_directory,
    migrate_to_yolo_layout,
    atomic_move_pair,
    MigrationError,
)


@pytest.fixture
def temp_project(tmp_path):
    """Create temporary project structure for testing."""
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True)

    # Create data subdirectories
    for subdir in ["working", "verified", "eval", "test"]:
        (data_dir / subdir).mkdir()

    return project_root


@pytest.fixture
def flat_directory(tmp_path):
    """Create a directory with flat structure."""
    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()

    # Create image/label pairs
    for i in range(3):
        (flat_dir / f"image_{i:03d}.jpg").write_text(f"fake image {i}")
        (flat_dir / f"image_{i:03d}.txt").write_text(f"0 0.5 0.5 0.1 0.1\n")

    # Create orphaned image (no label)
    (flat_dir / "orphan.png").write_text("orphaned image")

    # Create orphaned label (no image)
    (flat_dir / "orphan_label.txt").write_text("1 0.3 0.3 0.2 0.2\n")

    return flat_dir


@pytest.fixture
def yolo_directory(tmp_path):
    """Create a directory with YOLO structure."""
    yolo_dir = tmp_path / "yolo"
    yolo_dir.mkdir()

    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Create image/label pairs in subdirectories
    for i in range(2):
        (images_dir / f"image_{i:03d}.jpg").write_text(f"fake image {i}")
        (labels_dir / f"image_{i:03d}.txt").write_text(f"0 0.5 0.5 0.1 0.1\n")

    return yolo_dir


class TestIsYoloLayout:
    """Tests for is_yolo_layout function."""

    def test_detects_yolo_layout(self, yolo_directory):
        """Should detect YOLO layout (has images/ and labels/ subdirs)."""
        assert is_yolo_layout(yolo_directory)

    def test_detects_flat_layout(self, flat_directory):
        """Should detect flat layout (no images/ and labels/ subdirs)."""
        assert not is_yolo_layout(flat_directory)

    def test_empty_directory(self, tmp_path):
        """Should return False for empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert not is_yolo_layout(empty_dir)

    def test_partial_yolo_layout(self, tmp_path):
        """Should return False if only one subdir exists."""
        partial_dir = tmp_path / "partial"
        partial_dir.mkdir()
        (partial_dir / "images").mkdir()  # Only images/, no labels/
        assert not is_yolo_layout(partial_dir)


class TestGetFilePairs:
    """Tests for get_file_pairs function."""

    def test_finds_pairs(self, flat_directory):
        """Should find all image/label pairs."""
        pairs = get_file_pairs(flat_directory)
        assert len(pairs) == 5  # 3 pairs + 2 orphans

        # Check paired files
        for i in range(3):
            stem = f"image_{i:03d}"
            assert stem in pairs
            assert "image" in pairs[stem]
            assert "label" in pairs[stem]
            assert pairs[stem]["image"].suffix == ".jpg"
            assert pairs[stem]["label"].suffix == ".txt"

    def test_finds_orphaned_image(self, flat_directory):
        """Should find image without corresponding label."""
        pairs = get_file_pairs(flat_directory)
        assert "orphan" in pairs
        assert "image" in pairs["orphan"]
        assert "label" not in pairs["orphan"]

    def test_finds_orphaned_label(self, flat_directory):
        """Should find label without corresponding image."""
        pairs = get_file_pairs(flat_directory)
        assert "orphan_label" in pairs
        assert "label" in pairs["orphan_label"]
        assert "image" not in pairs["orphan_label"]

    def test_handles_different_extensions(self, tmp_path):
        """Should handle different image extensions."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        # Create files with different extensions
        (test_dir / "img1.jpg").write_text("image")
        (test_dir / "img2.JPG").write_text("image")
        (test_dir / "img3.png").write_text("image")
        (test_dir / "img4.PNG").write_text("image")
        (test_dir / "img5.jpeg").write_text("image")
        (test_dir / "img6.JPEG").write_text("image")

        pairs = get_file_pairs(test_dir)
        assert len(pairs) == 6

    def test_ignores_subdirectories(self, yolo_directory):
        """Should only look at files directly in directory."""
        pairs = get_file_pairs(yolo_directory)
        assert len(pairs) == 0  # Should not find files in images/ and labels/


class TestValidateMigration:
    """Tests for validate_migration function."""

    def test_no_warnings_for_clean_migration(self, flat_directory):
        """Should return no warnings for clean migration."""
        pairs = get_file_pairs(flat_directory)
        warnings = validate_migration(flat_directory, pairs)
        # Should warn about orphans
        assert len(warnings) == 2
        assert any("orphan" in w.lower() for w in warnings)

    def test_warns_about_orphaned_files(self, tmp_path):
        """Should warn about orphaned images and labels."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        (test_dir / "orphan_image.jpg").write_text("image")
        (test_dir / "orphan_label.txt").write_text("label")

        pairs = get_file_pairs(test_dir)
        warnings = validate_migration(test_dir, pairs)

        assert len(warnings) == 2
        assert any("without label" in w for w in warnings)
        assert any("without image" in w for w in warnings)

    def test_warns_about_conflicts(self, tmp_path):
        """Should warn about files that already exist in target."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        # Create flat files
        (test_dir / "image_001.jpg").write_text("image")
        (test_dir / "image_001.txt").write_text("label")

        # Create target directories with existing files
        images_dir = test_dir / "images"
        labels_dir = test_dir / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        (images_dir / "image_001.jpg").write_text("existing image")
        (labels_dir / "image_001.txt").write_text("existing label")

        pairs = get_file_pairs(test_dir)
        warnings = validate_migration(test_dir, pairs)

        assert len(warnings) == 2
        assert any("already exists" in w for w in warnings)


class TestMigrateDirectory:
    """Tests for migrate_directory function."""

    def test_successful_migration(self, flat_directory):
        """Should successfully migrate directory to YOLO layout."""
        # Remove orphans to avoid warnings
        (flat_directory / "orphan.png").unlink()
        (flat_directory / "orphan_label.txt").unlink()

        stats = migrate_directory(flat_directory, force=True)

        assert stats["images_moved"] == 3
        assert stats["labels_moved"] == 3
        assert stats["errors"] == 0

        # Verify YOLO layout created
        assert is_yolo_layout(flat_directory)

        # Verify files moved
        images_dir = flat_directory / "images"
        labels_dir = flat_directory / "labels"
        assert len(list(images_dir.glob("*.jpg"))) == 3
        assert len(list(labels_dir.glob("*.txt"))) == 3

        # Verify original files removed
        assert len(list(flat_directory.glob("*.jpg"))) == 0
        assert len(list(flat_directory.glob("*.txt"))) == 0

    def test_skip_already_migrated(self, yolo_directory):
        """Should skip directories already in YOLO layout."""
        stats = migrate_directory(yolo_directory)

        assert stats["images_moved"] == 0
        assert stats["labels_moved"] == 0

    def test_dry_run_no_changes(self, flat_directory):
        """Should not modify files in dry-run mode."""
        # Remove orphans
        (flat_directory / "orphan.png").unlink()
        (flat_directory / "orphan_label.txt").unlink()

        # Get original file list
        original_files = set(f.name for f in flat_directory.iterdir())

        stats = migrate_directory(flat_directory, dry_run=True)

        # Check stats recorded
        assert stats["images_moved"] == 3
        assert stats["labels_moved"] == 3

        # Verify no changes made
        current_files = set(f.name for f in flat_directory.iterdir())
        assert current_files == original_files
        assert not is_yolo_layout(flat_directory)

    def test_backup_mode_copies_files(self, flat_directory):
        """Should copy files in backup mode."""
        # Remove orphans
        (flat_directory / "orphan.png").unlink()
        (flat_directory / "orphan_label.txt").unlink()

        stats = migrate_directory(flat_directory, backup=True, force=True)

        assert stats["images_moved"] == 3
        assert stats["labels_moved"] == 3

        # Verify YOLO layout created
        assert is_yolo_layout(flat_directory)

        # Verify files copied (originals still exist)
        assert len(list(flat_directory.glob("*.jpg"))) == 3
        assert len(list(flat_directory.glob("*.txt"))) == 3

        images_dir = flat_directory / "images"
        labels_dir = flat_directory / "labels"
        assert len(list(images_dir.glob("*.jpg"))) == 3
        assert len(list(labels_dir.glob("*.txt"))) == 3

    def test_handles_missing_directory(self, tmp_path):
        """Should handle missing directory gracefully."""
        missing_dir = tmp_path / "nonexistent"
        stats = migrate_directory(missing_dir)

        assert stats["images_moved"] == 0
        assert stats["labels_moved"] == 0

    def test_handles_empty_directory(self, tmp_path):
        """Should handle empty directory gracefully."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        stats = migrate_directory(empty_dir)

        assert stats["images_moved"] == 0
        assert stats["labels_moved"] == 0

    def test_requires_force_with_warnings(self, tmp_path):
        """Should require --force flag when warnings present."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        # Create orphaned file (will trigger warning)
        (test_dir / "orphan.jpg").write_text("orphan")

        with pytest.raises(MigrationError):
            migrate_directory(test_dir, force=False)

    def test_force_proceeds_despite_warnings(self, tmp_path):
        """Should proceed with --force despite warnings."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        # Create orphaned file
        (test_dir / "orphan.jpg").write_text("orphan")

        stats = migrate_directory(test_dir, force=True)

        assert stats["images_moved"] == 1
        assert stats["warnings"] > 0

    def test_skips_existing_destinations(self, flat_directory):
        """Should skip files if destination already exists."""
        # Remove orphans to avoid warnings
        (flat_directory / "orphan.png").unlink()
        (flat_directory / "orphan_label.txt").unlink()

        # Create YOLO directories
        images_dir = flat_directory / "images"
        labels_dir = flat_directory / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Pre-populate with one existing file to test skip behavior
        (images_dir / "image_000.jpg").write_text("existing image")

        stats = migrate_directory(flat_directory, force=True)

        # Should skip image_000 pair (destination exists), but move others
        assert stats["skipped"] == 1
        assert stats["images_moved"] == 2  # image_001 and image_002 moved
        assert stats["labels_moved"] == 2
        # image_000 originals should still exist (not deleted due to skip)
        assert (flat_directory / "image_000.jpg").exists()
        assert (flat_directory / "image_000.txt").exists()
        # Others should be moved
        assert (images_dir / "image_001.jpg").exists()
        assert (images_dir / "image_002.jpg").exists()
        # Existing file should not be overwritten
        assert (images_dir / "image_000.jpg").read_text() == "existing image"


class TestMigrateToYoloLayout:
    """Tests for migrate_to_yolo_layout function."""

    def test_migrates_multiple_directories(self, temp_project):
        """Should migrate multiple directories."""
        data_dir = temp_project / "data"

        # Create flat structure in working and verified
        for subdir in ["working", "verified"]:
            dir_path = data_dir / subdir
            for i in range(2):
                (dir_path / f"img_{i}.jpg").write_text(f"image {i}")
                (dir_path / f"img_{i}.txt").write_text(f"label {i}")

        results = migrate_to_yolo_layout(
            temp_project,
            directories=["working", "verified"],
            force=True
        )

        assert len(results) == 2
        assert results["working"]["images_moved"] == 2
        assert results["working"]["labels_moved"] == 2
        assert results["verified"]["images_moved"] == 2
        assert results["verified"]["labels_moved"] == 2

    def test_handles_missing_data_directory(self, tmp_path):
        """Should raise error if data directory doesn't exist."""
        with pytest.raises(MigrationError, match="Data directory not found"):
            migrate_to_yolo_layout(tmp_path)

    def test_skips_empty_directories(self, temp_project):
        """Should skip directories with no files."""
        results = migrate_to_yolo_layout(
            temp_project,
            directories=["working", "verified", "eval", "test"]
        )

        # All directories empty, so no files moved
        for dir_name in ["working", "verified", "eval", "test"]:
            assert results[dir_name]["images_moved"] == 0
            assert results[dir_name]["labels_moved"] == 0

    def test_dry_run_mode(self, temp_project):
        """Should not modify files in dry-run mode."""
        data_dir = temp_project / "data"
        working_dir = data_dir / "working"

        # Create flat structure
        (working_dir / "img.jpg").write_text("image")
        (working_dir / "img.txt").write_text("label")

        results = migrate_to_yolo_layout(
            temp_project,
            directories=["working"],
            dry_run=True
        )

        assert results["working"]["images_moved"] == 1
        assert results["working"]["labels_moved"] == 1

        # Verify no changes
        assert not is_yolo_layout(working_dir)
        assert (working_dir / "img.jpg").exists()
        assert (working_dir / "img.txt").exists()


class TestAtomicMovePair:
    """Tests for atomic_move_pair function."""

    def test_atomic_move_success(self, tmp_path):
        """Should successfully move both files atomically."""
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        # Create source files
        label_src = src_dir / "test.txt"
        image_src = src_dir / "test.jpg"
        label_src.write_text("label data")
        image_src.write_text("image data")

        # Define destinations
        label_dst = dst_dir / "test.txt"
        image_dst = dst_dir / "test.jpg"

        # Perform atomic move
        success = atomic_move_pair(label_src, image_src, label_dst, image_dst, backup=False)

        assert success
        assert label_dst.exists()
        assert image_dst.exists()
        assert not label_src.exists()
        assert not image_src.exists()
        assert label_dst.read_text() == "label data"
        assert image_dst.read_text() == "image data"

    def test_atomic_move_backup_mode(self, tmp_path):
        """Should copy files in backup mode."""
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        # Create source files
        label_src = src_dir / "test.txt"
        image_src = src_dir / "test.jpg"
        label_src.write_text("label data")
        image_src.write_text("image data")

        # Define destinations
        label_dst = dst_dir / "test.txt"
        image_dst = dst_dir / "test.jpg"

        # Perform atomic copy (backup mode)
        success = atomic_move_pair(label_src, image_src, label_dst, image_dst, backup=True)

        assert success
        assert label_dst.exists()
        assert image_dst.exists()
        # Originals should still exist in backup mode
        assert label_src.exists()
        assert image_src.exists()

    def test_atomic_move_rollback_on_missing_source(self, tmp_path):
        """Should fail gracefully if source file doesn't exist."""
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        # Create only label source (missing image)
        label_src = src_dir / "test.txt"
        image_src = src_dir / "test.jpg"  # Does not exist
        label_src.write_text("label data")

        # Define destinations
        label_dst = dst_dir / "test.txt"
        image_dst = dst_dir / "test.jpg"

        # Attempt atomic move (should fail due to missing image)
        success = atomic_move_pair(label_src, image_src, label_dst, image_dst, backup=False)

        assert not success
        # Original label should still exist (rollback behavior)
        assert label_src.exists()
        # Destinations should not exist
        assert not label_dst.exists()
        assert not image_dst.exists()
        # No .tmp files should remain
        tmp_files = list(dst_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_atomic_move_handles_partial_failure(self, tmp_path):
        """Should handle case where one file exists in destination."""
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        # Create source files
        label_src = src_dir / "test.txt"
        image_src = src_dir / "test.jpg"
        label_src.write_text("label data")
        image_src.write_text("image data")

        # Create existing destination (label)
        label_dst = dst_dir / "test.txt"
        image_dst = dst_dir / "test.jpg"
        label_dst.write_text("existing label")

        # Attempt atomic move
        # Note: atomic_move_pair doesn't check for existing files (that's migrate_directory's job)
        # So this will overwrite. That's expected - the check happens at a higher level.
        success = atomic_move_pair(label_src, image_src, label_dst, image_dst, backup=False)

        # Should succeed (overwrites are allowed at this level)
        assert success
        assert label_dst.read_text() == "label data"
        assert image_dst.read_text() == "image data"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_unicode_filenames(self, tmp_path):
        """Should handle unicode characters in filenames."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        # Create files with unicode names
        (test_dir / "图像_001.jpg").write_text("image")
        (test_dir / "图像_001.txt").write_text("label")

        stats = migrate_directory(test_dir, force=True)

        assert stats["images_moved"] == 1
        assert stats["labels_moved"] == 1

    def test_handles_special_characters_in_names(self, tmp_path):
        """Should handle special characters in filenames."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        # Create files with special characters
        (test_dir / "img-001_v2.jpg").write_text("image")
        (test_dir / "img-001_v2.txt").write_text("label")

        stats = migrate_directory(test_dir, force=True)

        assert stats["images_moved"] == 1
        assert stats["labels_moved"] == 1

    def test_preserves_file_metadata(self, tmp_path):
        """Should preserve file modification times in backup mode."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        img_file = test_dir / "img.jpg"
        img_file.write_text("image")
        original_mtime = img_file.stat().st_mtime

        migrate_directory(test_dir, backup=True, force=True)

        copied_file = test_dir / "images" / "img.jpg"
        assert copied_file.stat().st_mtime == original_mtime

    def test_handles_case_sensitivity(self, tmp_path):
        """Should handle case-sensitive filesystems correctly."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        # Create files with different cases
        (test_dir / "Image.JPG").write_text("image")
        (test_dir / "Image.txt").write_text("label")

        pairs = get_file_pairs(test_dir)
        assert "Image" in pairs
        assert "image" in pairs["Image"]
        assert "label" in pairs["Image"]
