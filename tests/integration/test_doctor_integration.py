"""Integration tests for doctor command.

Tests the full doctor command end-to-end, verifying it correctly detects
and reports various pipeline issues.
"""

import pytest
import subprocess
from pathlib import Path
import yaml


@pytest.fixture
def valid_pipeline_structure(tmp_path):
    """Create complete valid YOLO pipeline structure.

    Returns:
        Path to pipeline root directory with all required directories
    """
    # Create all 14 required directories with YOLO layout
    (tmp_path / "data" / "working" / "images").mkdir(parents=True)
    (tmp_path / "data" / "working" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "verified" / "images").mkdir(parents=True)
    (tmp_path / "data" / "verified" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "images").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "test" / "images").mkdir(parents=True)
    (tmp_path / "data" / "test" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "models" / "active").mkdir(parents=True)
    (tmp_path / "models" / "checkpoints").mkdir(parents=True)
    (tmp_path / "models" / "deployed").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)

    # Create classes.txt in verified and working directories
    (tmp_path / "data" / "verified" / "classes.txt").write_text("boat\nhuman\nmotor\n")
    (tmp_path / "data" / "working" / "classes.txt").write_text("boat\nhuman\nmotor\n")

    # Create valid pipeline_config.yaml
    pipeline_config = {
        "project_name": "test_project",
        "classes": ["boat", "human", "motor"],
        "trigger_threshold": 50,
        "eval_split_ratio": 0.15,
        "early_trigger": 25
    }
    with open(tmp_path / "configs" / "pipeline_config.yaml", "w") as f:
        yaml.dump(pipeline_config, f)

    # Create valid yolo_config.yaml
    yolo_config = {
        "model": "yolo11n.pt",
        "imgsz": 1280,
        "batch": 16,
        "epochs": 100,
        "patience": 10,
        "device": "0,1"
    }
    with open(tmp_path / "configs" / "yolo_config.yaml", "w") as f:
        yaml.dump(yolo_config, f)

    # Create data.yaml (required by YOLO)
    data_yaml = {
        "path": str(tmp_path / "data"),
        "train": "verified/images",
        "val": "eval/images",
        "test": "test/images",
        "names": {0: "boat", 1: "human", 2: "motor"}
    }
    with open(tmp_path / "data" / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    return tmp_path


@pytest.fixture
def pipeline_with_annotations(valid_pipeline_structure):
    """Add valid YOLO annotations to pipeline structure.

    Args:
        valid_pipeline_structure: Base pipeline structure

    Returns:
        Path to pipeline root with sample annotations
    """
    root = valid_pipeline_structure

    # Add sample images
    verified_images = root / "data" / "verified" / "images"
    (verified_images / "image1.jpg").touch()
    (verified_images / "image2.jpg").touch()

    # Add valid YOLO annotations
    verified_labels = root / "data" / "verified" / "labels"

    # Valid annotation for image1.jpg (2 objects)
    with open(verified_labels / "image1.txt", "w") as f:
        f.write("0 0.5 0.5 0.3 0.4\n")  # boat at center
        f.write("1 0.7 0.3 0.1 0.1\n")  # human at top-right

    # Valid annotation for image2.jpg (1 object)
    with open(verified_labels / "image2.txt", "w") as f:
        f.write("2 0.2 0.8 0.15 0.2\n")  # motor at bottom-left

    return root


def test_valid_pipeline_structure(valid_pipeline_structure):
    """Test doctor command reports healthy for valid structure."""
    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=valid_pipeline_structure,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should exit with success
    assert result.returncode == 0, f"Expected exit code 0, got {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"

    # Should report healthy
    assert "✅ Pipeline is healthy" in result.stdout
    assert "All checks passed" in result.stdout

    # Should show all validation sections
    assert "Structure Validation" in result.stdout
    assert "Configuration Validation" in result.stdout
    assert "Annotation Validation" in result.stdout
    assert "Model Validation" in result.stdout


def test_valid_pipeline_with_annotations(pipeline_with_annotations):
    """Test doctor validates annotations correctly."""
    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=pipeline_with_annotations,
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0
    assert "✅ Pipeline is healthy" in result.stdout

    # Should mention annotation validation passed
    assert "✅ Annotation Validation" in result.stdout


def test_missing_required_directories(tmp_path):
    """Test doctor detects missing directories."""
    # Create incomplete structure - missing data/verified/labels
    (tmp_path / "data" / "working" / "images").mkdir(parents=True)
    (tmp_path / "data" / "working" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "verified" / "images").mkdir(parents=True)
    # NOTE: Missing data/verified/labels
    (tmp_path / "data" / "eval" / "images").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "test" / "images").mkdir(parents=True)
    (tmp_path / "data" / "test" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "models" / "active").mkdir(parents=True)
    (tmp_path / "models" / "checkpoints").mkdir(parents=True)
    (tmp_path / "models" / "deployed").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)

    # Create minimal configs
    pipeline_config = {
        "project_name": "test",
        "classes": ["boat"],
        "trigger_threshold": 50
    }
    with open(tmp_path / "configs" / "pipeline_config.yaml", "w") as f:
        yaml.dump(pipeline_config, f)

    yolo_config = {"model": "yolo11n.pt"}
    with open(tmp_path / "configs" / "yolo_config.yaml", "w") as f:
        yaml.dump(yolo_config, f)

    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should exit with error code
    assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}\nstdout: {result.stdout}"

    # Should report error
    assert "❌ Pipeline has errors" in result.stdout

    # Should mention the missing directory
    assert "data/verified/labels" in result.stdout
    assert "Missing required directory" in result.stdout or "✗" in result.stdout


def test_multiple_missing_directories(tmp_path):
    """Test doctor detects multiple missing directories."""
    # Create very incomplete structure
    (tmp_path / "data" / "working" / "images").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)

    # Create minimal config
    pipeline_config = {
        "project_name": "test",
        "classes": ["boat"],
        "trigger_threshold": 50
    }
    with open(tmp_path / "configs" / "pipeline_config.yaml", "w") as f:
        yaml.dump(pipeline_config, f)

    yolo_config = {"model": "yolo11n.pt"}
    with open(tmp_path / "configs" / "yolo_config.yaml", "w") as f:
        yaml.dump(yolo_config, f)

    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 1
    assert "❌ Pipeline has errors" in result.stdout

    # Should report multiple missing directories
    # At minimum should be missing verified/labels, test dirs, eval dirs, models dirs
    assert result.stdout.count("Missing required directory") >= 3 or result.stdout.count("✗") >= 3


def test_invalid_annotations(valid_pipeline_structure):
    """Test doctor detects invalid YOLO annotations."""
    root = valid_pipeline_structure

    # Create invalid annotations
    verified_labels = root / "data" / "verified" / "labels"
    verified_images = root / "data" / "verified" / "images"

    (verified_images / "bad1.jpg").touch()
    (verified_images / "bad2.jpg").touch()

    # Wrong format - only 3 values instead of 5
    with open(verified_labels / "bad1.txt", "w") as f:
        f.write("0 0.5 0.5\n")

    # Out of range coordinates
    with open(verified_labels / "bad2.txt", "w") as f:
        f.write("0 1.5 0.5 0.3 0.4\n")  # x_center = 1.5 (out of [0,1])

    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should exit with error
    assert result.returncode == 1

    # Should report annotation errors
    assert "❌" in result.stdout  # Error symbol
    assert ("Invalid format" in result.stdout or
            "expected 5 values" in result.stdout or
            "out of range" in result.stdout)


def test_orphaned_files(valid_pipeline_structure):
    """Test doctor detects orphaned files in parent directories."""
    root = valid_pipeline_structure

    # Create orphaned files in parent directory (should be in images/labels subdirs)
    verified_dir = root / "data" / "verified"
    (verified_dir / "orphan1.jpg").touch()
    (verified_dir / "orphan2.txt").touch()

    working_dir = root / "data" / "working"
    (working_dir / "orphan3.png").touch()

    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Warnings don't cause failure - should still exit 0
    assert result.returncode == 0, f"Expected exit code 0 (warnings don't fail), got {result.returncode}\nstdout: {result.stdout}"

    # Should report warnings but still be runnable
    assert "⚠️  Pipeline has warnings" in result.stdout or "⚠" in result.stdout

    # Should show Structure Validation with warning symbol
    # Note: The specific orphaned files message may be truncated in output (shows first 5 messages)
    # but the section should still have warning status
    assert "⚠️  Structure Validation" in result.stdout or "⚠ Structure Validation" in result.stdout


def test_missing_config_file(tmp_path):
    """Test doctor detects missing configuration files."""
    # Create structure without configs/pipeline_config.yaml
    (tmp_path / "data" / "verified" / "images").mkdir(parents=True)
    (tmp_path / "data" / "verified" / "labels").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    # NOTE: Not creating pipeline_config.yaml

    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should exit with error
    assert result.returncode == 1

    # Should mention missing config
    assert "pipeline_config.yaml not found" in result.stdout or "Error" in result.stdout


def test_invalid_config_yaml(valid_pipeline_structure):
    """Test doctor detects invalid YAML in config files."""
    root = valid_pipeline_structure

    # Create invalid YAML (malformed)
    with open(root / "configs" / "yolo_config.yaml", "w") as f:
        f.write("invalid: yaml: syntax: [unclosed\n")

    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should exit with error
    assert result.returncode == 1

    # Should report config validation error
    assert "❌" in result.stdout
    assert ("Invalid YAML" in result.stdout or
            "yolo_config.yaml" in result.stdout)


def test_no_active_model_is_ok(valid_pipeline_structure):
    """Test doctor accepts missing active model (first run scenario)."""
    root = valid_pipeline_structure

    # Don't create any model in models/active/
    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should still be healthy - no model is OK for first run
    assert result.returncode == 0
    assert "✅ Pipeline is healthy" in result.stdout

    # Should mention no model found (but as info, not error)
    assert "No active model" in result.stdout or "first run" in result.stdout


def test_corrupted_model_file(valid_pipeline_structure):
    """Test doctor detects corrupted model files."""
    root = valid_pipeline_structure

    # Create corrupted model file (not a valid PyTorch model)
    model_path = root / "models" / "active" / "best.pt"
    with open(model_path, "w") as f:
        f.write("This is not a valid PyTorch model file\n")

    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should exit with error (model exists but can't load)
    assert result.returncode == 1

    # Should report model loading error
    assert "❌" in result.stdout
    assert ("failed to load" in result.stdout.lower() or
            "Model Validation" in result.stdout)


def test_empty_annotation_is_ok(valid_pipeline_structure):
    """Test doctor accepts empty annotation directories (no data yet)."""
    root = valid_pipeline_structure

    # Don't create any annotations - empty dirs are OK
    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should be healthy - empty dirs are fine (haven't started annotating)
    assert result.returncode == 0
    assert "✅ Pipeline is healthy" in result.stdout


def test_missing_image_for_label(valid_pipeline_structure):
    """Test doctor warns about labels without matching images."""
    root = valid_pipeline_structure

    # Create label without matching image
    verified_labels = root / "data" / "verified" / "labels"
    with open(verified_labels / "orphan_label.txt", "w") as f:
        f.write("0 0.5 0.5 0.3 0.4\n")

    # No corresponding orphan_label.jpg in images/

    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should exit 0 with warnings (not an error, just suspicious)
    assert result.returncode == 0

    # Should warn about missing images
    assert "⚠" in result.stdout
    assert "missing matching images" in result.stdout.lower() or "missing" in result.stdout.lower()


def test_class_id_out_of_range(valid_pipeline_structure):
    """Test doctor detects class IDs outside valid range."""
    root = valid_pipeline_structure

    # Config has 3 classes (0, 1, 2), create annotation with class 5
    verified_labels = root / "data" / "verified" / "labels"
    verified_images = root / "data" / "verified" / "images"

    (verified_images / "bad_class.jpg").touch()
    with open(verified_labels / "bad_class.txt", "w") as f:
        f.write("5 0.5 0.5 0.3 0.4\n")  # class_id=5 but only 0-2 valid

    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should exit with error
    assert result.returncode == 1

    # Should report class ID error
    assert "❌" in result.stdout
    assert ("out of range" in result.stdout or
            "Invalid format" in result.stdout or
            "class_id" in result.stdout)


def test_output_format(valid_pipeline_structure):
    """Test doctor output follows expected format."""
    result = subprocess.run(
        ["yolo-pipeline-doctor"],
        cwd=valid_pipeline_structure,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should have header
    assert "YOLO Pipeline Health Check" in result.stdout
    assert "=" in result.stdout  # Separator lines

    # Should have all sections
    assert "Structure Validation" in result.stdout
    assert "Configuration Validation" in result.stdout
    assert "Annotation Validation" in result.stdout
    assert "Model Validation" in result.stdout

    # Should have summary at end
    assert "All checks passed" in result.stdout or "Some checks" in result.stdout
