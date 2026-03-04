"""Integration tests for manifest-based training workflow.

Tests the complete flow: generate manifests from verified/ dataset,
create data.yaml, verify YOLO can read manifests, and files remain in verified/.
"""

import pytest
import yaml
from pathlib import Path
from pipeline.data_utils import generate_manifests
from pipeline.train import create_data_yaml
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


@pytest.fixture
def valid_pipeline_structure(tmp_path):
    """Create a valid pipeline directory structure."""
    # Create all required directories
    directories = [
        "data/working/images",
        "data/working/labels",
        "data/verified/images",
        "data/verified/labels",
        "data/eval/images",
        "data/eval/labels",
        "data/test/images",
        "data/test/labels",
        "data/sam3_annotations",
        "data/splits",
        "models/active",
        "models/checkpoints",
        "models/deployed",
        "configs",
        "logs",
    ]

    for directory in directories:
        (tmp_path / directory).mkdir(parents=True)

    return tmp_path


@pytest.fixture
def verified_dataset(valid_pipeline_structure):
    """Create sample verified dataset with images and labels."""
    root = valid_pipeline_structure
    verified_images = root / "data" / "verified" / "images"
    verified_labels = root / "data" / "verified" / "labels"

    # Create 100 sample files with multiple classes
    for i in range(100):
        # Create image (empty file is fine for testing)
        (verified_images / f"img_{i:03d}.jpg").touch()

        # Create label with class rotation (3 classes: 0, 1, 2)
        class_id = i % 3
        (verified_labels / f"img_{i:03d}.txt").write_text(
            f"{class_id} 0.5 0.5 0.2 0.3\n"
        )

    return root


@pytest.fixture
def pipeline_config():
    """Create test pipeline configuration."""
    return PipelineConfig(
        project_name="test_project",
        classes=["boat", "human", "motor"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=50,
        eval_split_ratio=0.15,
        stratify=True,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )


def test_generate_manifests_creates_files(verified_dataset, pipeline_config):
    """Test that generate_manifests creates train.txt and eval.txt files."""
    root = verified_dataset
    paths = PathManager(root, pipeline_config)

    # Generate manifests
    train_count, eval_count = generate_manifests(paths, pipeline_config)

    # Verify files created
    assert paths.train_manifest().exists(), "train.txt should be created"
    assert paths.eval_manifest().exists(), "eval.txt should be created"

    # Verify split ratio
    assert train_count + eval_count == 100, "Total count should be 100"
    # Allow 5% tolerance for split ratio (15 ± 5 images)
    assert abs(eval_count - 15) <= 5, f"Eval count {eval_count} should be ~15"


def test_generate_manifests_correct_split_ratio(verified_dataset, pipeline_config):
    """Test that manifests have correct split ratio."""
    root = verified_dataset
    paths = PathManager(root, pipeline_config)

    train_count, eval_count = generate_manifests(paths, pipeline_config)

    # Verify approximately 85/15 split
    total = train_count + eval_count
    eval_ratio = eval_count / total

    assert abs(eval_ratio - 0.15) < 0.05, \
        f"Eval ratio {eval_ratio:.2f} should be ~0.15 ± 0.05"


def test_manifests_contain_absolute_paths(verified_dataset, pipeline_config):
    """Test that manifests contain absolute paths to images."""
    root = verified_dataset
    paths = PathManager(root, pipeline_config)

    generate_manifests(paths, pipeline_config)

    # Read train manifest
    with open(paths.train_manifest()) as f:
        train_lines = [line.strip() for line in f if line.strip()]

    # Read eval manifest
    with open(paths.eval_manifest()) as f:
        eval_lines = [line.strip() for line in f if line.strip()]

    # Verify paths are absolute
    assert all(Path(line).is_absolute() for line in train_lines), \
        "Train manifest should contain absolute paths"
    assert all(Path(line).is_absolute() for line in eval_lines), \
        "Eval manifest should contain absolute paths"

    # Verify paths point to verified/images/
    verified_images = paths.verified_images()
    for line in train_lines + eval_lines:
        assert line.startswith(str(verified_images)), \
            f"Path {line} should start with {verified_images}"


def test_create_data_yaml_with_manifest_paths(verified_dataset, pipeline_config):
    """Test that create_data_yaml references manifest files correctly."""
    root = verified_dataset
    paths = PathManager(root, pipeline_config)

    # Generate manifests first
    generate_manifests(paths, pipeline_config)

    # Create data.yaml
    data_yaml_path = paths.data_yaml()
    create_data_yaml(
        train_manifest=paths.train_manifest(),
        eval_manifest=paths.eval_manifest(),
        test_dir=paths.test_dir(),
        classes=pipeline_config.classes,
        output_path=data_yaml_path,
        root_dir=root
    )

    # Verify data.yaml created
    assert data_yaml_path.exists(), "data.yaml should be created"

    # Load and verify contents
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)

    # Verify structure
    assert "train" in data, "data.yaml should have 'train' key"
    assert "val" in data, "data.yaml should have 'val' key"
    assert "test" in data, "data.yaml should have 'test' key"
    assert "names" in data, "data.yaml should have 'names' key"
    assert "path" in data, "data.yaml should have 'path' key"

    # Verify train/val reference manifest files (not directories)
    train_path = data["train"]
    val_path = data["val"]

    # Should be relative paths or absolute paths
    assert "train.txt" in train_path, \
        f"Train should reference train.txt, got {train_path}"
    assert "eval.txt" in val_path, \
        f"Val should reference eval.txt, got {val_path}"


def test_yolo_can_read_manifests(verified_dataset, pipeline_config):
    """Test that YOLO can read manifest files (format validation)."""
    root = verified_dataset
    paths = PathManager(root, pipeline_config)

    # Generate manifests
    generate_manifests(paths, pipeline_config)

    # Create data.yaml
    create_data_yaml(
        train_manifest=paths.train_manifest(),
        eval_manifest=paths.eval_manifest(),
        test_dir=paths.test_dir(),
        classes=pipeline_config.classes,
        output_path=paths.data_yaml(),
        root_dir=root
    )

    # Verify manifest files are readable
    with open(paths.train_manifest()) as f:
        train_lines = [line.strip() for line in f if line.strip()]

    with open(paths.eval_manifest()) as f:
        eval_lines = [line.strip() for line in f if line.strip()]

    # Verify all paths in manifests point to existing images
    for line in train_lines:
        image_path = Path(line)
        assert image_path.exists(), f"Image {image_path} should exist"
        assert image_path.suffix.lower() in ['.jpg', '.jpeg', '.png'], \
            f"Image {image_path} should have valid extension"

    for line in eval_lines:
        image_path = Path(line)
        assert image_path.exists(), f"Image {image_path} should exist"


def test_files_remain_in_verified_directory(verified_dataset, pipeline_config):
    """Test that files remain in verified/ after manifest generation."""
    root = verified_dataset
    paths = PathManager(root, pipeline_config)

    # Count files before manifest generation
    verified_images = paths.verified_images()
    verified_labels = paths.verified_labels()

    original_image_count = len(list(verified_images.glob("*.jpg")))
    original_label_count = len(list(verified_labels.glob("*.txt")))

    # Generate manifests
    generate_manifests(paths, pipeline_config)

    # Count files after manifest generation
    final_image_count = len(list(verified_images.glob("*.jpg")))
    final_label_count = len(list(verified_labels.glob("*.txt")))

    # Verify files remain in verified/
    assert final_image_count == original_image_count, \
        "Images should remain in verified/images/"
    assert final_label_count == original_label_count, \
        "Labels should remain in verified/labels/"

    # Verify no files moved to data/splits/
    splits_dir = paths.splits_dir()
    image_files_in_splits = list(splits_dir.glob("**/*.jpg"))
    label_files_in_splits = list(splits_dir.glob("**/*.txt"))

    # Only train.txt and eval.txt should be in splits/
    txt_files_in_splits = list(splits_dir.glob("*.txt"))
    assert len(txt_files_in_splits) == 2, \
        "Only train.txt and eval.txt should be in splits/"

    # No image or label files should be in splits/
    assert len(image_files_in_splits) == 0, \
        "No image files should be in splits/"
    assert len(label_files_in_splits) - 2 == 0, \
        "No label files (except manifests) should be in splits/"


def test_stratified_sampling_works(verified_dataset, pipeline_config):
    """Test that stratification ensures each class is represented."""
    root = verified_dataset

    # Enable stratification in config
    config_with_stratify = PipelineConfig(
        project_name="test_project",
        classes=["boat", "human", "motor"],
        trigger_threshold=50,
        min_train_images=50,
        eval_split_ratio=0.15,
        stratify=True,  # Enable stratification
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(root, config_with_stratify)

    # Generate manifests with stratification
    generate_manifests(paths, config_with_stratify, random_seed=42)

    # Read eval manifest
    with open(paths.eval_manifest()) as f:
        eval_lines = [line.strip() for line in f if line.strip()]

    # Get corresponding label files
    verified_labels = paths.verified_labels()
    eval_label_files = []
    for image_path in eval_lines:
        image_stem = Path(image_path).stem
        label_path = verified_labels / f"{image_stem}.txt"
        if label_path.exists():
            eval_label_files.append(label_path)

    # Count classes in eval set
    class_counts = {0: 0, 1: 0, 2: 0}
    for label_file in eval_label_files:
        with open(label_file) as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

    # Verify each class is represented (no class is 0)
    assert all(count > 0 for count in class_counts.values()), \
        f"All classes should be represented in eval set, got {class_counts}"


def test_regenerating_manifests_updates_files(verified_dataset, pipeline_config):
    """Test that re-generating manifests updates files with new data."""
    root = verified_dataset
    paths = PathManager(root, pipeline_config)

    # Generate manifests first time
    train_count_1, eval_count_1 = generate_manifests(paths, pipeline_config)

    # Read manifest contents
    with open(paths.train_manifest()) as f:
        train_lines_1 = set(line.strip() for line in f if line.strip())

    # Add more files to verified/
    verified_images = paths.verified_images()
    verified_labels = paths.verified_labels()

    for i in range(100, 120):  # Add 20 more files
        (verified_images / f"img_{i:03d}.jpg").touch()
        (verified_labels / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.3\n")

    # Re-generate manifests
    train_count_2, eval_count_2 = generate_manifests(paths, pipeline_config)

    # Read new manifest contents
    with open(paths.train_manifest()) as f:
        train_lines_2 = set(line.strip() for line in f if line.strip())

    # Verify new files included
    assert train_count_2 + eval_count_2 == 120, \
        "Total count should be 120 after adding 20 files"
    assert train_count_2 > train_count_1, \
        "Train count should increase after adding files"

    # Verify split ratio still correct
    eval_ratio = eval_count_2 / 120
    assert abs(eval_ratio - 0.15) < 0.05, \
        f"Eval ratio {eval_ratio:.2f} should still be ~0.15"


def test_manifests_handle_different_image_extensions(valid_pipeline_structure, pipeline_config):
    """Test that manifests work with different image extensions (.jpg, .png)."""
    root = valid_pipeline_structure
    verified_images = root / "data" / "verified" / "images"
    verified_labels = root / "data" / "verified" / "labels"

    # Create files with mixed extensions
    for i in range(50):
        ext = ".jpg" if i % 2 == 0 else ".png"
        (verified_images / f"img_{i:03d}{ext}").touch()
        (verified_labels / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.3\n")

    paths = PathManager(root, pipeline_config)

    # Generate manifests
    train_count, eval_count = generate_manifests(paths, pipeline_config)

    # Verify all files included
    assert train_count + eval_count == 50, \
        "Should handle both .jpg and .png files"

    # Read manifests and verify extensions
    with open(paths.train_manifest()) as f:
        train_lines = [line.strip() for line in f if line.strip()]

    with open(paths.eval_manifest()) as f:
        eval_lines = [line.strip() for line in f if line.strip()]

    all_lines = train_lines + eval_lines
    jpg_count = sum(1 for line in all_lines if line.endswith('.jpg'))
    png_count = sum(1 for line in all_lines if line.endswith('.png'))

    # Should have both types
    assert jpg_count > 0, "Should have .jpg files"
    assert png_count > 0, "Should have .png files"


def test_generate_manifests_fails_with_insufficient_images(valid_pipeline_structure, pipeline_config):
    """Test that generate_manifests raises error with too few images."""
    root = valid_pipeline_structure
    verified_images = root / "data" / "verified" / "images"
    verified_labels = root / "data" / "verified" / "labels"

    # Create only 10 images (less than min_train_images=50)
    for i in range(10):
        (verified_images / f"img_{i:03d}.jpg").touch()
        (verified_labels / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.3\n")

    paths = PathManager(root, pipeline_config)

    # Should raise ValueError
    with pytest.raises(ValueError, match="Need at least 50 labels"):
        generate_manifests(paths, pipeline_config)


def test_manifests_maintain_consistency_across_regenerations(verified_dataset, pipeline_config):
    """Test that manifests are deterministic with same random seed."""
    root = verified_dataset
    paths = PathManager(root, pipeline_config)

    # Generate manifests with seed 42
    generate_manifests(paths, pipeline_config, random_seed=42)

    with open(paths.train_manifest()) as f:
        train_lines_1 = [line.strip() for line in f if line.strip()]

    with open(paths.eval_manifest()) as f:
        eval_lines_1 = [line.strip() for line in f if line.strip()]

    # Re-generate with same seed
    generate_manifests(paths, pipeline_config, random_seed=42)

    with open(paths.train_manifest()) as f:
        train_lines_2 = [line.strip() for line in f if line.strip()]

    with open(paths.eval_manifest()) as f:
        eval_lines_2 = [line.strip() for line in f if line.strip()]

    # Verify same split
    assert train_lines_1 == train_lines_2, \
        "Train manifest should be deterministic with same seed"
    assert eval_lines_1 == eval_lines_2, \
        "Eval manifest should be deterministic with same seed"
