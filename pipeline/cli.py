# pipeline/cli.py
import sys
from pathlib import Path


def init_project(project_dir=None):
    """Initialize project directory structure.

    Args:
        project_dir: Optional root directory. Defaults to current directory.
    """
    root = Path(project_dir) if project_dir else Path.cwd()
    print(f"Initializing project in: {root.absolute()}\n")

    base_dirs = [
        "data/raw",
        "data/sam3_annotations",
        "data/working",
        "data/verified",
        "data/eval",
        "data/test/images",
        "data/test/labels",
        "models/checkpoints",
        "models/active",
        "models/deployed",
        "pipeline",
        "configs",
        "logs",
        "notebooks",
        "tests/unit",
        "tests/integration",
    ]

    try:
        for dir_path in base_dirs:
            full_path = root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created {full_path}")

        # Create .gitkeep files
        for dir_path in ["data/raw", "data/verified", "logs"]:
            gitkeep = root / dir_path / ".gitkeep"
            gitkeep.touch()

        print("\n✓ Project structure initialized")
    except PermissionError as e:
        print(f"✗ Permission denied: {e}")
        sys.exit(1)
    except OSError as e:
        print(f"✗ Failed to create directories: {e}")
        sys.exit(1)
