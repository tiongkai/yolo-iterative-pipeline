# pipeline/cli.py
from pathlib import Path

def init_project():
    """Initialize project directory structure."""
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

    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")

    # Create .gitkeep files
    for dir_path in ["data/raw", "data/verified", "logs"]:
        gitkeep = Path(dir_path) / ".gitkeep"
        gitkeep.touch()

    print("\n✓ Project structure initialized")
