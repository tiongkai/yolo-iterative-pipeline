# setup.py
from setuptools import setup, find_packages
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


setup(
    name="yolo-iterative-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.3.0",
        "torch>=2.0.0",
        "watchdog>=3.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "yolo-pipeline-init=pipeline.cli:init_project",
            "yolo-pipeline-watch=pipeline.watcher:main",
            "yolo-pipeline-train=pipeline.train:main",
            "yolo-pipeline-monitor=pipeline.monitor:main",
            "yolo-pipeline-score=pipeline.active_learning:main",
            "yolo-pipeline-export=pipeline.export:main",
        ],
    },
    python_requires=">=3.8",
)
