"""Centralized path management for YOLO pipeline with YOLO structure enforcement."""

from pathlib import Path
from typing import Optional
from .config import PipelineConfig


class PathManager:
    """Manages all file paths for the YOLO pipeline.

    Enforces YOLO directory structure (images/ and labels/ subdirectories)
    for all data directories. Provides single source of truth for paths.
    """

    def __init__(self, root_dir: Path, config: PipelineConfig):
        """Initialize PathManager.

        Args:
            root_dir: Root directory for the project
            config: Pipeline configuration
        """
        self.root_dir = Path(root_dir)
        self.config = config
        self.data_dir = self.root_dir / "data"

    # Working directory methods
    def working_dir(self) -> Path:
        """Get working directory path."""
        return self.data_dir / "working"

    def working_images(self) -> Path:
        """Get working images directory path."""
        return self.working_dir() / "images"

    def working_labels(self) -> Path:
        """Get working labels directory path."""
        return self.working_dir() / "labels"

    # Verified directory methods
    def verified_dir(self) -> Path:
        """Get verified directory path."""
        return self.data_dir / "verified"

    def verified_images(self) -> Path:
        """Get verified images directory path."""
        return self.verified_dir() / "images"

    def verified_labels(self) -> Path:
        """Get verified labels directory path."""
        return self.verified_dir() / "labels"

    # Eval directory methods
    def eval_dir(self) -> Path:
        """Get eval directory path."""
        return self.data_dir / "eval"

    def eval_images(self) -> Path:
        """Get eval images directory path."""
        return self.eval_dir() / "images"

    def eval_labels(self) -> Path:
        """Get eval labels directory path."""
        return self.eval_dir() / "labels"

    # Test directory methods
    def test_dir(self) -> Path:
        """Get test directory path."""
        return self.data_dir / "test"

    def test_images(self) -> Path:
        """Get test images directory path."""
        return self.test_dir() / "images"

    def test_labels(self) -> Path:
        """Get test labels directory path."""
        return self.test_dir() / "labels"
