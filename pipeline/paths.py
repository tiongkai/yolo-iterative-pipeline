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

    # Manifest paths
    def splits_dir(self) -> Path:
        """Get splits directory path."""
        return self.root_dir / "data" / "splits"

    def train_manifest(self) -> Path:
        """Get train manifest file path."""
        return self.splits_dir() / "train.txt"

    def eval_manifest(self) -> Path:
        """Get eval manifest file path."""
        return self.splits_dir() / "eval.txt"

    # Model paths
    def active_model(self) -> Path:
        """Get active model file path."""
        return self.root_dir / "models" / "active" / "best.pt"

    def checkpoint_dir(self) -> Path:
        """Get checkpoints directory path."""
        return self.root_dir / "models" / "checkpoints"

    def deployed_dir(self) -> Path:
        """Get deployed models directory path."""
        return self.root_dir / "models" / "deployed"

    # Config paths
    def data_yaml(self) -> Path:
        """Get data.yaml file path."""
        return self.root_dir / "data" / "data.yaml"

    def pipeline_config(self) -> Path:
        """Get pipeline config file path."""
        return self.root_dir / "configs" / "pipeline_config.yaml"

    def yolo_config(self) -> Path:
        """Get YOLO config file path."""
        return self.root_dir / "configs" / "yolo_config.yaml"

    # Log paths
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        return self.root_dir / "logs"

    def training_history(self) -> Path:
        """Get training history file path."""
        return self.logs_dir() / "training_history.json"

    def watcher_log(self) -> Path:
        """Get watcher log file path."""
        return self.logs_dir() / "watcher.log"

    def auto_move_log(self) -> Path:
        """Get auto-move log file path."""
        return self.logs_dir() / "auto_move.log"

    def priority_queue(self) -> Path:
        """Get priority queue file path."""
        return self.logs_dir() / "priority_queue.txt"
