# pipeline/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    project_name: str
    classes: List[str]
    trigger_threshold: int = 50
    early_trigger: int = 25
    min_train_images: int = 50
    eval_split_ratio: float = 0.15
    stratify: bool = True

    # Active learning weights
    uncertainty_weight: float = 0.40
    disagreement_weight: float = 0.35
    diversity_weight: float = 0.25

    # Notifications
    desktop_notify: bool = True
    slack_webhook: Optional[str] = None

    # Cleanup
    keep_last_n_checkpoints: int = 10

    def __post_init__(self):
        """Validate configuration."""
        if self.trigger_threshold <= 0:
            raise ValueError("trigger_threshold must be positive")
        if not 0 < self.eval_split_ratio < 1:
            raise ValueError("eval_split_ratio must be between 0 and 1")
        if len(self.classes) == 0:
            raise ValueError("classes list cannot be empty")

        # Normalize weights
        total_weight = (
            self.uncertainty_weight +
            self.disagreement_weight +
            self.diversity_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError("Active learning weights must sum to 1.0")

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


@dataclass
class YOLOConfig:
    """YOLO training configuration."""
    model: str = "yolo26n.pt"
    epochs: int = 50
    batch_size: int = 16
    imgsz: int = 1280
    device: List[int] = field(default_factory=lambda: [0, 1])
    patience: int = 10

    # Augmentation
    close_mosaic: int = 10
    copy_paste: float = 0.5
    mixup: float = 0.1
    scale: float = 0.9
    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    mosaic: float = 1.0

    @classmethod
    def from_yaml(cls, path: str) -> "YOLOConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
