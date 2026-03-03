# pipeline/watcher.py
import time
from pathlib import Path
from typing import Optional
import logging

from pipeline.config import PipelineConfig, YOLOConfig
from pipeline.train import train_model, promote_model
from pipeline.active_learning import score_all_images, save_priority_queue
from pipeline.paths import PathManager

# Ensure logs directory exists before setting up logging
Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/watcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def should_trigger_training(
    current_count: int,
    last_train_count: int,
    trigger_threshold: int,
    iteration: int,
    early_trigger: int = 25
) -> bool:
    """Determine if training should be triggered.

    Args:
        current_count: Current number of verified images
        last_train_count: Count at last training
        trigger_threshold: Default trigger threshold
        iteration: Current iteration number (0, 1, 2, ...)
        early_trigger: Lower threshold for first 3 iterations

    Returns:
        True if training should trigger
    """
    threshold = early_trigger if iteration < 3 else trigger_threshold
    new_images = current_count - last_train_count
    return new_images >= threshold


class FileWatcher:
    """Watch verified directory and trigger training."""

    def __init__(
        self,
        verified_dir: Path,
        trigger_threshold: int,
        pipeline_config_path: Optional[Path] = None,
        yolo_config_path: Optional[Path] = None,
    ):
        self.verified_dir = verified_dir
        self.trigger_threshold = trigger_threshold
        self.pipeline_config_path = pipeline_config_path or Path("configs/pipeline_config.yaml")
        self.yolo_config_path = yolo_config_path or Path("configs/yolo_config.yaml")

        self.last_train_count = 0
        self.iteration = 0
        self.is_training = False
        self.lock_file = Path("logs/.training.lock")

    def count_verified_images(self) -> int:
        """
        Count verified annotation files.

        Expects structure: verified/labels/*.txt
        """
        labels_dir = self.verified_dir / 'labels'
        if not labels_dir.exists():
            return 0
        return len(list(labels_dir.glob("*.txt")))

    def check_and_trigger(self):
        """Check if training should trigger and execute if so."""
        if self.is_training:
            logger.debug("Training already in progress, skipping check")
            return

        if self.lock_file.exists():
            logger.warning("Lock file exists, another training may be running")
            return

        current_count = self.count_verified_images()

        # Load configs
        pipeline_config = PipelineConfig.from_yaml(self.pipeline_config_path)

        if should_trigger_training(
            current_count=current_count,
            last_train_count=self.last_train_count,
            trigger_threshold=self.trigger_threshold,
            iteration=self.iteration,
            early_trigger=pipeline_config.early_trigger
        ):
            logger.info(
                f"TRIGGER: {current_count - self.last_train_count} new images "
                f"since last training, starting pipeline..."
            )
            self.trigger_training()
        else:
            logger.debug(
                f"No trigger: {current_count - self.last_train_count} new images "
                f"(need {self.trigger_threshold})"
            )

    def trigger_training(self):
        """Execute training pipeline."""
        self.is_training = True
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file.touch()

        try:
            # Load configs
            pipeline_config = PipelineConfig.from_yaml(self.pipeline_config_path)
            yolo_config = YOLOConfig.from_yaml(self.yolo_config_path)

            # Create PathManager
            paths = PathManager(Path.cwd(), pipeline_config)

            # Train
            logger.info("Training started...")
            version, checkpoint_dir = train_model(
                pipeline_config,
                yolo_config,
                paths,
                bootstrap=False,
                from_scratch=False
            )
            logger.info(f"Training completed: {version}")

            # Promote if improved
            promoted = promote_model(checkpoint_dir, Path("models/active"))

            if promoted:
                # Re-score priority queue
                logger.info("Re-scoring priority queue...")
                # TODO: These paths should be configurable via PipelineConfig
                # Currently hardcoded for consistency with train.py patterns
                scores = score_all_images(
                    working_dir=Path("data/working"),
                    sam3_dir=Path("data/sam3_annotations"),
                    model_path=Path("models/active/best.pt")
                )
                save_priority_queue(
                    scores,
                    Path("logs/priority_queue.txt"),
                    version
                )
                logger.info(f"Priority queue updated ({len(scores)} images)")

                # Notify user (desktop notification)
                if pipeline_config.desktop_notify:
                    try:
                        import subprocess
                        subprocess.run([
                            "notify-send",
                            "YOLO Pipeline",
                            f"Model {version} trained and promoted"
                        ], check=False)
                    except Exception:
                        pass  # Notification is optional

            # Update state
            self.last_train_count = self.count_verified_images()
            self.iteration += 1

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
        finally:
            self.is_training = False
            if self.lock_file.exists():
                self.lock_file.unlink()

    def run(self, check_interval: int = 60):
        """Run file watcher.

        Args:
            check_interval: Seconds between checks
        """
        logger.info(f"File watcher started, monitoring {self.verified_dir}")
        logger.info(f"Trigger threshold: {self.trigger_threshold} images")

        try:
            while True:
                self.check_and_trigger()
                time.sleep(check_interval)
        except KeyboardInterrupt:
            logger.info("File watcher stopped")


def main():
    """CLI entry point for file watcher."""
    import argparse

    parser = argparse.ArgumentParser(description="Watch verified directory and trigger training")
    parser.add_argument("--verified-dir", type=Path, default="data/verified")
    parser.add_argument("--config", type=Path, default="configs/pipeline_config.yaml")
    parser.add_argument("--yolo-config", type=Path, default="configs/yolo_config.yaml")
    parser.add_argument("--interval", type=int, default=60,
                       help="Check interval in seconds")

    args = parser.parse_args()

    # Load config for trigger threshold
    pipeline_config = PipelineConfig.from_yaml(args.config)

    watcher = FileWatcher(
        verified_dir=args.verified_dir,
        trigger_threshold=pipeline_config.trigger_threshold,
        pipeline_config_path=args.config,
        yolo_config_path=args.yolo_config
    )

    watcher.run(check_interval=args.interval)


if __name__ == "__main__":
    main()
