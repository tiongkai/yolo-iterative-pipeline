"""Process manager for running all pipeline services."""

import sys
import signal
import subprocess
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from pipeline.paths import PathManager
from pipeline.config import PipelineConfig
from pipeline.doctor import print_health_report
from pipeline.validation import PipelineValidator

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessManager:
    """Manages lifecycle of all pipeline services.

    Launches and coordinates:
    - auto_move_verified.py (working → verified file movement)
    - watcher.py (training trigger)
    - monitor.py (status display)

    Handles graceful shutdown on SIGINT/SIGTERM.
    """

    def __init__(self, paths: PathManager, config: PipelineConfig):
        """Initialize ProcessManager.

        Args:
            paths: PathManager instance
            config: PipelineConfig instance
        """
        self.paths = paths
        self.config = config
        self.processes: List[Tuple[str, subprocess.Popen]] = []
        self.running = False

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.stop_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def stop_all(self, timeout: float = 5.0) -> None:
        """Stop all running processes gracefully.

        Sends SIGTERM, waits for timeout, then sends SIGKILL if needed.

        Args:
            timeout: Seconds to wait for graceful shutdown
        """
        if not self.running:
            return

        logger.info("Stopping all processes...")
        self.running = False

        # Send SIGTERM to all processes
        for name, proc in self.processes:
            if proc.poll() is None:  # Still running
                logger.info(f"Terminating {name}...")
                proc.terminate()

        # Wait for graceful shutdown
        start_time = time.time()
        while time.time() - start_time < timeout:
            all_stopped = all(proc.poll() is not None for _, proc in self.processes)
            if all_stopped:
                break
            time.sleep(0.1)

        # Force kill any remaining processes
        for name, proc in self.processes:
            if proc.poll() is None:  # Still running
                logger.warning(f"Force killing {name}...")
                proc.kill()
                proc.wait()

        self.processes.clear()
        logger.info("All processes stopped")

    def run(
        self,
        debug: bool = False,
        no_doctor: bool = False,
        no_auto_move: bool = False
    ) -> None:
        """Run all pipeline services.

        Args:
            debug: Enable debug logging
            no_doctor: Skip doctor check
            no_auto_move: Skip auto-move service (for manual workflow)
        """
        try:
            # Step 1: Run doctor check (unless disabled)
            if not no_doctor:
                logger.info("Running health check...")
                validator = PipelineValidator(self.paths)
                report = validator.full_health_check()

                print_health_report(report)

                if not report.is_healthy():
                    logger.error("Health check failed. Fix issues before running pipeline.")
                    sys.exit(1)

                logger.info("Health check passed ✓")

            # Step 2: Register signal handlers
            self._register_signal_handlers()

            # Step 3: Launch subprocesses
            logger.info("Starting pipeline services...")
            self.running = True

            # Launch auto_move if enabled
            if not no_auto_move:
                auto_move_script = Path.cwd() / "scripts" / "auto_move_verified.py"
                auto_move_cmd = [
                    sys.executable,
                    str(auto_move_script)
                ]
                auto_move_proc = subprocess.Popen(
                    auto_move_cmd,
                    stdout=None,
                    stderr=None,
                    text=True
                )
                self.processes.append(("auto_move", auto_move_proc))
                logger.info("✓ Started auto-move service")

            # Launch watcher
            watcher_cmd = ["yolo-pipeline-watch"]
            watcher_proc = subprocess.Popen(
                watcher_cmd,
                stdout=None,
                stderr=None,
                text=True
            )
            self.processes.append(("watcher", watcher_proc))
            logger.info("✓ Started training watcher")

            # Launch monitor (stream its output)
            monitor_cmd = ["yolo-pipeline-monitor"]
            monitor_proc = subprocess.Popen(
                monitor_cmd,
                stdout=None,  # Stream to console
                stderr=None,
                text=True
            )
            self.processes.append(("monitor", monitor_proc))
            logger.info("✓ Started status monitor")

            logger.info("\n" + "="*60)
            logger.info("🚀 Pipeline is running!")
            logger.info("="*60)
            logger.info("Press Ctrl+C to stop all services")
            logger.info("="*60 + "\n")

            # Step 4: Monitor processes
            while self.running:
                # Check if any process died unexpectedly
                for name, proc in self.processes:
                    if proc.poll() is not None:
                        logger.error(f"Process {name} exited unexpectedly")
                        self.stop_all()
                        sys.exit(1)

                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\nReceived Ctrl+C, shutting down...")
            self.stop_all()

        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
            self.stop_all()
            raise


def main() -> None:
    """CLI entry point for yolo-pipeline-run command.

    Starts all pipeline services with a single command:
    - Health check (optional via --no-doctor)
    - Auto-move service (optional via --no-auto-move)
    - Training watcher
    - Status monitor

    Handles graceful shutdown on Ctrl+C.
    """
    parser = argparse.ArgumentParser(
        description="Run all YOLO pipeline services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start all services with health check
  yolo-pipeline-run

  # Skip health check (faster startup)
  yolo-pipeline-run --no-doctor

  # Manual workflow (no auto-move)
  yolo-pipeline-run --no-auto-move

  # Enable debug logging
  yolo-pipeline-run --debug
        """
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--no-doctor",
        action="store_true",
        help="Skip health check (faster startup, use with caution)"
    )

    parser.add_argument(
        "--no-auto-move",
        action="store_true",
        help="Skip auto-move service (for manual workflow)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config_path = Path.cwd() / "configs" / "pipeline_config.yaml"

        if not config_path.exists():
            logger.error("❌ Error: configs/pipeline_config.yaml not found")
            logger.error("   Run this command from your project root directory")
            sys.exit(1)

        config = PipelineConfig.from_yaml(config_path)

        # Create PathManager and ProcessManager
        paths = PathManager(Path.cwd(), config)
        process_manager = ProcessManager(paths, config)

        # Set debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        # Run all services
        process_manager.run(
            debug=args.debug,
            no_doctor=args.no_doctor,
            no_auto_move=args.no_auto_move
        )

    except KeyboardInterrupt:
        logger.info("\nShutdown complete")
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Error starting pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
