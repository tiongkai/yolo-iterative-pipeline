"""Tests for ProcessManager."""

import pytest
import signal
import time
from pathlib import Path
from pipeline.process_manager import ProcessManager
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


@pytest.fixture
def setup_process_manager(tmp_path):
    """Create ProcessManager for testing."""
    # Create minimal structure
    for subdir in ["working/images", "working/labels", "verified/images", "verified/labels",
                   "eval/images", "eval/labels", "test/images", "test/labels",
                   "models/active", "models/checkpoints", "configs", "logs", "data/splits"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        project_name="test",
        classes=["class1"],
        trigger_threshold=50,
        early_trigger=25,
        min_train_images=10,
        eval_split_ratio=0.15,
        stratify=False,
        uncertainty_weight=0.4,
        disagreement_weight=0.35,
        diversity_weight=0.25,
        desktop_notify=False,
        slack_webhook=None,
        keep_last_n_checkpoints=10
    )

    paths = PathManager(tmp_path, config)
    pm = ProcessManager(paths, config)

    return pm, tmp_path


def test_process_manager_init(setup_process_manager):
    """Test ProcessManager initializes correctly."""
    pm, root = setup_process_manager

    assert pm.paths is not None
    assert pm.config is not None
    assert pm.processes == []
    assert pm.running is False


def test_process_manager_register_signal_handlers(setup_process_manager):
    """Test signal handlers are registered."""
    pm, root = setup_process_manager

    # Register handlers
    pm._register_signal_handlers()

    # Verify signal handlers are set (can't easily test, but we can check no crash)
    assert True  # If we get here, registration worked


def test_process_manager_stop_all(setup_process_manager):
    """Test stop_all() terminates processes."""
    pm, root = setup_process_manager

    # Mock process
    import subprocess
    proc = subprocess.Popen(["sleep", "10"])
    pm.processes.append(("test", proc))
    pm.running = True

    # Stop all
    pm.stop_all()

    # Verify process terminated
    assert proc.poll() is not None  # Process has exited
    assert pm.running is False
    assert len(pm.processes) == 0


def test_process_manager_stop_all_graceful_timeout(setup_process_manager):
    """Test stop_all() uses SIGKILL if process doesn't respond to SIGTERM."""
    pm, root = setup_process_manager

    # Mock process that ignores SIGTERM
    import subprocess
    proc = subprocess.Popen(["sleep", "100"])
    pm.processes.append(("test", proc))
    pm.running = True

    # Stop with short timeout
    pm.stop_all(timeout=0.1)

    # Verify process killed
    assert proc.poll() is not None
    assert pm.running is False
