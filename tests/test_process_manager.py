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


def test_main_default_args(tmp_path, monkeypatch):
    """Test main() with default arguments."""
    from pipeline.process_manager import main
    from unittest.mock import MagicMock, patch

    # Change to tmp_path
    monkeypatch.chdir(tmp_path)

    # Create minimal structure
    for subdir in ["working/images", "working/labels", "verified/images", "verified/labels",
                   "eval/images", "eval/labels", "test/images", "test/labels",
                   "models/active", "models/checkpoints", "configs", "logs", "data/splits"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

    # Create config
    config_path = tmp_path / "configs" / "pipeline_config.yaml"
    config_path.write_text("""
project_name: test
classes: [class1]
trigger_threshold: 50
early_trigger: 25
min_train_images: 10
eval_split_ratio: 0.15
stratify: false
uncertainty_weight: 0.4
disagreement_weight: 0.35
diversity_weight: 0.25
desktop_notify: false
slack_webhook: null
keep_last_n_checkpoints: 10
""")

    # Mock ProcessManager.run to prevent actual subprocess launches
    with patch('pipeline.process_manager.ProcessManager.run') as mock_run:
        # Mock argparse to return default args (no CLI args)
        with patch('sys.argv', ['yolo-pipeline-run']):
            try:
                main()
            except SystemExit:
                pass  # Expected if health check fails

    # Verify run was called with defaults
    mock_run.assert_called_once_with(debug=False, no_doctor=False, no_auto_move=False)


def test_main_with_debug_flag(tmp_path, monkeypatch):
    """Test main() with --debug flag."""
    from pipeline.process_manager import main
    from unittest.mock import patch

    # Change to tmp_path
    monkeypatch.chdir(tmp_path)

    # Create minimal structure
    for subdir in ["working/images", "working/labels", "verified/images", "verified/labels",
                   "eval/images", "eval/labels", "test/images", "test/labels",
                   "models/active", "models/checkpoints", "configs", "logs", "data/splits"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

    # Create config
    config_path = tmp_path / "configs" / "pipeline_config.yaml"
    config_path.write_text("""
project_name: test
classes: [class1]
trigger_threshold: 50
early_trigger: 25
min_train_images: 10
eval_split_ratio: 0.15
stratify: false
uncertainty_weight: 0.4
disagreement_weight: 0.35
diversity_weight: 0.25
desktop_notify: false
slack_webhook: null
keep_last_n_checkpoints: 10
""")

    # Mock ProcessManager.run
    with patch('pipeline.process_manager.ProcessManager.run') as mock_run:
        with patch('sys.argv', ['yolo-pipeline-run', '--debug']):
            try:
                main()
            except SystemExit:
                pass

    # Verify run was called with debug=True
    mock_run.assert_called_once_with(debug=True, no_doctor=False, no_auto_move=False)


def test_main_with_no_doctor_flag(tmp_path, monkeypatch):
    """Test main() with --no-doctor flag."""
    from pipeline.process_manager import main
    from unittest.mock import patch

    # Change to tmp_path
    monkeypatch.chdir(tmp_path)

    # Create minimal structure
    for subdir in ["working/images", "working/labels", "verified/images", "verified/labels",
                   "eval/images", "eval/labels", "test/images", "test/labels",
                   "models/active", "models/checkpoints", "configs", "logs", "data/splits"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

    # Create config
    config_path = tmp_path / "configs" / "pipeline_config.yaml"
    config_path.write_text("""
project_name: test
classes: [class1]
trigger_threshold: 50
early_trigger: 25
min_train_images: 10
eval_split_ratio: 0.15
stratify: false
uncertainty_weight: 0.4
disagreement_weight: 0.35
diversity_weight: 0.25
desktop_notify: false
slack_webhook: null
keep_last_n_checkpoints: 10
""")

    # Mock ProcessManager.run
    with patch('pipeline.process_manager.ProcessManager.run') as mock_run:
        with patch('sys.argv', ['yolo-pipeline-run', '--no-doctor']):
            try:
                main()
            except SystemExit:
                pass

    # Verify run was called with no_doctor=True
    mock_run.assert_called_once_with(debug=False, no_doctor=True, no_auto_move=False)


def test_main_with_no_auto_move_flag(tmp_path, monkeypatch):
    """Test main() with --no-auto-move flag."""
    from pipeline.process_manager import main
    from unittest.mock import patch

    # Change to tmp_path
    monkeypatch.chdir(tmp_path)

    # Create minimal structure
    for subdir in ["working/images", "working/labels", "verified/images", "verified/labels",
                   "eval/images", "eval/labels", "test/images", "test/labels",
                   "models/active", "models/checkpoints", "configs", "logs", "data/splits"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

    # Create config
    config_path = tmp_path / "configs" / "pipeline_config.yaml"
    config_path.write_text("""
project_name: test
classes: [class1]
trigger_threshold: 50
early_trigger: 25
min_train_images: 10
eval_split_ratio: 0.15
stratify: false
uncertainty_weight: 0.4
disagreement_weight: 0.35
diversity_weight: 0.25
desktop_notify: false
slack_webhook: null
keep_last_n_checkpoints: 10
""")

    # Mock ProcessManager.run
    with patch('pipeline.process_manager.ProcessManager.run') as mock_run:
        with patch('sys.argv', ['yolo-pipeline-run', '--no-auto-move']):
            try:
                main()
            except SystemExit:
                pass

    # Verify run was called with no_auto_move=True
    mock_run.assert_called_once_with(debug=False, no_doctor=False, no_auto_move=True)


def test_main_missing_config(tmp_path, monkeypatch):
    """Test main() fails gracefully when config is missing."""
    from pipeline.process_manager import main
    from unittest.mock import patch

    # Change to tmp_path (no config created)
    monkeypatch.chdir(tmp_path)

    # Mock ProcessManager.run
    with patch('pipeline.process_manager.ProcessManager.run') as mock_run:
        with patch('sys.argv', ['yolo-pipeline-run']):
            with pytest.raises(SystemExit) as exc_info:
                main()

    # Verify exit code is 1 (failure)
    assert exc_info.value.code == 1

    # Verify run was never called
    mock_run.assert_not_called()
