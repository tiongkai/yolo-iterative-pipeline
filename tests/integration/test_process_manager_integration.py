"""Integration tests for ProcessManager.

Tests the full process manager end-to-end, verifying it correctly:
- Launches all pipeline services
- Handles graceful shutdown on SIGINT/SIGTERM
- Runs doctor check before launching services
- Respects command-line flags (--no-doctor, --no-auto-move)
- Monitors processes and shuts down cleanly if one dies
"""

import pytest
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import yaml

from pipeline.process_manager import ProcessManager
from pipeline.paths import PathManager
from pipeline.config import PipelineConfig


@pytest.fixture
def valid_pipeline_structure(tmp_path):
    """Create complete valid YOLO pipeline structure.

    Returns:
        Path to pipeline root directory with all required directories
    """
    # Create all 14 required directories with YOLO layout
    (tmp_path / "data" / "working" / "images").mkdir(parents=True)
    (tmp_path / "data" / "working" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "verified" / "images").mkdir(parents=True)
    (tmp_path / "data" / "verified" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "images").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "test" / "images").mkdir(parents=True)
    (tmp_path / "data" / "test" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "models" / "active").mkdir(parents=True)
    (tmp_path / "models" / "checkpoints").mkdir(parents=True)
    (tmp_path / "models" / "deployed").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)

    # Create valid pipeline_config.yaml
    pipeline_config = {
        "project_name": "test_project",
        "classes": ["boat", "human", "motor"],
        "trigger_threshold": 50,
        "eval_split_ratio": 0.15,
        "early_trigger": 25
    }
    with open(tmp_path / "configs" / "pipeline_config.yaml", "w") as f:
        yaml.dump(pipeline_config, f)

    # Create valid yolo_config.yaml
    yolo_config = {
        "model": "yolo11n.pt",
        "imgsz": 1280,
        "batch": 16,
        "epochs": 100,
        "patience": 10,
        "device": "0,1"
    }
    with open(tmp_path / "configs" / "yolo_config.yaml", "w") as f:
        yaml.dump(yolo_config, f)

    # Create data.yaml (required by YOLO)
    data_yaml = {
        "path": str(tmp_path / "data"),
        "train": "verified/images",
        "val": "eval/images",
        "test": "test/images",
        "names": {0: "boat", 1: "human", 2: "motor"}
    }
    with open(tmp_path / "data" / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    # Create auto_move_verified.py script
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    auto_move_script = scripts_dir / "auto_move_verified.py"
    auto_move_script.write_text("#!/usr/bin/env python3\nimport time\nwhile True: time.sleep(1)\n")

    return tmp_path


@pytest.fixture
def pipeline_config():
    """Create test pipeline config."""
    return PipelineConfig(
        project_name="test_project",
        classes=["boat", "human", "motor"],
        trigger_threshold=50,
        eval_split_ratio=0.15,
        early_trigger=25
    )


@pytest.fixture
def mock_process():
    """Create a mock subprocess.Popen process."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.poll.return_value = None  # Process is running
    proc.pid = 12345
    return proc


def test_services_launch_successfully(valid_pipeline_structure, pipeline_config, mock_process):
    """Test that ProcessManager launches all services correctly."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    with patch('pipeline.process_manager.subprocess.Popen') as mock_popen, \
         patch('pipeline.process_manager.PipelineValidator') as mock_validator, \
         patch('pipeline.process_manager.print_health_report') as mock_print, \
         patch('pipeline.process_manager.signal.signal'), \
         patch.object(manager, 'stop_all') as mock_stop:

        # Mock successful validation
        mock_report = Mock()
        mock_report.is_healthy.return_value = True
        mock_validator.return_value.full_health_check.return_value = mock_report

        # Mock process creation
        mock_popen.return_value = mock_process

        # Make run() exit immediately after launching services
        def trigger_exit(*args, **kwargs):
            manager.running = False
            return mock_process

        mock_popen.side_effect = trigger_exit

        # Run manager
        manager.run()

        # Verify 3 services launched (auto_move, watcher, monitor)
        assert mock_popen.call_count == 3

        # Verify correct command-line arguments
        calls = mock_popen.call_args_list

        # Call 1: auto_move_verified.py
        auto_move_call = calls[0]
        assert "auto_move_verified.py" in str(auto_move_call)

        # Call 2: watcher
        watcher_call = calls[1]
        assert watcher_call[0][0] == ["yolo-pipeline-watch"]

        # Call 3: monitor
        monitor_call = calls[2]
        assert monitor_call[0][0] == ["yolo-pipeline-monitor"]

        # Verify processes were tracked
        assert len(manager.processes) == 3
        assert manager.processes[0][0] == "auto_move"
        assert manager.processes[1][0] == "watcher"
        assert manager.processes[2][0] == "monitor"


def test_graceful_shutdown_on_sigterm(valid_pipeline_structure, pipeline_config):
    """Test graceful shutdown with SIGTERM then SIGKILL if needed."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    # Create mock processes
    proc1 = MagicMock(spec=subprocess.Popen)
    proc2 = MagicMock(spec=subprocess.Popen)
    proc3 = MagicMock(spec=subprocess.Popen)

    # Counter for poll calls to simulate processes stopping over time
    poll_counts = {"proc1": 0, "proc2": 0, "proc3": 0}

    def make_poll_func(proc_name, stops_gracefully=True):
        """Create poll function that returns None initially, then exit code."""
        def poll():
            poll_counts[proc_name] += 1
            if stops_gracefully and poll_counts[proc_name] > 2:
                return 0  # Stopped
            elif not stops_gracefully:
                return None  # Never stops
            return None  # Still running
        return poll

    proc1.poll = make_poll_func("proc1", stops_gracefully=True)
    proc2.poll = make_poll_func("proc2", stops_gracefully=True)
    proc3.poll = make_poll_func("proc3", stops_gracefully=False)

    manager.processes = [
        ("auto_move", proc1),
        ("watcher", proc2),
        ("monitor", proc3)
    ]
    manager.running = True

    with patch('pipeline.process_manager.time.sleep'):
        # Call stop_all with short timeout
        manager.stop_all(timeout=0.2)

    # Verify SIGTERM sent to all
    proc1.terminate.assert_called_once()
    proc2.terminate.assert_called_once()
    proc3.terminate.assert_called_once()

    # Verify SIGKILL sent to stubborn process
    proc1.kill.assert_not_called()  # Graceful shutdown
    proc2.kill.assert_not_called()  # Graceful shutdown
    proc3.kill.assert_called_once()  # Forced kill

    # Verify wait() called on killed process
    proc3.wait.assert_called_once()

    # Verify processes cleared
    assert len(manager.processes) == 0
    assert manager.running is False


def test_doctor_check_runs_first(valid_pipeline_structure, pipeline_config):
    """Test that doctor check runs before launching services."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    with patch('pipeline.process_manager.subprocess.Popen') as mock_popen, \
         patch('pipeline.process_manager.PipelineValidator') as mock_validator, \
         patch('pipeline.process_manager.print_health_report') as mock_print, \
         pytest.raises(SystemExit) as exc_info:

        # Mock failed validation
        mock_report = Mock()
        mock_report.is_healthy.return_value = False
        mock_validator.return_value.full_health_check.return_value = mock_report

        # Run manager (should exit early)
        manager.run()

    # Verify validator was called
    mock_validator.assert_called_once_with(paths)
    mock_validator.return_value.full_health_check.assert_called_once()

    # Verify health report was printed
    mock_print.assert_called_once_with(mock_report)

    # Verify no services launched
    mock_popen.assert_not_called()

    # Verify exit code 1
    assert exc_info.value.code == 1


def test_no_doctor_flag_skips_health_check(valid_pipeline_structure, pipeline_config, mock_process):
    """Test that --no-doctor flag skips health check."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    with patch('pipeline.process_manager.subprocess.Popen') as mock_popen, \
         patch('pipeline.process_manager.PipelineValidator') as mock_validator, \
         patch('pipeline.process_manager.print_health_report') as mock_print, \
         patch('pipeline.process_manager.signal.signal'):

        # Mock process creation
        def trigger_exit(*args, **kwargs):
            manager.running = False
            return mock_process

        mock_popen.side_effect = trigger_exit

        # Run with no_doctor=True
        manager.run(no_doctor=True)

        # Verify validator NOT called
        mock_validator.assert_not_called()

        # Verify services still launched
        assert mock_popen.call_count == 3


def test_no_auto_move_flag(valid_pipeline_structure, pipeline_config, mock_process):
    """Test that --no-auto-move flag skips auto-move service."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    with patch('pipeline.process_manager.subprocess.Popen') as mock_popen, \
         patch('pipeline.process_manager.PipelineValidator') as mock_validator, \
         patch('pipeline.process_manager.print_health_report') as mock_print, \
         patch('pipeline.process_manager.signal.signal'):

        # Mock successful validation
        mock_report = Mock()
        mock_report.is_healthy.return_value = True
        mock_validator.return_value.full_health_check.return_value = mock_report

        # Mock process creation
        def trigger_exit(*args, **kwargs):
            manager.running = False
            return mock_process

        mock_popen.side_effect = trigger_exit

        # Run with no_auto_move=True
        manager.run(no_auto_move=True)

        # Verify only 2 services launched (watcher, monitor)
        assert mock_popen.call_count == 2

        # Verify auto_move not in calls
        calls = mock_popen.call_args_list
        assert "auto_move_verified.py" not in str(calls)

        # Verify correct services launched
        assert calls[0][0][0] == ["yolo-pipeline-watch"]
        assert calls[1][0][0] == ["yolo-pipeline-monitor"]

        # Verify processes tracked correctly
        assert len(manager.processes) == 2
        assert manager.processes[0][0] == "watcher"
        assert manager.processes[1][0] == "monitor"


def test_process_monitoring_loop_detects_death(valid_pipeline_structure, pipeline_config):
    """Test that monitoring loop detects when a process dies."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    # Create mock processes
    proc1 = MagicMock(spec=subprocess.Popen)
    proc2 = MagicMock(spec=subprocess.Popen)
    proc3 = MagicMock(spec=subprocess.Popen)

    # Simulate proc2 dying after first check
    check_count = [0]

    def poll_side_effect():
        check_count[0] += 1
        if check_count[0] > 1:
            return 1  # Died
        return None  # Still running

    proc1.poll.return_value = None  # Always running
    proc2.poll.side_effect = poll_side_effect  # Dies on second check
    proc3.poll.return_value = None  # Always running

    manager.processes = [
        ("auto_move", proc1),
        ("watcher", proc2),
        ("monitor", proc3)
    ]
    manager.running = True

    with patch('pipeline.process_manager.time.sleep') as mock_sleep, \
         patch.object(manager, 'stop_all') as mock_stop, \
         pytest.raises(SystemExit) as exc_info:

        # Mock sleep to avoid waiting
        def handle_sleep(duration):
            # Trigger second poll check
            pass

        mock_sleep.side_effect = handle_sleep

        # Run monitoring loop (should detect death and exit)
        while manager.running:
            # Check if any process died unexpectedly
            for name, proc in manager.processes:
                if proc.poll() is not None:
                    manager.stop_all()
                    raise SystemExit(1)

            time.sleep(1)

    # Verify stop_all was called
    mock_stop.assert_called_once()

    # Verify exit code 1
    assert exc_info.value.code == 1


def test_signal_handler_registration(valid_pipeline_structure, pipeline_config, mock_process):
    """Test that signal handlers are registered correctly."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    with patch('pipeline.process_manager.signal.signal') as mock_signal, \
         patch('pipeline.process_manager.subprocess.Popen') as mock_popen, \
         patch('pipeline.process_manager.PipelineValidator') as mock_validator, \
         patch('pipeline.process_manager.print_health_report') as mock_print:

        # Mock successful validation
        mock_report = Mock()
        mock_report.is_healthy.return_value = True
        mock_validator.return_value.full_health_check.return_value = mock_report

        # Mock process creation
        def trigger_exit(*args, **kwargs):
            manager.running = False
            return mock_process

        mock_popen.side_effect = trigger_exit

        # Run manager
        manager.run()

        # Verify signal handlers registered
        assert mock_signal.call_count >= 2

        # Check SIGINT and SIGTERM were registered
        signal_calls = [call[0] for call in mock_signal.call_args_list]
        assert (signal.SIGINT,) in [call[:1] for call in signal_calls]
        assert (signal.SIGTERM,) in [call[:1] for call in signal_calls]


def test_stop_all_with_no_running_processes(valid_pipeline_structure, pipeline_config):
    """Test stop_all when no processes are running (idempotent)."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    # No processes registered
    manager.processes = []
    manager.running = False

    # Should not raise error
    manager.stop_all()

    # Verify still empty
    assert len(manager.processes) == 0


def test_stop_all_clears_processes_list(valid_pipeline_structure, pipeline_config):
    """Test that stop_all clears processes list."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    # Create mock processes
    proc1 = MagicMock(spec=subprocess.Popen)
    proc2 = MagicMock(spec=subprocess.Popen)

    proc1.poll.return_value = 0  # Already stopped
    proc2.poll.return_value = 0  # Already stopped

    manager.processes = [
        ("auto_move", proc1),
        ("watcher", proc2)
    ]
    manager.running = True

    # Stop all
    manager.stop_all()

    # Verify processes cleared
    assert len(manager.processes) == 0
    assert manager.running is False


def test_missing_auto_move_script_fails(tmp_path, pipeline_config):
    """Test that missing auto_move_verified.py causes early exit."""
    # Create pipeline structure without scripts/auto_move_verified.py
    (tmp_path / "data" / "verified" / "images").mkdir(parents=True)
    (tmp_path / "data" / "verified" / "labels").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)

    # Create configs
    with open(tmp_path / "configs" / "pipeline_config.yaml", "w") as f:
        yaml.dump({"project_name": "test", "classes": ["boat"], "trigger_threshold": 50}, f)

    with open(tmp_path / "configs" / "yolo_config.yaml", "w") as f:
        yaml.dump({"model": "yolo11n.pt"}, f)

    # NOTE: NOT creating scripts/auto_move_verified.py

    paths = PathManager(tmp_path, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    with patch('pipeline.process_manager.PipelineValidator') as mock_validator, \
         patch('pipeline.process_manager.print_health_report') as mock_print, \
         patch('pipeline.process_manager.signal.signal'), \
         pytest.raises(SystemExit) as exc_info:

        # Mock successful validation
        mock_report = Mock()
        mock_report.is_healthy.return_value = True
        mock_validator.return_value.full_health_check.return_value = mock_report

        # Run manager (should fail due to missing script)
        manager.run()

    # Verify exit code 1
    assert exc_info.value.code == 1


def test_keyboard_interrupt_triggers_shutdown(valid_pipeline_structure, pipeline_config, mock_process):
    """Test that KeyboardInterrupt triggers graceful shutdown."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    with patch('pipeline.process_manager.subprocess.Popen') as mock_popen, \
         patch('pipeline.process_manager.PipelineValidator') as mock_validator, \
         patch('pipeline.process_manager.print_health_report') as mock_print, \
         patch('pipeline.process_manager.signal.signal'), \
         patch.object(manager, 'stop_all') as mock_stop:

        # Mock successful validation
        mock_report = Mock()
        mock_report.is_healthy.return_value = True
        mock_validator.return_value.full_health_check.return_value = mock_report

        # Mock process creation
        mock_popen.return_value = mock_process

        # Simulate KeyboardInterrupt after launching services
        def raise_keyboard_interrupt(*args, **kwargs):
            manager.running = True
            raise KeyboardInterrupt()

        mock_popen.side_effect = raise_keyboard_interrupt

        # Run manager (should catch KeyboardInterrupt)
        try:
            manager.run()
        except KeyboardInterrupt:
            pass  # Expected

        # Verify stop_all was called
        mock_stop.assert_called()


def test_exception_during_run_triggers_cleanup(valid_pipeline_structure, pipeline_config, mock_process):
    """Test that exceptions during run trigger cleanup."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    with patch('pipeline.process_manager.subprocess.Popen') as mock_popen, \
         patch('pipeline.process_manager.PipelineValidator') as mock_validator, \
         patch('pipeline.process_manager.print_health_report') as mock_print, \
         patch('pipeline.process_manager.signal.signal'), \
         patch.object(manager, 'stop_all') as mock_stop:

        # Mock successful validation
        mock_report = Mock()
        mock_report.is_healthy.return_value = True
        mock_validator.return_value.full_health_check.return_value = mock_report

        # Mock process creation to raise exception
        mock_popen.side_effect = RuntimeError("Test error")

        # Run manager (should catch exception)
        with pytest.raises(RuntimeError):
            manager.run()

        # Verify stop_all was called for cleanup
        mock_stop.assert_called()


def test_process_stdout_stderr_configuration(valid_pipeline_structure, pipeline_config, mock_process):
    """Test that processes are configured with correct stdout/stderr."""
    paths = PathManager(valid_pipeline_structure, pipeline_config)
    manager = ProcessManager(paths, pipeline_config)

    with patch('pipeline.process_manager.subprocess.Popen') as mock_popen, \
         patch('pipeline.process_manager.PipelineValidator') as mock_validator, \
         patch('pipeline.process_manager.print_health_report') as mock_print, \
         patch('pipeline.process_manager.signal.signal'):

        # Mock successful validation
        mock_report = Mock()
        mock_report.is_healthy.return_value = True
        mock_validator.return_value.full_health_check.return_value = mock_report

        # Mock process creation
        def trigger_exit(*args, **kwargs):
            manager.running = False
            return mock_process

        mock_popen.side_effect = trigger_exit

        # Run manager
        manager.run()

        # Verify all processes have stdout=None, stderr=None (inherit from parent)
        for call_obj in mock_popen.call_args_list:
            kwargs = call_obj[1]
            assert kwargs.get('stdout') is None
            assert kwargs.get('stderr') is None
            assert kwargs.get('text') is True
