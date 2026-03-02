"""Monitoring and status display for pipeline."""

from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

from pipeline.metrics import load_training_history

console = Console()

def display_status():
    """Display current pipeline status."""
    # Load training history
    history_path = Path("logs/training_history.json")
    history = load_training_history(history_path)

    # Count files
    verified_count = len(list(Path("data/verified").glob("*.txt")))
    working_count = len(list(Path("data/working").glob("*.txt")))
    test_count = len(list(Path("data/test/labels").glob("*.txt")))

    # Active model info
    active_model = Path("models/active/best.pt")
    model_info = "No active model"
    if active_model.exists() and history:
        latest = history[-1]
        model_info = f"{latest['version']} ({latest['timestamp'][:10]})"
        eval_map = latest.get('eval_mAP50', 0)
        eval_f1 = latest.get('eval_f1', 0)
        test_map = latest.get('test_mAP50', 0)
        test_f1 = latest.get('test_f1', 0)

        improvement = latest.get('improvement', {})
        eval_map_delta = improvement.get('eval_mAP50', 0)
        eval_f1_delta = improvement.get('eval_f1', 0)
    else:
        eval_map = eval_f1 = test_map = test_f1 = 0
        eval_map_delta = eval_f1_delta = 0

    # Pipeline status
    lock_file = Path("logs/.training.lock")
    status = "TRAINING" if lock_file.exists() else "HEALTHY"
    status_color = "yellow" if status == "TRAINING" else "green"

    # Build display
    console.print("\n[bold]YOLO Iterative Pipeline Status[/bold]\n", style="cyan")

    # Model info
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Active Model:", f"[bold]{model_info}[/bold]")
    table.add_row("Eval Metrics:",
                 f"mAP50: {eval_map:.3f} ({eval_map_delta:+.3f})  "
                 f"F1: {eval_f1:.3f} ({eval_f1_delta:+.3f})")
    table.add_row("Test Metrics:",
                 f"mAP50: {test_map:.3f}  F1: {test_f1:.3f}")
    console.print(table)

    console.print()

    # Data progress
    total_estimate = 1500  # TODO: make configurable
    progress_pct = (verified_count / total_estimate) * 100 if total_estimate else 0

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Data Progress:", "")
    table.add_row("  Verified:",
                 f"{verified_count} / {total_estimate} images  "
                 f"[{'█' * int(progress_pct / 10)}{'░' * (10 - int(progress_pct / 10))}] "
                 f"{progress_pct:.1f}%")
    table.add_row("  Working:", f"{working_count} images")
    table.add_row("  Test:", f"{test_count} images (fixed)")
    console.print(table)

    console.print()

    # Pipeline status
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Pipeline Status:", f"[{status_color}]{status}[/{status_color}]")

    # Check watcher log for last activity
    watcher_log = Path("logs/watcher.log")
    if watcher_log.exists():
        with open(watcher_log) as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1]
                if "Monitoring" in last_line:
                    table.add_row("  File Watcher:", "[green]Running ✓[/green]")
                else:
                    table.add_row("  File Watcher:", "[yellow]Unknown[/yellow]")

    if history:
        import datetime
        last_train_time = datetime.datetime.fromisoformat(history[-1]['timestamp'])
        time_since = datetime.datetime.now() - last_train_time
        hours = time_since.total_seconds() / 3600
        table.add_row("  Last Training:", f"{hours:.1f} hours ago")

    console.print(table)

    console.print()

    # Priority queue preview
    priority_file = Path("logs/priority_queue.txt")
    if priority_file.exists():
        console.print("[bold]Priority Queue Preview:[/bold]")
        with open(priority_file) as f:
            lines = [l for l in f.readlines() if not l.startswith("#")]
            for i, line in enumerate(lines[:5], 1):
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    filename = parts[0].strip()
                    priority = parts[1].strip()
                    console.print(f"  {i}. {filename} (score: {priority})")
        console.print()


def display_training_history():
    """Display training history as table."""
    history = load_training_history(Path("logs/training_history.json"))

    if not history:
        console.print("[yellow]No training history found[/yellow]")
        return

    table = Table(title="Training History")
    table.add_column("Version", style="cyan")
    table.add_column("Date", style="dim")
    table.add_column("Train Images", justify="right")
    table.add_column("Eval mAP50", justify="right")
    table.add_column("Eval F1", justify="right")
    table.add_column("Test mAP50", justify="right")
    table.add_column("Test F1", justify="right")
    table.add_column("Time (min)", justify="right")

    for entry in history[-10:]:  # Last 10 entries
        table.add_row(
            entry['version'],
            entry['timestamp'][:10],
            str(entry['train_images']),
            f"{entry.get('eval_mAP50', 0):.3f}",
            f"{entry.get('eval_f1', 0):.3f}",
            f"{entry.get('test_mAP50', 0):.3f}",
            f"{entry.get('test_f1', 0):.3f}",
            f"{entry['training_time_minutes']:.1f}"
        )

    console.print(table)


def main():
    """CLI entry point for monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor pipeline status")
    parser.add_argument("--history", action="store_true",
                       help="Show full training history")
    parser.add_argument("--health-check", action="store_true",
                       help="Run health check and exit")

    args = parser.parse_args()

    if args.history:
        display_training_history()
    else:
        display_status()

    if args.health_check:
        # Simple health check
        active_model = Path("models/active/best.pt")
        if not active_model.exists():
            console.print("[red]✗ No active model[/red]")
            exit(1)
        console.print("[green]✓ Pipeline healthy[/green]")


if __name__ == "__main__":
    main()
