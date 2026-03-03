"""Doctor command for pipeline health checks."""

import sys
from pathlib import Path
from typing import Optional

from pipeline.config import PipelineConfig
from pipeline.paths import PathManager
from pipeline.validation import PipelineValidator, HealthReport


def print_health_report(report: HealthReport) -> None:
    """Print formatted health report to console.

    Args:
        report: HealthReport from full_health_check()
    """
    print("\n" + "="*60)
    print("🏥 YOLO Pipeline Health Check")
    print("="*60 + "\n")

    # Print overall status
    if report.overall_status == "healthy":
        print("✅ Pipeline is healthy - ready to run\n")
    elif report.overall_status == "warnings":
        print("⚠️  Pipeline has warnings - can run but needs attention\n")
    else:
        print("❌ Pipeline has errors - cannot run safely\n")

    # Print each check
    _print_check("Structure Validation", report.structure)
    _print_check("Configuration Validation", report.config)
    _print_check("Annotation Validation", report.annotations)
    _print_check("Model Validation", report.models)

    print("="*60)

    # Print summary
    if report.overall_status == "healthy":
        print("✓ All checks passed")
    elif report.overall_status == "warnings":
        print("⚠ Some checks have warnings")
    else:
        print("✗ Some checks failed")
    print("="*60 + "\n")


def _print_check(name: str, result) -> None:
    """Print individual validation check result.

    Args:
        name: Check name
        result: ValidationResult
    """
    # Status symbol
    if result.status == "pass":
        symbol = "✅"
    elif result.status == "warning":
        symbol = "⚠️ "
    else:
        symbol = "❌"

    print(f"{symbol} {name}")

    # Print messages (limit to first 5 for readability)
    for i, msg in enumerate(result.messages[:5]):
        print(f"   {msg}")

    if len(result.messages) > 5:
        print(f"   ... and {len(result.messages) - 5} more messages")

    print()


def main() -> None:
    """Run doctor command - comprehensive pipeline health check."""
    try:
        # Load configuration
        config_path = Path.cwd() / "configs" / "pipeline_config.yaml"

        if not config_path.exists():
            print("❌ Error: configs/pipeline_config.yaml not found")
            print("   Run this command from your project root directory")
            sys.exit(1)

        config = PipelineConfig.from_yaml(config_path)

        # Create PathManager and validator
        paths = PathManager(Path.cwd(), config)
        validator = PipelineValidator(paths)

        # Run full health check
        report = validator.full_health_check()

        # Print report
        print_health_report(report)

        # Exit with appropriate code
        sys.exit(0 if report.is_healthy() else 1)

    except Exception as e:
        print(f"❌ Error running health check: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
