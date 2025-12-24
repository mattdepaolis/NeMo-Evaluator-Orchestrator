#!/usr/bin/env python3
"""
Display system status and overview of the benchmark evaluation system.
"""
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemo_evaluator_orchestrator.utils.benchmark_utils import load_benchmark_catalog
from nemo_evaluator_orchestrator.utils.paths import get_config_dir

console = Console()


def main():
    """Display system overview."""
    config_dir = get_config_dir()

    console.print(
        Panel.fit(
            "[bold cyan]NeMo Evaluator - Comprehensive Benchmark System[/bold cyan]"
        )
    )

    # Check catalog
    catalog_path = config_dir / "benchmark_catalog.yaml"
    if catalog_path.exists():
        catalog = load_benchmark_catalog(catalog_path)
        console.print(
            f"\n[green]✓ Benchmark Catalog: {catalog['total_benchmarks']} benchmarks across {len(catalog['harnesses'])} harnesses[/green]"
        )
    else:
        console.print(
            "\n[yellow]⚠ Benchmark catalog not found. Run 'python generate_catalog.py'[/yellow]"
        )

    # Check config files
    config_files = {
        "Model Config": config_dir / "model_config.yaml",
        "Eval Params": config_dir / "eval_params.yaml",
        "Selected Benchmarks": config_dir / "selected_benchmarks.yaml",
        "Benchmark Suites": config_dir / "benchmark_suites.yaml",
    }

    table = Table(title="Configuration Files")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Path", style="yellow")

    for name, path in config_files.items():
        exists = "✓" if path.exists() else "✗"
        status_style = "green" if path.exists() else "red"
        table.add_row(
            name, f"[{status_style}]{exists}[/{status_style}]", str(path.name)
        )

    console.print("\n")
    console.print(table)

    # Available tools
    tools = {
        "generate_catalog.py": "Generate benchmark catalog",
        "select_benchmarks.py": "Interactive benchmark selector",
        "manage_containers.py": "Container management",
        "run_all_benchmarks.py": "Execute evaluations",
        "monitor_benchmarks.py": "Real-time progress monitoring",
        "aggregate_results.py": "Aggregate and analyze results",
    }

    table2 = Table(title="Available Tools")
    table2.add_column("Tool", style="cyan")
    table2.add_column("Description", style="green")
    table2.add_column("Status", style="yellow")

    # Tools are now in src/nemo_evaluator_orchestrator/cli/
    from nemo_evaluator_orchestrator.utils.paths import get_project_root

    project_root = get_project_root()
    cli_dir = project_root / "src" / "nemo_evaluator_orchestrator" / "cli"

    for tool, desc in tools.items():
        tool_path = cli_dir / tool
        exists = "✓ Available" if tool_path.exists() else "✗ Missing"
        status_style = "green" if tool_path.exists() else "red"
        table2.add_row(tool, desc, f"[{status_style}]{exists}[/{status_style}]")

    console.print("\n")
    console.print(table2)

    # Quick start guide
    console.print()
    console.print(
        Panel.fit(
            "[bold]Quick Start:[/bold]\n"
            "1. Configure model: [cyan]vim config/model_config.yaml[/cyan]\n"
            "2. Select benchmarks: [cyan]nemo-select[/cyan]\n"
            "3. Verify containers: [cyan]nemo-containers[/cyan]\n"
            "4. Run evaluations: [cyan]nemo-run[/cyan]\n"
            "   (Optional: [cyan]nemo-monitor[/cyan] in another terminal)\n"
            "5. View results: [cyan]http://127.0.0.1:5000[/cyan] (MLflow UI)\n\n"
            "See [cyan]docs/README.md[/cyan] for detailed documentation."
        )
    )


if __name__ == "__main__":
    main()
