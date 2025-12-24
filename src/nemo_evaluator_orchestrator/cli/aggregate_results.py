#!/usr/bin/env python3
"""
Results aggregation and analysis tool.
Aggregates results from multiple benchmark runs and generates reports.
"""
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import mlflow

from nemo_evaluator_orchestrator.utils.paths import get_project_root, get_output_dir

console = Console()


def load_execution_summary(summary_path: Path) -> Dict:
    """Load execution summary from YAML file."""
    with open(summary_path, "r") as f:
        return yaml.safe_load(f)


def aggregate_metrics_from_outputs(output_dir: Path) -> Dict[str, Dict]:
    """Aggregate metrics from evaluation output directories."""
    metrics = {}

    # Find all evaluation result directories
    for eval_dir in output_dir.glob("eval_*"):
        for benchmark_dir in eval_dir.iterdir():
            if benchmark_dir.is_dir():
                artifacts_dir = benchmark_dir / "artifacts"
                if artifacts_dir.exists():
                    metrics_file = artifacts_dir / "eval_factory_metrics.json"
                    if metrics_file.exists():
                        with open(metrics_file, "r") as f:
                            benchmark_metrics = json.load(f)
                            benchmark_name = benchmark_dir.name
                            metrics[benchmark_name] = benchmark_metrics

    return metrics


def generate_summary_report(summary: Dict, metrics: Dict[str, Dict]) -> str:
    """Generate a text summary report."""
    report = []
    report.append("=" * 60)
    report.append("NeMo Evaluator - Execution Summary Report")
    report.append("=" * 60)
    report.append(f"\nTimestamp: {summary.get('timestamp', 'Unknown')}")
    report.append(f"Total Benchmarks: {summary.get('total', 0)}")
    report.append(f"Successful: {summary.get('successful', 0)}")
    report.append(f"Failed: {summary.get('failed', 0)}")

    report.append("\n" + "=" * 60)
    report.append("Benchmark Results")
    report.append("=" * 60)

    for result in summary.get("results", []):
        status = "✓" if result["success"] else "✗"
        report.append(f"{status} {result['benchmark']}")
        if result["success"] and result["benchmark"] in metrics:
            # Add key metrics
            bench_metrics = metrics[result["benchmark"]]
            for key, value in list(bench_metrics.items())[:3]:  # First 3 metrics
                if isinstance(value, (int, float)):
                    report.append(f"    {key}: {value:.4f}")

    return "\n".join(report)


def export_to_mlflow_summary(summary: Dict, metrics: Dict[str, Dict]):
    """Export aggregated summary to MLflow."""
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        experiment = mlflow.get_experiment_by_name("llm-evaluation")

        if not experiment:
            console.print(
                "[yellow]MLflow experiment 'llm-evaluation' not found.[/yellow]"
            )
            return

        # Create a summary run
        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=f"summary_{summary.get('timestamp', 'unknown')}",
        ):
            mlflow.log_param("total_benchmarks", summary.get("total", 0))
            mlflow.log_param("successful", summary.get("successful", 0))
            mlflow.log_param("failed", summary.get("failed", 0))

            # Log aggregated metrics
            for benchmark_name, bench_metrics in metrics.items():
                for metric_name, metric_value in bench_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(
                            f"{benchmark_name}.{metric_name}", metric_value
                        )

        console.print("[green]✓ Summary exported to MLflow[/green]")

    except Exception as e:
        console.print(f"[yellow]Could not export to MLflow: {e}[/yellow]")


def main():
    """Main entry point."""
    import argparse
    from rich.prompt import Confirm

    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompts"
    )
    args = parser.parse_args()

    output_dir = get_output_dir()

    console.print(Panel.fit("[bold cyan]Results Aggregator[/bold cyan]"))

    # Find latest execution summary
    summaries = list(output_dir.glob("eval_*/execution_summary.yaml"))
    if not summaries:
        console.print("[red]No execution summaries found.[/red]")
        sys.exit(1)

    # Use most recent
    latest_summary = max(summaries, key=lambda p: p.stat().st_mtime)
    console.print(f"[green]Loading summary from: {latest_summary}[/green]")

    summary = load_execution_summary(latest_summary)

    # Aggregate metrics
    console.print("\n[yellow]Aggregating metrics...[/yellow]")
    metrics = aggregate_metrics_from_outputs(output_dir)
    console.print(f"[green]✓ Found metrics for {len(metrics)} benchmarks[/green]")

    # Generate report
    report = generate_summary_report(summary, metrics)
    console.print("\n" + report)

    # Save report
    report_file = latest_summary.parent / "aggregated_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    console.print(f"\n[green]✓ Report saved to {report_file}[/green]")

    # Export to MLflow
    if args.yes or Confirm.ask("\nExport summary to MLflow?"):
        export_to_mlflow_summary(summary, metrics)


if __name__ == "__main__":
    main()
