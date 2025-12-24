#!/usr/bin/env python3
"""
Dataset management CLI tool for NeMo Evaluator Orchestrator.
Provides commands for dataset caching, preloading, and status checking.
"""
import sys
import argparse
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from nemo_evaluator_orchestrator.utils.dataset_utils import create_dataset_manager
from nemo_evaluator_orchestrator.utils.benchmark_utils import load_benchmark_catalog
from nemo_evaluator_orchestrator.utils.paths import get_config_dir, get_cache_dir

console = Console()


def show_dataset_status(benchmarks: Optional[List[str]] = None, cache_dir: Optional[Path] = None):
    """Show the current dataset cache status."""
    console.print("[bold cyan]Dataset Cache Status[/bold cyan]\n")

    if cache_dir is None:
        cache_dir = get_cache_dir()

    dataset_manager = create_dataset_manager(cache_dir, console)

    if benchmarks is None:
        # Load all benchmarks from catalog
        config_dir = get_config_dir()
        catalog_path = config_dir / "benchmark_catalog.yaml"
        if catalog_path.exists():
            catalog = load_benchmark_catalog(catalog_path)
            benchmarks = []
            for harness_data in catalog.get("harnesses", {}).values():
                for task in harness_data.get("tasks", []):
                    benchmarks.append(task["name"])
        else:
            console.print("[red]No benchmark catalog found. Run 'nemo-catalog' first.[/red]")
            return

    # Get dataset status table
    status_table = dataset_manager.get_dataset_info_table(benchmarks)
    console.print(status_table)

    # Summary statistics
    cached_count = 0
    total_size = 0

    for benchmark in benchmarks:
        status = dataset_manager.check_dataset_cache(benchmark)
        if status["cached"]:
            cached_count += 1
            total_size += status["size_mb"]

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total benchmarks: {len(benchmarks)}")
    console.print(f"  Cached datasets: {cached_count}")
    console.print(f"  Total cache size: {total_size:.1f} MB")
    console.print(f"  Cache directory: {cache_dir}")


def preload_datasets(benchmarks: Optional[List[str]] = None, cache_dir: Optional[Path] = None, max_workers: int = 3):
    """Pre-load datasets for benchmarks."""
    console.print("[bold cyan]Dataset Pre-loading[/bold cyan]\n")

    if cache_dir is None:
        cache_dir = get_cache_dir()

    dataset_manager = create_dataset_manager(cache_dir, console)

    if benchmarks is None:
        # Load all benchmarks from catalog
        config_dir = get_config_dir()
        catalog_path = config_dir / "benchmark_catalog.yaml"
        if catalog_path.exists():
            catalog = load_benchmark_catalog(catalog_path)
            benchmarks = []
            for harness_data in catalog.get("harnesses", {}).values():
                for task in harness_data.get("tasks", []):
                    benchmarks.append(task["name"])
        else:
            console.print("[red]No benchmark catalog found. Run 'nemo-catalog' first.[/red]")
            return

    console.print(f"Pre-loading datasets for {len(benchmarks)} benchmarks...\n")

    # Preload datasets with progress tracking
    results = dataset_manager.preload_datasets(benchmarks)

    # Summary
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful

    console.print("\n[bold]Pre-loading Results:[/bold]")
    console.print(f"  Successful: {successful}")
    console.print(f"  Failed: {failed}")
    console.print(f"  Total: {len(results)}")

    if failed > 0:
        console.print("\n[red]Failed benchmarks:[/red]")
        for benchmark, success in results.items():
            if not success:
                console.print(f"  - {benchmark}")


def clear_cache(benchmarks: Optional[List[str]] = None, cache_dir: Optional[Path] = None, confirm: bool = True):
    """Clear dataset cache."""
    if cache_dir is None:
        cache_dir = get_cache_dir()

    dataset_manager = create_dataset_manager(cache_dir, console)

    if benchmarks:
        # Clear specific benchmarks
        console.print(f"[yellow]Clearing cache for {len(benchmarks)} benchmark(s)...[/yellow]")

        if confirm:
            from rich.prompt import Confirm
            if not Confirm.ask("Are you sure you want to clear the dataset cache?"):
                console.print("[yellow]Cancelled.[/yellow]")
                return

        cleared_count = 0
        for benchmark in benchmarks:
            cache_path = dataset_manager.get_benchmark_cache_dir(benchmark)
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                console.print(f"  [green]✓[/green] Cleared cache for {benchmark}")
                cleared_count += 1
            else:
                console.print(f"  [dim]⚠ No cache found for {benchmark}[/dim]")

        console.print(f"\n[green]Cleared cache for {cleared_count} benchmark(s)[/green]")
    else:
        # Clear all cache
        console.print(f"[yellow]Clearing all dataset cache in {cache_dir}...[/yellow]")

        if confirm:
            from rich.prompt import Confirm
            if not Confirm.ask("Are you sure you want to clear ALL dataset cache?"):
                console.print("[yellow]Cancelled.[/yellow]")
                return

        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]✓ Cleared all dataset cache[/green]")
        else:
            console.print("[dim]⚠ Cache directory does not exist[/dim]")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Dataset management for NeMo Evaluator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show dataset cache status")
    status_parser.add_argument("--benchmarks", nargs="*", help="Specific benchmarks to check")
    status_parser.add_argument("--cache-dir", type=Path, help="Cache directory path")

    # Preload command
    preload_parser = subparsers.add_parser("preload", help="Pre-load datasets")
    preload_parser.add_argument("--benchmarks", nargs="*", help="Specific benchmarks to preload")
    preload_parser.add_argument("--cache-dir", type=Path, help="Cache directory path")
    preload_parser.add_argument("--max-workers", type=int, default=3, help="Maximum parallel workers")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear dataset cache")
    clear_parser.add_argument("--benchmarks", nargs="*", help="Specific benchmarks to clear")
    clear_parser.add_argument("--cache-dir", type=Path, help="Cache directory path")
    clear_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "status":
            show_dataset_status(args.benchmarks, args.cache_dir)
        elif args.command == "preload":
            preload_datasets(args.benchmarks, args.cache_dir, args.max_workers)
        elif args.command == "clear":
            clear_cache(args.benchmarks, args.cache_dir, not args.yes)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
