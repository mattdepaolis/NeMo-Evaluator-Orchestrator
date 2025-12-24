"""
Dataset management utilities for NeMo Evaluator Orchestrator.
Provides progress tracking, caching, and optimization for dataset loading.
"""
import os
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskID,
    TimeElapsedColumn,
)
from rich.panel import Panel
from rich.table import Table


class DatasetManager:
    """Manages dataset loading and caching for benchmark evaluations."""

    def __init__(self, cache_dir: Path, console: Optional[Console] = None):
        self.cache_dir = cache_dir
        self.console = console or Console()
        self.datasets_status = {}  # Track dataset loading status

    def get_benchmark_cache_dir(self, benchmark: str) -> Path:
        """Get the cache directory for a specific benchmark."""
        # Clean benchmark name for directory creation
        clean_name = benchmark.replace(".", "_").replace("/", "_").replace("-", "_")
        return self.cache_dir / clean_name

    def check_dataset_cache(self, benchmark: str) -> Dict[str, any]:
        """Check if dataset is already cached for a benchmark."""
        cache_dir = self.get_benchmark_cache_dir(benchmark)

        # Check for common dataset files/directories
        cache_status = {
            "cached": False,
            "size_mb": 0,
            "last_modified": None,
            "files": []
        }

        if cache_dir.exists():
            total_size = 0
            files = []

            # Recursively check directory contents
            for file_path in cache_dir.rglob("*"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    total_size += size
                    files.append({
                        "path": str(file_path.relative_to(cache_dir)),
                        "size_mb": size / (1024 * 1024),
                        "modified": file_path.stat().st_mtime
                    })

            if files:
                cache_status["cached"] = True
                cache_status["size_mb"] = total_size / (1024 * 1024)
                cache_status["last_modified"] = max(f["modified"] for f in files)
                cache_status["files"] = files

        return cache_status

    def preload_datasets(self, benchmarks: List[str], progress: Optional[Progress] = None) -> Dict[str, bool]:
        """
        Pre-load datasets for benchmarks to provide better user feedback.

        Args:
            benchmarks: List of benchmark names to preload
            progress: Optional progress bar instance

        Returns:
            Dictionary mapping benchmark names to success status
        """
        results = {}

        if not progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            ) as progress:
                return self._preload_datasets_internal(benchmarks, progress)

        return self._preload_datasets_internal(benchmarks, progress)

    def _preload_datasets_internal(self, benchmarks: List[str], progress: Progress) -> Dict[str, bool]:
        """Internal method for dataset preloading."""
        results = {}

        # First pass: Check cache status
        cache_check_task = progress.add_task("Checking dataset cache status...", total=len(benchmarks))

        for benchmark in benchmarks:
            cache_status = self.check_dataset_cache(benchmark)
            self.datasets_status[benchmark] = cache_status

            if cache_status["cached"]:
                self.console.print(f"  [green]✓[/green] {benchmark}: Dataset cached ({cache_status['size_mb']:.1f} MB)")
            else:
                self.console.print(f"  [yellow]⚠[/yellow] {benchmark}: Dataset not cached")

            progress.advance(cache_check_task)

        progress.remove_task(cache_check_task)

        # Second pass: Preload uncached datasets
        uncached_benchmarks = [
            b for b in benchmarks
            if not self.datasets_status[b]["cached"]
        ]

        if not uncached_benchmarks:
            self.console.print("\n[green]✓ All datasets are already cached![/green]")
            return {b: True for b in benchmarks}

        self.console.print(f"\n[yellow]Pre-loading {len(uncached_benchmarks)} uncached datasets...[/yellow]")

        # Use threading for parallel preloading (limited to avoid overwhelming the system)
        max_workers = min(3, len(uncached_benchmarks))

        preload_task = progress.add_task("Pre-loading datasets...", total=len(uncached_benchmarks))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_benchmark = {
                executor.submit(self._preload_single_dataset, benchmark): benchmark
                for benchmark in uncached_benchmarks
            }

            for future in as_completed(future_to_benchmark):
                benchmark = future_to_benchmark[future]
                try:
                    success = future.result()
                    results[benchmark] = success

                    if success:
                        self.console.print(f"  [green]✓[/green] {benchmark}: Dataset pre-loaded successfully")
                    else:
                        self.console.print(f"  [red]✗[/red] {benchmark}: Dataset pre-loading failed")

                except Exception as e:
                    self.console.print(f"  [red]✗[/red] {benchmark}: Dataset pre-loading error: {e}")
                    results[benchmark] = False

                progress.advance(preload_task)

        progress.remove_task(preload_task)

        # Set cached benchmarks as successful
        for benchmark in benchmarks:
            if self.datasets_status[benchmark]["cached"]:
                results[benchmark] = True

        return results

    def _preload_single_dataset(self, benchmark: str) -> bool:
        """
        Pre-load a single dataset by running a minimal evaluation configuration.

        This triggers dataset downloading and caching without running the full evaluation.
        """
        try:
            # Create a temporary config for dataset preloading
            cache_dir = self.get_benchmark_cache_dir(benchmark)

            # Create a minimal config that will trigger dataset loading
            preload_config = {
                "defaults": [{"execution": "local"}, {"deployment": "none"}, "_self_"],
                "execution": {
                    "output_dir": str(cache_dir / "preload_temp"),
                    "type": "local"
                },
                "target": {
                    "api_endpoint": {
                        "url": "http://dummy-endpoint/v1/chat/completions",  # Won't be reached
                        "model_id": "dummy-model",
                    }
                },
                "evaluation": {
                    "nemo_evaluator_config": {
                        "config": {
                            "params": {
                                "cache_dir": str(cache_dir),
                                "limit_samples": 1,  # Only load one sample to minimize work
                                "parallelism": 1,
                                "request_timeout": 5,  # Short timeout since endpoint is dummy
                            }
                        },
                        "tasks": [{"name": benchmark}],
                    },
                },
            }

            # Save temporary config
            config_file = cache_dir / "preload_config.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)

            import yaml
            with open(config_file, "w") as f:
                yaml.dump(preload_config, f, default_flow_style=False, sort_keys=False)

            # Run with very short timeout to trigger dataset loading but not evaluation
            env = os.environ.copy()
            env["NEMO_EVALUATOR_CACHE_DIR"] = str(cache_dir)

            process = subprocess.Popen(
                ["nemo-evaluator-launcher", "run", "--config", str(config_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=30,  # 30 second timeout for preloading
            )

            try:
                stdout, stderr = process.communicate(timeout=30)

                # Even if it fails due to dummy endpoint, dataset might be cached
                # Check if cache directory has been populated
                cache_status = self.check_dataset_cache(benchmark)
                if cache_status["cached"]:
                    return True

                # If process succeeded, dataset was likely preloaded
                return process.returncode == 0

            except subprocess.TimeoutExpired:
                # Timeout is expected with dummy endpoint
                process.kill()

                # Check if any dataset files were created during the timeout
                cache_status = self.check_dataset_cache(benchmark)
                return cache_status["cached"]

        except Exception:
            return False
        finally:
            # Clean up temporary config
            try:
                if config_file.exists():
                    config_file.unlink()
            except Exception:
                pass

    def get_dataset_info_table(self, benchmarks: List[str]) -> Table:
        """Create a table showing dataset cache information."""
        table = Table(
            title="Dataset Cache Status",
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("Benchmark", style="cyan", width=25)
        table.add_column("Status", style="yellow", width=12)
        table.add_column("Size", style="green", width=10)
        table.add_column("Files", style="dim", width=8)
        table.add_column("Last Modified", style="dim", width=15)

        for benchmark in benchmarks:
            cache_status = self.datasets_status.get(benchmark, self.check_dataset_cache(benchmark))

            if cache_status["cached"]:
                status = "[green]Cached[/green]"
                size_str = f"{cache_status['size_mb']:.1f} MB"
                file_count = len(cache_status["files"])

                if cache_status["last_modified"]:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(cache_status["last_modified"])
                    modified_str = dt.strftime("%Y-%m-%d %H:%M")
                else:
                    modified_str = "Unknown"
            else:
                status = "[yellow]Not Cached[/yellow]"
                size_str = "0 MB"
                file_count = 0
                modified_str = "N/A"

            table.add_row(
                benchmark[:25],
                status,
                size_str,
                str(file_count),
                modified_str,
            )

        return table


def create_dataset_manager(cache_dir: Path, console: Optional[Console] = None) -> DatasetManager:
    """Factory function to create a DatasetManager instance."""
    return DatasetManager(cache_dir, console)
