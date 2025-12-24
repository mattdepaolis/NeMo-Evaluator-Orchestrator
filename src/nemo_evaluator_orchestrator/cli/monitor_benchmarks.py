#!/usr/bin/env python3
"""
Real-time monitoring tool for running benchmark evaluations.
Shows container status, logs, and progress information.
"""
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from datetime import datetime

from nemo_evaluator_orchestrator.utils.benchmark_utils import load_benchmark_catalog
from nemo_evaluator_orchestrator.utils.paths import get_project_root, get_output_dir
from nemo_evaluator_orchestrator.utils.container_health import (
    get_container_health_status,
    get_container_resource_usage,
)

console = Console()

# Track errors for display
_error_history: list = []
_max_error_history = 10


def get_running_containers() -> list:
    """Get list of running evaluation containers with enhanced health information."""
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=client-",
                "--format",
                "{{.Names}}|{{.Status}}|{{.Image}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        containers = []
        for line in result.stdout.strip().split("\n"):
            if line and "|" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    container_name = parts[0]
                    # Get detailed health status
                    health_info = get_container_health_status(container_name)
                    # Get resource usage
                    resource_usage = get_container_resource_usage(container_name)
                    
                    # Extract benchmark name from container name
                    # Format: client-<benchmark>-<timestamp>-<id>
                    benchmark = "unknown"
                    if container_name.startswith("client-"):
                        benchmark_part = container_name.split("client-")[1]
                        # Remove timestamp and ID parts
                        benchmark = benchmark_part.split("-")[0] if "-" in benchmark_part else benchmark_part
                        # Handle dots in benchmark names
                        if "." in benchmark:
                            benchmark = benchmark.replace(".", " ").title()

                    containers.append(
                        {
                            "name": container_name,
                            "status": parts[1],
                            "image": parts[2] if len(parts) > 2 else "unknown",
                            "health": health_info.get("health", "none"),
                            "uptime": health_info.get("uptime", "N/A"),
                            "resources": resource_usage,
                            "benchmark": benchmark,
                        }
                    )
        return containers
    except Exception as e:
        # Track error
        _error_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": "container_list",
            "error": str(e),
        })
        if len(_error_history) > _max_error_history:
            _error_history.pop(0)
        return []


def get_container_logs(container_name: str, lines: int = 10) -> str:
    """Get recent logs from a container."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Combine stdout and stderr (some containers log to stderr)
            logs = result.stdout + result.stderr
            return logs.strip()
    except Exception:
        pass
    return "No logs available"


def get_invocation_status() -> list:
    """Get status of recent invocations."""
    try:
        result = subprocess.run(
            ["nemo-evaluator-launcher", "ls", "runs", "--since", "2h"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            invocations = []
            # Skip header lines
            for line in lines[3:]:  # Skip first 3 header lines
                if line.strip() and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 4:
                        invocations.append(
                            {
                                "id": parts[0],
                                "timestamp": parts[1],
                                "num_jobs": parts[2],
                                "executor": parts[3],
                                "benchmarks": parts[4] if len(parts) > 4 else "",
                            }
                        )
            return invocations
    except Exception:
        pass
    return []


def create_status_table(containers: list, invocations: list) -> Table:
    """Create a status table showing current evaluations with health status."""
    table = Table(
        title="Running Benchmarks", show_header=True, header_style="bold cyan"
    )
    table.add_column("Container", style="cyan", width=30)
    table.add_column("Health", style="yellow", width=12)
    table.add_column("Status", style="yellow", width=15)
    table.add_column("Benchmark", style="green", width=25)
    table.add_column("Uptime", style="dim", width=10)
    table.add_column("Resources", style="dim", width=15)

    if not containers:
        table.add_row("No containers running", "", "", "", "", "")
        return table

    # Match containers to invocations
    for container in containers:
        container_name = container["name"]
        status = container["status"]
        health = container.get("health", "none")
        uptime = container.get("uptime", "N/A")

        # Format health status with color
        health_display = health
        if health == "healthy":
            health_display = "[green]✓ healthy[/green]"
        elif health == "unhealthy":
            health_display = "[red]✗ unhealthy[/red]"
        elif health == "starting":
            health_display = "[yellow]⏳ starting[/yellow]"
        elif health == "none":
            # Container is running but has no health check configured
            health_display = "[dim]no health check[/dim]"
        else:
            health_display = f"[dim]{health}[/dim]"

        # Extract benchmark name from container name
        benchmark = "unknown"
        if "client-" in container_name:
            parts = container_name.replace("client-", "").split("-")
            if len(parts) > 0:
                benchmark = parts[0].replace("_", ".")

        # Format resource usage
        resources = container.get("resources")
        resources_display = "N/A"
        if resources:
            cpu = resources.get("cpu_percent", "N/A")
            mem = resources.get("memory_percent", "N/A")
            resources_display = f"CPU:{cpu} MEM:{mem}"

        table.add_row(
            container_name[:30],
            health_display,
            status[:15],
            benchmark[:25],
            uptime[:10],
            resources_display[:15],
        )

    return table


def create_logs_panel(containers: list) -> Panel:
    """Create a panel showing recent logs from containers."""
    if not containers:
        return Panel("No containers running", title="Recent Logs")

    # Get logs from the first running container
    container = containers[0]
    logs = get_container_logs(container["name"], lines=15)

    # Truncate long logs
    log_lines = logs.split("\n")
    if len(log_lines) > 15:
        log_lines = log_lines[-15:]
        logs = "\n".join(log_lines)

    return Panel(logs, title=f"Logs: {container['name'][:50]}", border_style="blue")








def create_errors_panel() -> Panel:
    """Create a panel showing recent errors."""
    if not _error_history:
        return Panel("No recent errors", title="Recent Errors", border_style="green")

    error_lines = []
    for error in _error_history[-5:]:  # Show last 5 errors
        error_lines.append(
            f"[{error['timestamp']}] [{error['type']}] {error['error'][:60]}"
        )

    error_text = "\n".join(error_lines) if error_lines else "No errors"
    return Panel(error_text, title="Recent Errors", border_style="red")


def main():
    """Main monitoring loop."""
    console.print(
        Panel.fit("[bold cyan]NeMo Evaluator - Real-time Monitor[/bold cyan]")
    )
    console.print("[yellow]Press Ctrl+C to exit[/yellow]\n")

    try:
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                containers = get_running_containers()
                invocations = get_invocation_status()

                # Calculate layout sizes
                status_size = len(containers) + 6 if containers else 5
                logs_size = 30
                errors_size = min(len(_error_history) + 3, 8) if _error_history else 3

                # Create layout
                layout = Layout()
                layout.split_column(
                    Layout(name="status", size=status_size),
                    Layout(name="logs", size=logs_size),
                    Layout(name="errors", size=errors_size),
                )

                # Status table
                status_table = create_status_table(containers, invocations)
                layout["status"].update(status_table)

                # Logs panel
                logs_panel = create_logs_panel(containers)
                layout["logs"].update(logs_panel)

                # Errors panel
                errors_panel = create_errors_panel()
                layout["errors"].update(errors_panel)

                # Update live display
                live.update(layout)

                time.sleep(2)  # Refresh every 2 seconds

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped.[/yellow]")
    except Exception as e:
        # Track unexpected errors
        _error_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": "monitor_error",
            "error": str(e),
        })
        console.print(f"\n[red]Error in monitoring: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
