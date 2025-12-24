#!/usr/bin/env python3
"""
Container management utility for NeMo Evaluator benchmarks.
Checks, pulls, and verifies required Docker containers.
"""
import sys
import subprocess
import yaml
from pathlib import Path
from typing import Set, Dict, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from nemo_evaluator_orchestrator.utils.benchmark_utils import (
    load_benchmark_catalog,
    get_benchmark_full_name,
)
from nemo_evaluator_orchestrator.utils.paths import get_config_dir
from nemo_evaluator_orchestrator.utils.container_health import (
    get_container_health_status,
    wait_for_container_health,
)

console = Console()


def get_local_containers() -> Set[str]:
    """Get list of locally available Docker containers."""
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return set(result.stdout.strip().split("\n"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()


def get_required_containers(benchmarks: List[str], catalog: Dict) -> Set[str]:
    """Get required containers for selected benchmarks."""
    containers = set()

    # Map benchmark names to harnesses - include both full names and short names
    benchmark_to_harness = {}
    for harness_name, harness_data in catalog["harnesses"].items():
        for task in harness_data["tasks"]:
            full_name = get_benchmark_full_name(harness_name, task["name"])
            short_name = task["name"]
            # Map both full name and short name
            benchmark_to_harness[full_name] = harness_name
            benchmark_to_harness[short_name] = harness_name

    # Find containers for selected benchmarks
    for benchmark in benchmarks:
        harness = benchmark_to_harness.get(benchmark)
        if harness and harness in catalog["harnesses"]:
            container = catalog["harnesses"][harness].get("container")
            if container:
                containers.add(container)
        else:
            # Try to find by checking if it's a task name directly
            for harness_name, harness_data in catalog["harnesses"].items():
                for task in harness_data["tasks"]:
                    if task["name"] == benchmark or benchmark.endswith(
                        "." + task["name"]
                    ):
                        container = harness_data.get("container")
                        if container:
                            containers.add(container)
                            break
                else:
                    continue
                break

    return containers


def check_container_available(container: str, local_containers: Set[str]) -> bool:
    """Check if a container is available locally."""
    # Check exact match
    if container in local_containers:
        return True

    # Check if base image exists (without tag)
    base_image = container.split(":")[0]
    for local in local_containers:
        if local.startswith(base_image + ":"):
            return True

    return False


def pull_container(container: str) -> bool:
    """Pull a Docker container."""
    console.print(f"[yellow]Pulling {container}...[/yellow]")
    try:
        result = subprocess.run(
            ["docker", "pull", container], capture_output=True, text=True, check=True
        )
        console.print(f"[green]✓ Successfully pulled {container}[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Failed to pull {container}[/red]")
        console.print(f"[red]Error: {e.stderr}[/red]")
        return False


def verify_containers(
    containers: Set[str], local_containers: Set[str]
) -> Dict[str, bool]:
    """Verify which containers are available."""
    status = {}
    for container in containers:
        status[container] = check_container_available(container, local_containers)
    return status


def get_running_container_instances() -> Dict[str, str]:
    """Get currently running container instances and their images."""
    running = {}
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--format",
                "{{.Image}}|{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line and "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        image = parts[0]
                        name = parts[1]
                        running[image] = name
    except Exception:
        pass
    return running


def display_container_status(containers: Set[str], status: Dict[str, bool]):
    """Display container availability status with health information."""
    table = Table(title="Container Status")
    table.add_column("Container", style="cyan")
    table.add_column("Availability", style="green", width=15)
    table.add_column("Health", style="yellow", width=12)
    table.add_column("Status", style="dim", width=20)

    running_instances = get_running_container_instances()

    for container in sorted(containers):
        available = "✓ Available" if status.get(container, False) else "✗ Missing"
        style = "green" if status.get(container, False) else "red"

        # Check if container is running and get health
        health_display = "[dim]N/A[/dim]"
        status_display = "[dim]Not running[/dim]"

        if container in running_instances:
            container_name = running_instances[container]
            health_info = get_container_health_status(container_name, use_cache=False)
            health = health_info.get("health", "none")
            container_status = health_info.get("status", "unknown")

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

            status_display = f"[dim]{container_status}[/dim]"

        table.add_row(
            container,
            f"[{style}]{available}[/{style}]",
            health_display,
            status_display,
        )

    console.print(table)


def main():
    """Main entry point."""
    config_dir = get_config_dir()

    # Load selected benchmarks
    selected_path = config_dir / "selected_benchmarks.yaml"
    if not selected_path.exists():
        console.print("[red]Error: No selected benchmarks found.[/red]")
        console.print("Run 'nemo-select' first to select benchmarks.")
        sys.exit(1)

    with open(selected_path, "r") as f:
        selected_config = yaml.safe_load(f)

    benchmarks = selected_config.get("selected_benchmarks", [])

    # Load catalog
    catalog_path = config_dir / "benchmark_catalog.yaml"
    catalog = load_benchmark_catalog(catalog_path)

    # Get required containers
    required_containers = get_required_containers(benchmarks, catalog)

    console.print(
        Panel.fit(
            f"[bold cyan]Container Manager[/bold cyan]\nRequired containers: {len(required_containers)}"
        )
    )

    # Check local containers
    console.print("\n[yellow]Checking local containers...[/yellow]")
    local_containers = get_local_containers()

    # Verify status
    status = verify_containers(required_containers, local_containers)
    display_container_status(required_containers, status)

    # Find missing containers
    missing = [c for c, available in status.items() if not available]

    if not missing:
        console.print(
            "\n[bold green]✓ All required containers are available![/bold green]"
        )
        return

    console.print(f"\n[yellow]Missing {len(missing)} container(s)[/yellow]")

    # Ask to pull missing containers
    from rich.prompt import Confirm

    if Confirm.ask("\nPull missing containers?"):
        console.print("\n[yellow]Note: These containers are publicly available and don't require NGC authentication.[/yellow]\n")

        for container in missing:
            if pull_container(container):
                # Verify container was pulled successfully
                console.print(f"[yellow]  Verifying {container}...[/yellow]")
                local_containers = get_local_containers()
                if check_container_available(container, local_containers):
                    console.print(f"[green]✓ Container {container} verified[/green]")

                    # If container is running, check its health
                    running_instances = get_running_container_instances()
                    if container in running_instances:
                        container_name = running_instances[container]
                        console.print(
                            f"[yellow]  Container {container_name} is running, checking health...[/yellow]"
                        )
                        health_info = get_container_health_status(
                            container_name, use_cache=False
                        )
                        health = health_info.get("health", "none")
                        if health == "healthy":
                            console.print(
                                f"[green]✓ Container {container_name} is healthy[/green]"
                            )
                        elif health == "unhealthy":
                            console.print(
                                f"[yellow]⚠ Container {container_name} is unhealthy[/yellow]"
                            )
                        elif health == "starting":
                            console.print(
                                f"[yellow]⏳ Container {container_name} is starting...[/yellow]"
                            )
                            # Optionally wait for health (with timeout)
                            try:
                                success, final_status = wait_for_container_health(
                                    container_name,
                                    timeout=30,
                                    check_interval=2,
                                    target_health="running",
                                )
                                if success:
                                    console.print(
                                        f"[green]✓ Container {container_name} is now running[/green]"
                                    )
                                else:
                                    console.print(
                                        f"[yellow]⚠ Container status: {final_status}[/yellow]"
                                    )
                            except Exception:
                                pass  # Non-critical
                else:
                    console.print(
                        f"[yellow]⚠ Could not verify {container} after pull[/yellow]"
                    )
            else:
                console.print(
                    f"[red]Failed to pull {container}. Please check your NGC credentials.[/red]"
                )
                console.print("[yellow]You can pull it manually later with:[/yellow]")
                console.print(f"  docker pull {container}")


if __name__ == "__main__":
    main()
