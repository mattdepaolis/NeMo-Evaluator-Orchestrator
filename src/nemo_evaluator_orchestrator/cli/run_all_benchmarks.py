#!/usr/bin/env python3
"""
Main execution orchestrator for running selected benchmarks.
Handles execution sequencing, error handling, and progress tracking.
"""
import sys
import os
import yaml
import subprocess
import time
import argparse
import threading
import signal
from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskID,
)
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

from nemo_evaluator_orchestrator.utils.benchmark_utils import load_benchmark_catalog, get_benchmark_full_name
from nemo_evaluator_orchestrator.utils.paths import get_project_root, get_config_dir, get_output_dir, get_cache_dir
from nemo_evaluator_orchestrator.utils.endpoint_utils import (
    convert_localhost_for_docker,
    should_convert_endpoint,
    validate_endpoint_url,
    validate_model_endpoint,
    check_model_exists,
)
from nemo_evaluator_orchestrator.utils.container_health import (
    get_container_health_status,
    wait_for_container_health,
    wait_for_service_ready,
    get_container_port_mapping,
)
from nemo_evaluator_orchestrator.utils.retry_utils import (
    retry_with_backoff,
    is_retryable_error,
    get_error_context,
)
from nemo_evaluator_orchestrator.utils.dataset_utils import create_dataset_manager
from nemo_evaluator_orchestrator.utils.vllm_utils import create_vllm_optimizer

console = Console()

# Global list to track running evaluation processes for graceful shutdown
running_processes = []


# Global variable to track if we should stop vLLM
stop_vllm_on_exit = False

def stop_vllm_server():
    """Stop any running vLLM server processes."""
    try:
        # Find and stop vLLM processes
        result = subprocess.run(
            ['pgrep', '-f', 'vllm.entrypoints.openai.api_server'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            console.print(f"[dim]  Stopping {len(pids)} vLLM server process(es)...[/dim]")

            # Try graceful termination first
            subprocess.run(['pkill', '-TERM', '-f', 'vllm.entrypoints.openai.api_server'], timeout=10)

            # Wait a bit and check if they're still running
            import time
            time.sleep(2)

            result2 = subprocess.run(
                ['pgrep', '-f', 'vllm.entrypoints.openai.api_server'],
                capture_output=True, text=True, timeout=5
            )
            if result2.returncode == 0 and result2.stdout.strip():
                # Force kill remaining processes
                console.print("[yellow]  Force killing remaining vLLM processes...[/yellow]")
                subprocess.run(['pkill', '-9', '-f', 'vllm.entrypoints.openai.api_server'], timeout=5)

            console.print("[green]✓ Stopped vLLM server[/green]")
        else:
            console.print("[dim]  No vLLM server processes found[/dim]")
    except Exception as e:
        console.print(f"[red]  Error stopping vLLM server: {e}[/red]")


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) to gracefully stop running evaluations."""
    console.print(f"\n[yellow]⚠ Received signal {signum}. Stopping evaluations gracefully...[/yellow]")

    # Stop all running processes
    stopped_count = 0
    for process_info in running_processes[:]:  # Copy list to avoid modification during iteration
        try:
            if process_info['process'] and process_info['process'].poll() is None:
                console.print(f"[dim]  Stopping {process_info['benchmark']}...[/dim]")
                process_info['process'].terminate()

                # Wait up to 10 seconds for graceful shutdown
                try:
                    process_info['process'].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    console.print(f"[yellow]  Force killing {process_info['benchmark']}...[/yellow]")
                    process_info['process'].kill()
                    process_info['process'].wait()

                stopped_count += 1
                running_processes.remove(process_info)
        except Exception as e:
            console.print(f"[red]  Error stopping {process_info['benchmark']}: {e}[/red]")

    if stopped_count > 0:
        console.print(f"[green]✓ Stopped {stopped_count} evaluation(s)[/green]")

    # Stop any running containers
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "label=nemo-evaluation"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            container_ids = result.stdout.strip().split('\n')
            console.print(f"[dim]  Stopping {len(container_ids)} evaluation container(s)...[/dim]")
            subprocess.run(["docker", "stop"] + container_ids, timeout=30)
            console.print("[green]✓ Stopped evaluation containers[/green]")
    except Exception as e:
        console.print(f"[red]  Error stopping containers: {e}[/red]")

    # Stop vLLM server if requested
    global stop_vllm_on_exit
    if stop_vllm_on_exit:
        stop_vllm_server()

    console.print("[green]✓ All evaluations stopped. Exiting.[/green]")
    sys.exit(0)


def load_configs(config_dir: Path) -> tuple:
    """Load all configuration files."""
    # Load selected benchmarks
    selected_path = config_dir / "selected_benchmarks.yaml"
    if not selected_path.exists():
        console.print("[red]Error: No selected benchmarks found.[/red]")
        console.print("Run 'nemo-select' first.")
        sys.exit(1)

    with open(selected_path, "r") as f:
        selected_config = yaml.safe_load(f)

    # Load model config
    model_path = config_dir / "model_config.yaml"
    if not model_path.exists():
        console.print("[red]Error: model_config.yaml not found.[/red]")
        sys.exit(1)

    with open(model_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Load eval params
    eval_path = config_dir / "eval_params.yaml"
    if not eval_path.exists():
        console.print("[red]Error: eval_params.yaml not found.[/red]")
        sys.exit(1)

    with open(eval_path, "r") as f:
        eval_params = yaml.safe_load(f)

    return selected_config, model_config, eval_params


def generate_eval_config(
    benchmark: str,
    model_config: Dict,
    eval_params: Dict,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    console: Optional[Console] = None,
    vllm_optimizer: Optional[any] = None,
) -> Path:
    """Generate evaluation configuration file for a benchmark."""
    # Convert endpoint URL for Docker container compatibility
    # Containers on Linux can't reach 'localhost' - they need Docker bridge IP
    endpoint_url = model_config["model"]["endpoint_url"]
    if should_convert_endpoint():
        converted_url = convert_localhost_for_docker(endpoint_url)
        if converted_url != endpoint_url and console:
            console.print(f"[yellow]  Converting endpoint: {endpoint_url} → {converted_url}[/yellow]")
            console.print("[dim]    (Containers need Docker bridge IP to reach host)[/dim]")
        endpoint_url = converted_url
    
    config = {
        "defaults": [{"execution": "local"}, {"deployment": "none"}, "_self_"],
        "execution": {"output_dir": str(output_dir)},
        "target": {
            "api_endpoint": {
                "url": endpoint_url,
                "model_id": model_config["model"]["model_id"],
            }
        },
        "evaluation": {
            "nemo_evaluator_config": {
                "config": {"params": eval_params["global"].copy()}
            },
            "tasks": [{"name": benchmark}],
        },
    }

    # Add cache directory configuration if enabled
    if cache_dir:
        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create benchmark-specific cache subdirectory
        benchmark_cache_dir = cache_dir / benchmark.replace(".", "_").replace("/", "_")
        benchmark_cache_dir.mkdir(parents=True, exist_ok=True)

        # Add cache_dir to evaluation params
        # Note: The cache_dir parameter tells benchmarks where to store/read cached datasets
        # This is benchmark-specific, so each benchmark gets its own subdirectory
        if (
            "cache_dir"
            not in config["evaluation"]["nemo_evaluator_config"]["config"]["params"]
        ):
            config["evaluation"]["nemo_evaluator_config"]["config"]["params"][
                "cache_dir"
            ] = str(benchmark_cache_dir)

        # Also set as environment variable for benchmarks that read from env
        # This ensures the cache directory is accessible inside containers
        if console:
            console.print(f"[dim]  Cache directory: {benchmark_cache_dir}[/dim]")

    # Save config

    # Add API key only if explicitly configured (not null/empty)
    api_key_name = model_config["model"].get("api_key_name")
    if api_key_name and api_key_name != "null":
        config["target"]["api_endpoint"]["api_key_name"] = api_key_name

    # Add MLflow export
    config["execution"]["auto_export"] = {"destinations": ["mlflow"]}
    config["export"] = {
        "mlflow": {
            "tracking_uri": "http://127.0.0.1:5000",
            "experiment_name": "llm-evaluation",
            "description": f"Evaluation of {model_config['model']['model_id']}",
            "tags": {
                "model_family": model_config["model"]
                .get("metadata", {})
                .get("model_family", "unknown"),
                "framework": model_config["model"].get("framework", "unknown"),
            },
            "log_artifacts": True,
            "log_logs": False,
        }
    }

    # Apply benchmark-specific overrides
    if "benchmark_overrides" in eval_params and eval_params["benchmark_overrides"]:
        bench_overrides = eval_params["benchmark_overrides"].get(benchmark, {})
        if bench_overrides:
            for key, value in bench_overrides.items():
                config["evaluation"]["nemo_evaluator_config"]["config"]["params"][
                    key
                ] = value

    # Apply vLLM optimizations if available
    if vllm_optimizer:
        config = vllm_optimizer.get_vllm_optimized_config(config)

    # Note: Benchmarks requiring judges are now handled by skipping them if API key is not available
    # This approach is more reliable than trying to disable the judge in the config
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / f"{benchmark.replace('.', '_')}.yaml"

    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_file


def get_invocation_id_from_output(output: str) -> Optional[str]:
    """Extract invocation ID from nemo-evaluator-launcher output."""
    for line in output.split("\n"):
        if "Invocation ID:" in line:
            return line.split("Invocation ID:")[-1].strip()
    return None


def get_container_name_from_invocation(invocation_id: str) -> Optional[str]:
    """Get container name for a given invocation ID."""
    try:
        result = subprocess.run(
            ["nemo-evaluator-launcher", "status", invocation_id],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Container" in line and "client-" in line:
                    # Extract container name
                    parts = line.split()
                    for part in parts:
                        if part.startswith("client-"):
                            return part
    except Exception:
        pass
    return None


def perform_preflight_checks(model_config: Dict, console: Console) -> bool:
    """
    Perform pre-flight health checks before starting benchmarks.
    
    Args:
        model_config: Model configuration dictionary
        console: Rich console for output
        
    Returns:
        True if all checks pass, False otherwise
    """
    console.print("\n[yellow]Performing pre-flight health checks...[/yellow]")
    
    # Validate endpoint URL format
    endpoint_url = model_config["model"]["endpoint_url"]
    is_valid, error_msg = validate_endpoint_url(endpoint_url)
    if not is_valid:
        console.print(f"[red]✗ Invalid endpoint URL: {error_msg}[/red]")
        console.print(f"[yellow]  Endpoint: {endpoint_url}[/yellow]")
        return False
    console.print("[green]✓ Endpoint URL format is valid[/green]")
    
    # Validate endpoint is accessible
    console.print("[yellow]  Checking endpoint accessibility...[/yellow]")
    is_accessible, access_error = validate_model_endpoint(endpoint_url, timeout=10)
    if not is_accessible:
        console.print(f"[red]✗ Endpoint is not accessible: {access_error}[/red]")
        console.print(f"[yellow]  Endpoint: {endpoint_url}[/yellow]")
        console.print("[yellow]  Troubleshooting steps:[/yellow]")
        console.print("    1. Ensure the model server is running")
        console.print("    2. Check the endpoint URL is correct")
        console.print("    3. Verify network connectivity")
        if "localhost" in endpoint_url or "127.0.0.1" in endpoint_url:
            console.print("    4. For Docker containers, ensure endpoint is accessible from containers")
        return False
    console.print("[green]✓ Endpoint is accessible[/green]")
    
    # Check if model exists at endpoint
    model_id = model_config["model"].get("model_id")
    if model_id:
        console.print(f"[yellow]  Checking if model '{model_id}' exists at endpoint...[/yellow]")
        model_exists, model_error, available_models = check_model_exists(endpoint_url, model_id, timeout=10, fetch_available=True)
        if not model_exists:
            console.print(f"[red]✗ Model not found: {model_error}[/red]")
            console.print(f"[yellow]  Model ID: {model_id}[/yellow]")
            console.print(f"[yellow]  Endpoint: {endpoint_url}[/yellow]")
            
            # Show available models if we were able to fetch them
            if available_models:
                console.print(f"\n[cyan]  Available models at endpoint:[/cyan]")
                for available_model in available_models:
                    console.print(f"    • {available_model}")
                console.print(f"\n[yellow]  💡 Tip: Update model_id in config/model_config.yaml to one of the models above[/yellow]")
            
            console.print("\n[yellow]  Troubleshooting steps:[/yellow]")
            console.print("    1. Verify the model is deployed at the endpoint")
            console.print("    2. Check the model_id matches the deployed model name (case-sensitive)")
            if available_models:
                console.print("    3. Use one of the available models listed above")
            else:
                console.print("    3. Check available models: curl http://localhost:8000/v1/models")
            console.print("    4. For NIM: Ensure the model container is running")
            console.print("    5. For vLLM: Verify the model path is correct")
            console.print("    6. Check endpoint logs for model loading errors")
            return False
        console.print(f"[green]✓ Model '{model_id}' exists at endpoint[/green]")
    else:
        console.print("[yellow]  ⚠ No model_id specified, skipping model existence check[/yellow]")
    
    # Check if Docker is available
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            console.print("[red]✗ Docker is not running or not accessible[/red]")
            return False
        console.print("[green]✓ Docker is available[/green]")
    except Exception as e:
        console.print(f"[red]✗ Cannot access Docker: {e}[/red]")
        return False
    
    # Check for running evaluation containers and their health
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=client-",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            containers = [c.strip() for c in result.stdout.strip().split("\n") if c.strip()]
            if containers:
                console.print(f"[yellow]  Found {len(containers)} running evaluation container(s)[/yellow]")
                # Check health of existing containers
                for container_name in containers[:3]:  # Check first 3
                    health_info = get_container_health_status(container_name, use_cache=False)
                    health = health_info.get("health", "none")
                    if health == "unhealthy":
                        console.print(f"[yellow]  ⚠ Container {container_name} is unhealthy[/yellow]")
                    elif health == "healthy":
                        console.print(f"[green]  ✓ Container {container_name} is healthy[/green]")
    except Exception:
        pass  # Non-critical check
    
    console.print("[green]✓ Pre-flight checks completed[/green]\n")
    return True


def get_container_status(container_name: Optional[str]) -> Dict[str, str]:
    """Get status information for a Docker container."""
    if not container_name:
        return {"status": "unknown", "uptime": "N/A"}

    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Status}}|{{.State.StartedAt}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split("|")
            status = parts[0] if parts else "unknown"
            started = parts[1] if len(parts) > 1 else ""

            # Calculate uptime
            uptime = "N/A"
            if started:
                try:
                    # Try to parse and calculate uptime
                    from dateutil import parser as date_parser

                    start_time = date_parser.parse(started)
                    uptime_seconds = (
                        datetime.now(start_time.tzinfo) - start_time
                    ).total_seconds()
                    uptime = f"{int(uptime_seconds // 60)}m {int(uptime_seconds % 60)}s"
                except ImportError:
                    # dateutil not available, just show start time
                    uptime = started[:19] if len(started) > 19 else started
                except Exception:
                    uptime = started[:19] if len(started) > 19 else started

            return {"status": status, "uptime": uptime}
    except Exception:
        pass

    return {"status": "unknown", "uptime": "N/A"}


def get_container_logs_tail(container_name: Optional[str], lines: int = 3) -> str:
    """Get last few lines of container logs."""
    if not container_name:
        return ""

    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return ""


def run_benchmark(
    config_file: Path,
    benchmark: str,
    progress: Progress,
    task: TaskID,
    env: Optional[Dict[str, str]] = None,
    benchmark_status: Optional[Dict[str, Dict]] = None,
) -> tuple[bool, str, Dict]:
    """Run a single benchmark evaluation with progress monitoring."""
    invocation_id = None
    container_name = None
    start_time = time.time()
    container_start_time = None
    evaluation_start_time = None
    phase = "initializing"
    dataset_loading_detected = False

    try:
        # Prepare environment variables for subprocess
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Start the evaluation process
        process = subprocess.Popen(
            ["nemo-evaluator-launcher", "run", "--config", str(config_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=process_env,  # Pass environment variables
        )

        # Track the running process for graceful shutdown
        running_processes.append({
            'process': process,
            'benchmark': benchmark,
            'start_time': start_time
        })

        # Monitor progress while process runs
        last_update = time.time()
        stdout_lines = []
        process_start_time = time.time()

        while process.poll() is None:
            # Check for timeout (max 30 minutes per benchmark)
            if time.time() - process_start_time > 30 * 60:  # 30 minutes timeout
                console.print(f"[yellow]⚠️  Timeout reached for {benchmark}, terminating process...[/yellow]")
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
                return False, "timeout", {"total_time": time.time() - process_start_time}

            # Update progress every 3 seconds
            if time.time() - last_update > 3:
                elapsed = time.time() - start_time
                elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

                # Try to get invocation ID from running containers
                if not invocation_id:
                    # Check for new containers
                    try:
                        result = subprocess.run(
                            [
                                "docker",
                                "ps",
                                "--filter",
                                "name=client-",
                                "--format",
                                "{{.Names}}",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=2,
                        )
                        if result.returncode == 0:
                            containers = [
                                c.strip()
                                for c in result.stdout.strip().split("\n")
                                if c.strip()
                            ]
                            # Find container matching this benchmark
                            for cont in containers:
                                if (
                                    benchmark.replace(".", "_").replace("-", "_")
                                    in cont
                                ):
                                    container_name = cont
                                    if not container_start_time:
                                        container_start_time = time.time()
                                        phase = "container_started"
                                        
                                        # Wait for container to be healthy (with timeout)
                                        try:
                                            health_success, health_status = wait_for_container_health(
                                                container_name,
                                                timeout=30,
                                                check_interval=2,
                                                target_health="running",  # Just check if running, healthy may take longer
                                            )
                                            if not health_success:
                                                console.print(f"[yellow]  ⚠ Container {container_name} health: {health_status}[/yellow]")
                                        except Exception:
                                            pass  # Non-critical, continue monitoring

                                    # Check for dataset loading indicators in container logs
                                    if container_name and not dataset_loading_detected:
                                        try:
                                            logs = get_container_logs_tail(container_name, lines=5)
                                            logs_lower = logs.lower()
                                            # Look for dataset loading indicators
                                            if any(keyword in logs_lower for keyword in [
                                                "downloading", "dataset", "loading", "cache",
                                                "huggingface", "nltk", "punkt"
                                            ]):
                                                dataset_loading_detected = True
                                                phase = "dataset_loading"
                                        except Exception:
                                            pass

                                # Also check for file creation as progress indicator
                                if container_name and evaluation_start_time is None:
                                    # Check if result files have been created
                                    result_files = list(output_dir.glob(f"**/*{benchmark.replace('.', '_')}*/artifacts/*.json"))
                                    if result_files:
                                        evaluation_start_time = time.time()
                                        phase = "evaluating"

                                # Try to get invocation ID from status with retry logic
                                    def get_invocation_id():
                                        inv_ids = subprocess.run(
                                            [
                                                "nemo-evaluator-launcher",
                                                "ls",
                                                "runs",
                                                "--since",
                                                "10m",
                                            ],
                                            capture_output=True,
                                            text=True,
                                            timeout=5,
                                        )
                                        return inv_ids
                                    
                                    # Use retry logic for status retrieval (handles 500 errors)
                                    try:
                                        inv_ids = retry_with_backoff(
                                            get_invocation_id,
                                            max_retries=3,
                                            base_delay=1.0,
                                            max_delay=10.0,
                                            log_retries=False,  # Don't log every retry to avoid spam
                                        )
                                        
                                        if inv_ids.returncode == 0:
                                            for line in inv_ids.stdout.split("\n"):
                                                if benchmark in line and "|" in line:
                                                    invocation_id = line.split("|")[
                                                        0
                                                    ].strip()
                                                    if not evaluation_start_time:
                                                        evaluation_start_time = time.time()
                                                        phase = "evaluating"
                                                    break
                                    except Exception:
                                        pass  # Continue monitoring even if status retrieval fails
                                    break
                    except Exception:
                        pass

                # Update progress description with status and health
                if container_name:
                    container_status = get_container_status(container_name)
                    # Get health status for better monitoring
                    health_info = get_container_health_status(container_name, use_cache=True)
                    health = health_info.get("health", "none")
                    
                    if phase == "evaluating":
                        status_icon = "🔄"
                    elif phase == "dataset_loading":
                        status_icon = "📥"
                    else:
                        status_icon = "⏳"
                    health_indicator = ""
                    if health == "healthy":
                        health_indicator = " [green]✓[/green]"
                    elif health == "unhealthy":
                        health_indicator = " [red]✗[/red]"
                    elif health == "starting":
                        health_indicator = " [yellow]⏳[/yellow]"
                    
                    status_text = f"{status_icon} {benchmark[:30]}... [{phase}]{health_indicator} ({elapsed_str})"
                else:
                    status_text = f"⏳ {benchmark[:30]}... (starting, {elapsed_str})"

                # Update benchmark status dict if provided
                if benchmark_status is not None:
                    benchmark_status[benchmark] = {
                        "status": phase,
                        "elapsed": elapsed_str,
                        "container": container_name or "waiting",
                        "invocation_id": invocation_id,
                    }

                progress.update(task, description=status_text)
                last_update = time.time()

            time.sleep(1)

        # Get final output
        stdout, stderr = process.communicate()
        total_time = time.time() - start_time

        # Calculate timing metrics
        timing_info = {
            "total_time": total_time,
            "container_startup": (container_start_time - start_time) if container_start_time else None,
            "evaluation_time": (time.time() - evaluation_start_time) if evaluation_start_time else None,
        }

        # Clean up process from running list
        for proc_info in running_processes[:]:
            if proc_info['process'] == process:
                running_processes.remove(proc_info)
                break

        if process.returncode == 0:
            # Extract invocation ID if not already found
            if not invocation_id:
                invocation_id = get_invocation_id_from_output(stdout)

            return True, invocation_id or "unknown", timing_info
        else:
            # Enhanced error message with context
            error_msg = stderr[:500] if stderr else stdout[:500]
            
            # Check for common error patterns and provide helpful messages
            error_lower = error_msg.lower()
            if "500" in error_msg or "internal server error" in error_lower:
                error_msg = f"500 Internal Server Error: {error_msg[:300]}\n"
                error_msg += "  [yellow]Tip: Service may be starting up. Check container health with 'nemo-monitor'[/yellow]"
            elif "connection" in error_lower and "refused" in error_lower:
                error_msg = f"Connection Refused: {error_msg[:300]}\n"
                error_msg += "  [yellow]Tip: Ensure the model endpoint is running and accessible[/yellow]"
            elif "timeout" in error_lower:
                error_msg = f"Timeout Error: {error_msg[:300]}\n"
                error_msg += "  [yellow]Tip: Increase request_timeout in eval_params.yaml or check network connectivity[/yellow]"
            
            return False, error_msg, timing_info

    except subprocess.TimeoutExpired:
        # Clean up process from running list
        for proc_info in running_processes[:]:
            if proc_info['process'] == process:
                running_processes.remove(proc_info)
                break
        return False, "Timeout after 2 hours. Consider increasing timeout limits.", {}
    except Exception as e:
        # Clean up process from running list
        for proc_info in running_processes[:]:
            if proc_info['process'] == process:
                running_processes.remove(proc_info)
                break
        # Get error context for better debugging
        error_context = get_error_context(e)
        error_msg = f"{type(e).__name__}: {str(e)}"
        if error_context.get("status_code"):
            error_msg += f" (Status: {error_context['status_code']})"
        return False, error_msg, {}


def group_benchmarks_by_container(
    benchmarks: List[str], catalog: Dict
) -> Dict[str, List[str]]:
    """Group benchmarks by their container to optimize execution."""
    groups = {}
    benchmark_to_harness = {}

    # Build mapping - include both full names and short names
    for harness_name, harness_data in catalog["harnesses"].items():
        for task in harness_data["tasks"]:
            full_name = get_benchmark_full_name(harness_name, task["name"])
            short_name = task["name"]
            # Map both full name and short name
            benchmark_to_harness[full_name] = harness_name
            benchmark_to_harness[short_name] = harness_name

    # Group benchmarks
    for benchmark in benchmarks:
        harness = benchmark_to_harness.get(benchmark)
        if harness:
            container = catalog["harnesses"][harness].get("container", "unknown")
            if container not in groups:
                groups[container] = []
            groups[container].append(benchmark)
        else:
            # Try to find by checking if it's a task name directly
            found = False
            for harness_name, harness_data in catalog["harnesses"].items():
                for task in harness_data["tasks"]:
                    if task["name"] == benchmark or benchmark.endswith(
                        "." + task["name"]
                    ):
                        container = harness_data.get("container", "unknown")
                        if container not in groups:
                            groups[container] = []
                        groups[container].append(benchmark)
                        found = True
                        break
                if found:
                    break

            if not found:
                # Unknown benchmark - use default container
                if "unknown" not in groups:
                    groups["unknown"] = []
                groups["unknown"].append(benchmark)

    return groups


def main():
    """Main execution entry point."""
    global stop_vllm_on_exit

    parser = argparse.ArgumentParser(description="Run NeMo Evaluator benchmarks")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--stop-vllm", action="store_true",
        help="Stop the vLLM server when evaluations finish or are interrupted"
    )
    args = parser.parse_args()

    # Set global flag for vLLM management
    stop_vllm_on_exit = args.stop_vllm

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    project_root = get_project_root()
    config_dir = get_config_dir()

    console.print(
        Panel.fit(
            "[bold cyan]NeMo Evaluator - Benchmark Execution Orchestrator[/bold cyan]"
        )
    )

    # Load configurations
    console.print("\n[yellow]Loading configurations...[/yellow]")
    selected_config, model_config, eval_params = load_configs(config_dir)

    benchmarks = selected_config.get("selected_benchmarks", [])
    console.print(f"[green]✓ Loaded {len(benchmarks)} benchmarks[/green]")
    console.print(f"[green]✓ Model: {model_config['model']['model_id']}[/green]")
    console.print(f"[green]✓ Endpoint: {model_config['model']['endpoint_url']}[/green]")
    
    # Perform pre-flight health checks
    if not perform_preflight_checks(model_config, console):
        console.print("[red]Pre-flight checks failed. Please fix the issues above before proceeding.[/red]")
        sys.exit(1)

    # vLLM-specific optimizations and validation
    framework = model_config["model"].get("framework", "").lower()
    if framework == "vllm":
        console.print("\n[yellow]Detecting vLLM setup...[/yellow]")
        vllm_optimizer = create_vllm_optimizer(
            model_config["model"]["endpoint_url"],
            model_config["model"]["model_id"],
            console
        )

        is_valid, message = vllm_optimizer.validate_vllm_setup()
        if is_valid:
            console.print(f"[green]✓ vLLM server is running[/green]")
            if message:
                console.print(f"[dim]  {message}[/dim]")
        else:
            console.print(f"[yellow]⚠ vLLM server not detected: {message}[/yellow]")

            # Offer to start vLLM server automatically
            if not args.yes:  # Only prompt if not in non-interactive mode
                from rich.prompt import Confirm
                start_vllm = Confirm.ask(
                    "\nWould you like to start a vLLM server automatically?",
                    default=True
                )

                if start_vllm:
                    console.print("[yellow]Starting interactive vLLM setup...[/yellow]")
                    # Run the vLLM setup
                    from nemo_evaluator_orchestrator.utils.vllm_setup import create_vllm_setup_manager

                    vllm_setup = create_vllm_setup_manager()
                    config_file = get_config_dir() / "model_config.yaml"

                    # Run interactive setup
                    config = vllm_setup.interactive_model_setup()
                    if config and vllm_setup.start_vllm_server(config):
                        vllm_setup.update_model_config(config, config_file)
                        console.print("[green]✓ vLLM server started successfully![/green]")
                        console.print("Restarting pre-flight checks...")
                        # Re-run validation
                        vllm_optimizer = create_vllm_optimizer(
                            config["endpoint_url"],
                            config["model_id"],
                            console
                        )
                        is_valid, message = vllm_optimizer.validate_vllm_setup()
                    else:
                        console.print("[red]✗ Failed to start vLLM server[/red]")
                        sys.exit(1)
                else:
                    console.print("[yellow]Continuing without vLLM server (evaluation may fail)[/yellow]")
            else:
                console.print("[red]✗ vLLM server required but not running. Run 'nemo-vllm setup' first.[/red]")
                sys.exit(1)
    else:
        vllm_optimizer = None

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_dir() / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[green]Output directory: {output_dir}[/green]")

    # Confirm execution (skip if --yes flag)
    if not args.yes:
        from rich.prompt import Confirm

        if not Confirm.ask(
            f"\nProceed with evaluation of {len(benchmarks)} benchmarks?"
        ):
            console.print("[yellow]Cancelled.[/yellow]")
            sys.exit(0)
    else:
        console.print(
            f"\n[green]Proceeding with evaluation of {len(benchmarks)} benchmarks...[/green]"
        )

    # Check for benchmarks that require external API keys and prompt user if needed
    judge_required_benchmarks = ["simple_evals.math_test_500", "simple_evals.mmath"]
    missing_api_key_benchmarks = [
        b
        for b in benchmarks
        if b in judge_required_benchmarks and not os.getenv("OPENAI_API_KEY")
    ]

    if missing_api_key_benchmarks:
        console.print(
            f"\n[yellow]⚠ Warning: The following benchmarks require an OpenAI API key for judge models:[/yellow]"
        )
        for bench in missing_api_key_benchmarks:
            console.print(f"  - {bench}")

        # Check if OPENAI_API_KEY is already set
        openai_key = os.getenv("OPENAI_API_KEY")

        if not openai_key:
            # Check if we're in non-interactive mode
            if args.yes:
                console.print(
                    f"\n[yellow]⚠ Skipping {len(missing_api_key_benchmarks)} benchmarks that require OpenAI API key (non-interactive mode)[/yellow]"
                )
                # Remove benchmarks that require OpenAI API key
                benchmarks = [b for b in benchmarks if b not in missing_api_key_benchmarks]
                console.print(f"[dim]  Remaining benchmarks: {len(benchmarks)}[/dim]")
                if not benchmarks:
                    console.print("[red]No benchmarks remaining to run. Exiting.[/red]")
                    return
            else:
                # Prompt user interactively
                from rich.prompt import Prompt

                console.print(
                    "\n[yellow]Some benchmarks require an OpenAI API key for judge models.[/yellow]"
                )
                openai_key = Prompt.ask(
                    "Enter your OpenAI API key (or press Enter to skip these benchmarks)",
                    password=True,
                    default="",
                )

        if openai_key:
            # Set as environment variable for subprocess
            os.environ["OPENAI_API_KEY"] = openai_key
            console.print("[green]✓ OpenAI API key configured[/green]\n")
            # Remove from missing list since we now have the key
            missing_api_key_benchmarks = []
        else:
            # User declined or provided empty - skip benchmarks
            console.print(
                "[yellow]Skipping benchmarks that require OpenAI API key...[/yellow]\n"
            )
            # Remove benchmarks that require API keys
            benchmarks = [b for b in benchmarks if b not in missing_api_key_benchmarks]

            if not benchmarks:
                console.print(
                    "[red]Error: All selected benchmarks require external API keys that are not configured.[/red]"
                )
                sys.exit(1)

            console.print(
                f"[green]Continuing with {len(benchmarks)} benchmark(s) that don't require external API keys.[/green]\n"
            )

    # Load catalog and group benchmarks by container AFTER filtering
    # This ensures groups only contain benchmarks that will actually run
    catalog_path = config_dir / "benchmark_catalog.yaml"
    catalog = load_benchmark_catalog(catalog_path)

    # Group benchmarks by container
    groups = group_benchmarks_by_container(benchmarks, catalog)
    console.print(f"\n[yellow]Grouped into {len(groups)} container(s)[/yellow]")

    # Setup persistent cache directory
    cache_enabled = model_config.get("cache", {}).get("enabled", True)
    cache_base_dir = Path(model_config.get("cache", {}).get("base_dir", "./cache"))
    cache_dir = None

    if cache_enabled:
        # Use absolute path for cache directory
        if not cache_base_dir.is_absolute():
            cache_dir = get_cache_dir()
        else:
            cache_dir = cache_base_dir

        cache_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓ Using persistent cache: {cache_dir}[/green]")
        console.print(
            f"[yellow]  Datasets will be cached for faster subsequent runs[/yellow]\n"
        )

    # Prepare environment variables for subprocess
    subprocess_env = {}
    if os.getenv("OPENAI_API_KEY"):
        subprocess_env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Add HuggingFace token if available
    if os.getenv("HF_TOKEN"):
        subprocess_env["HF_TOKEN"] = os.getenv("HF_TOKEN")
    elif os.getenv("HUGGINGFACE_HUB_TOKEN"):
        subprocess_env["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Add cache directory to environment if enabled
    if cache_dir:
        # Set cache directory as environment variable for benchmarks
        subprocess_env["NEMO_EVALUATOR_CACHE_DIR"] = str(cache_dir)
        hf_cache_dir = cache_dir / "huggingface"
        subprocess_env["HF_HOME"] = str(hf_cache_dir)
        subprocess_env["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")
        subprocess_env["HF_HUB_CACHE"] = str(hf_cache_dir / "hub")
        subprocess_env["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")
    else:
        # Even without explicit cache, set HuggingFace cache to default location
        # This ensures datasets are cached and reused
        default_hf_cache = Path.home() / ".cache" / "huggingface"
        subprocess_env["HF_HOME"] = str(default_hf_cache)
        subprocess_env["HF_DATASETS_CACHE"] = str(default_hf_cache / "datasets")
        subprocess_env["HF_HUB_CACHE"] = str(default_hf_cache / "hub")
        subprocess_env["TRANSFORMERS_CACHE"] = str(default_hf_cache / "transformers")

    # Configure HuggingFace timeout and retry settings
    # Increase timeout from default 10s to 60s for slow networks
    subprocess_env["HF_HUB_DOWNLOAD_TIMEOUT"] = os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    # Enable retries for HuggingFace downloads
    subprocess_env["HF_HUB_NUM_DOWNLOAD_RETRIES"] = os.getenv("HF_HUB_NUM_DOWNLOAD_RETRIES", "3")
    # Set connection timeout
    subprocess_env["HF_HUB_CONNECTION_TIMEOUT"] = os.getenv("HF_HUB_CONNECTION_TIMEOUT", "30")

    # Dataset status check (no preloading for now)
    console.print("\n[yellow]Checking dataset cache status...[/yellow]")
    dataset_manager = create_dataset_manager(cache_dir, console)

    # Show dataset cache status
    dataset_table = dataset_manager.get_dataset_info_table(benchmarks)
    console.print(dataset_table)

    uncached_count = sum(1 for benchmark in benchmarks
                        if not dataset_manager.check_dataset_cache(benchmark)["cached"])

    if uncached_count > 0:
        console.print(f"\n[yellow]📥 {uncached_count} datasets will be downloaded during evaluation[/yellow]")
        console.print("[dim]   (This is normal for the first run - datasets will be cached for future runs)[/dim]")
    else:
        console.print("\n[green]✓ All datasets are already cached![/green]")

    # Show monitoring tip before execution starts
    console.print(
        f"[yellow]💡 Tip: Run 'nemo-monitor' in another terminal for real-time monitoring[/yellow]\n"
    )

    # Execute benchmarks
    results = []
    total = len(benchmarks)
    benchmark_status = {}  # Track status of all benchmarks
    all_timing_info = []

    # Show initial benchmark list
    console.print("\n[bold]Benchmarks to evaluate:[/bold]")
    for i, benchmark in enumerate(benchmarks, 1):
        console.print(f"  {i}. {benchmark}")
    console.print(f"\n[dim]Parallelism: {eval_params['global'].get('parallelism', 1)}[/dim]")
    console.print(f"[dim]Limit samples: {eval_params['global'].get('limit_samples', 'full')}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Running benchmarks...", total=total)

        for container, container_benchmarks in groups.items():
            console.print(
                f"\n[bold]Container: {container}[/bold] ({len(container_benchmarks)} benchmarks)"
            )

            for benchmark in container_benchmarks:
                # Initialize status
                benchmark_status[benchmark] = {
                    "status": "pending",
                    "elapsed": "0s",
                    "container": "waiting",
                }

                # Generate config with cache directory and vLLM optimizations
                config_file = generate_eval_config(
                    benchmark, model_config, eval_params, output_dir, cache_dir, console, vllm_optimizer
                )

                # Run benchmark with progress monitoring
                success, result, timing_info = run_benchmark(
                    config_file, benchmark, progress, task, subprocess_env, benchmark_status
                )

                all_timing_info.append(timing_info)

                results.append(
                    {
                        "benchmark": benchmark,
                        "success": success,
                        "result": result,
                        "config_file": str(config_file),
                        "timing": timing_info,
                    }
                )

                if success:
                    timing_str = ""
                    if timing_info.get("total_time"):
                        total_sec = timing_info["total_time"]
                        timing_str = f" ({int(total_sec // 60)}m {int(total_sec % 60)}s)"
                    console.print(f"  [green]✓ {benchmark}[/green] - ID: {result[:8]}{timing_str}")
                else:
                    console.print(f"  [red]✗ {benchmark}[/red] - {result[:100]}")

                progress.advance(task)

                # Small delay between benchmarks
                time.sleep(2)

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Execution Summary[/bold]")
    console.print("=" * 60)

    successful = sum(1 for r in results if r["success"])
    failed = total - successful

    console.print(f"Total benchmarks: {total}")
    console.print(f"[green]Successful: {successful}[/green]")
    console.print(f"[red]Failed: {failed}[/red]")

    # Performance summary
    if all_timing_info:
        total_times = [t.get("total_time", 0) for t in all_timing_info if t.get("total_time")]
        if total_times:
            avg_time = sum(total_times) / len(total_times)
            total_eval_time = sum(total_times)
            console.print(f"\n[bold]Performance Metrics:[/bold]")
            console.print(f"  Average time per benchmark: {int(avg_time // 60)}m {int(avg_time % 60)}s")
            console.print(f"  Total evaluation time: {int(total_eval_time // 60)}m {int(total_eval_time % 60)}s")
            
            # Show container startup overhead
            startup_times = [t.get("container_startup", 0) for t in all_timing_info if t.get("container_startup")]
            if startup_times:
                avg_startup = sum(startup_times) / len(startup_times)
                console.print(f"  Average container startup: {int(avg_startup)}s")

    # Save results summary
    summary_file = output_dir / "execution_summary.yaml"
    with open(summary_file, "w") as f:
        yaml.dump(
            {
                "timestamp": timestamp,
                "total": total,
                "successful": successful,
                "failed": failed,
                "results": results,
            },
            f,
            default_flow_style=False,
        )

    # Stop vLLM server if requested
    if stop_vllm_on_exit:
        console.print("\n[yellow]Stopping vLLM server...[/yellow]")
        stop_vllm_server()

    console.print(f"\n[green]✓ Summary saved to {summary_file}[/green]")

    # Check MLflow status and provide appropriate message
    try:
        import requests
        response = requests.get("http://127.0.0.1:5000/health", timeout=2)
        if response.status_code == 200:
            console.print(f"\n[bold]📊 View results in MLflow:[/bold] http://127.0.0.1:5000")
            console.print(f"[dim]💡 MLflow experiment: 'llm-evaluation'[/dim]")
        else:
            console.print(f"\n[yellow]⚠️  MLflow server not responding at http://127.0.0.1:5000[/yellow]")
            console.print(f"[dim]💡 Start MLflow with: nemo-mlflow start[/dim]")
    except:
        console.print(f"\n[yellow]⚠️  MLflow server not running[/yellow]")
        console.print(f"[dim]💡 Start MLflow with: nemo-mlflow start[/dim]")

    # Log sample Q&A to MLflow if available
    log_sample_qa_to_mlflow(output_dir, model_config)

    console.print(f"\n[yellow]💡 Tip: Run 'nemo-watch' for continuous monitoring[/yellow]")


def log_sample_qa_to_mlflow(output_dir: Path, model_config: Dict):
    """Log sample questions and answers from evaluation results to MLflow."""
    try:
        import mlflow
        import json
        from pathlib import Path

        # Check if MLflow is running
        try:
            import requests
            response = requests.get("http://127.0.0.1:5000/health", timeout=2)
            if response.status_code != 200:
                return  # MLflow not available
        except:
            return  # MLflow not available

        # Set MLflow tracking
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        experiment_name = "llm-evaluation"

        # Get or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception:
            return  # Can't access MLflow

        # Find report.json files
        report_files = list(output_dir.glob("**/*report.json"))
        if not report_files:
            return

        # Extract sample Q&A from each report
        samples = []
        for report_file in report_files[:3]:  # Limit to 3 reports for brevity
            try:
                with open(report_file, 'r') as f:
                    data = json.load(f)

                # Handle both single object and array formats
                if isinstance(data, dict):
                    entries = [data]
                elif isinstance(data, list):
                    entries = data[:5]  # Take first 5 samples per report
                else:
                    continue

                for entry in entries:
                    if (entry.get('request_data') and
                        entry.get('response_data') and
                        entry['response_data'].get('choices')):

                        question = entry['request_data']['messages'][0]['content'] if entry['request_data'].get('messages') else "N/A"
                        answer = entry['response_data']['choices'][0]['message']['content'] if entry['response_data']['choices'][0].get('message') else "N/A"

                        # Truncate long content for display
                        question = question[:500] + "..." if len(question) > 500 else question
                        answer = answer[:500] + "..." if len(answer) > 500 else answer

                        benchmark_name = report_file.parent.parent.name.split('/')[-1].split('\\')[-1]

                        samples.append({
                            'benchmark': benchmark_name,
                            'question': question,
                            'answer': answer,
                            'model': model_config['model']['model_id'],
                            'timestamp': entry.get('timestamp', 'N/A')
                        })

            except Exception as e:
                console.print(f"[dim]Warning: Could not parse {report_file}: {e}[/dim]")
                continue

        if not samples:
            return

        # Log samples to MLflow
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"qa-samples-{len(samples)}"):
            # Log as structured data
            mlflow.log_param("total_qa_samples", len(samples))
            mlflow.log_param("model", model_config['model']['model_id'])

            # Create a formatted text artifact with samples
            qa_content = "# Sample Questions & Answers\n\n"
            qa_content += f"**Model:** {model_config['model']['model_id']}\n"
            qa_content += f"**Total Samples:** {len(samples)}\n\n"

            for i, sample in enumerate(samples, 1):
                qa_content += f"## Sample {i}: {sample['benchmark']}\n\n"
                qa_content += f"**Question:**\n{sample['question']}\n\n"
                qa_content += f"**Answer:**\n{sample['answer']}\n\n"
                qa_content += "---\n\n"

            # Save as artifact
            qa_file = output_dir / "qa_samples.md"
            with open(qa_file, 'w') as f:
                f.write(qa_content)

            mlflow.log_artifact(str(qa_file), "qa_samples")

            console.print(f"[green]✓ Logged {len(samples)} Q&A samples to MLflow[/green]")
            console.print(f"[dim]📊 View samples in MLflow experiment artifacts[/dim]")

    except Exception as e:
        console.print(f"[dim]Warning: Could not log Q&A samples to MLflow: {e}[/dim]")


if __name__ == "__main__":
    main()
