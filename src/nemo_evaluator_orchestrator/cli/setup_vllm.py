#!/usr/bin/env python3
"""
Interactive vLLM setup CLI tool for NeMo Evaluator Orchestrator.
Helps users configure and start vLLM servers with guided prompts.
"""
import sys
import signal
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from nemo_evaluator_orchestrator.utils.vllm_setup import create_vllm_setup_manager
from nemo_evaluator_orchestrator.utils.paths import get_config_dir

console = Console()

# Global variable to track the setup manager
setup_manager = None


def _check_vllm_endpoint_quietly(endpoint_url: str, timeout: int = 5) -> bool:
    """Quietly check if vLLM endpoint is accessible without progress messages."""
    import json
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

    try:
        # Try health endpoint first
        parsed = endpoint_url.replace("/v1/chat/completions", "")
        health_url = f"{parsed}/health" if not parsed.endswith("/") else f"{parsed}health"

        try:
            request = Request(health_url)
            with urlopen(request, timeout=timeout) as response:
                if response.getcode() == 200:
                    return True
        except (URLError, HTTPError):
            pass

        # Try model endpoint
        test_payload = {
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
        }

        data = json.dumps(test_payload).encode("utf-8")
        request = Request(endpoint_url, data=data)
        request.add_header("Content-Type", "application/json")

        with urlopen(request, timeout=timeout) as response:
            # Even if we get a 4xx error, the server is responding
            if response.getcode() < 500:
                return True

    except (URLError, HTTPError):
        pass

    return False


def signal_handler(signum, frame):
    """Handle interrupt signals to gracefully stop the server."""
    console.print("\n[yellow]Received interrupt signal...[/yellow]")
    if setup_manager:
        setup_manager.stop_server()
    console.print("[red]Setup cancelled.[/red]")
    sys.exit(1)


def setup_vllm_interactive(config_file: Optional[Path] = None) -> bool:
    """
    Interactive vLLM setup process.

    Args:
        config_file: Optional path to model config file to update

    Returns:
        True if setup completed successfully, False otherwise
    """
    global setup_manager
    setup_manager = create_vllm_setup_manager()

    try:
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Show welcome message
        console.print(Panel.fit(
            "[bold cyan]🧠 NeMo Evaluator - vLLM Setup Wizard[/bold cyan]\n"
            "[dim]This wizard will help you configure and start a vLLM server for model evaluation.\n"
            "The server will be optimized for the NeMo Evaluator workflow.[/dim]"
        ))

        # Interactive model setup
        config = setup_manager.interactive_model_setup()

        # Start vLLM server
        if setup_manager.start_vllm_server(config):
            # Update configuration file if specified
            if config_file:
                setup_manager.update_model_config(config, config_file)
            else:
                # Default config file
                default_config = get_config_dir() / "model_config.yaml"
                setup_manager.update_model_config(config, default_config)

            console.print("\n" + "="*60)
            console.print("[bold green]🎉 vLLM Setup Complete![/bold green]")
            console.print("="*60)
            console.print("Your vLLM server is now running and ready for evaluation!")
            console.print(f"[cyan]Endpoint:[/cyan] {config['endpoint_url']}")
            console.print(f"[cyan]Model:[/cyan] {config['model_id']}")

            console.print("\n[bold yellow]Next Steps:[/bold yellow]")
            console.print("1. Run evaluations: [cyan]nemo-run --yes[/cyan]")
            console.print("2. Monitor progress: [cyan]nemo-monitor[/cyan] (in another terminal)")
            console.print("3. Stop server later: [cyan]nemo-vllm stop[/cyan]")

            console.print("\n[yellow]ℹ  The server will continue running in the background.[/yellow]")
            console.print("[yellow]ℹ  You can stop it anytime with Ctrl+C or 'nemo-vllm stop'[/yellow]")

            return True
        else:
            console.print("[red]✗ vLLM setup failed. Please check the error messages above.[/red]")
            return False

    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled by user.[/yellow]")
        if setup_manager:
            setup_manager.stop_server()
        return False
    except Exception as e:
        console.print(f"[red]✗ Unexpected error during setup: {e}[/red]")
        if setup_manager:
            setup_manager.stop_server()
        return False


def stop_vllm_server():
    """Stop the running vLLM server."""
    import subprocess

    try:
        # First try to stop the server process managed by this session
        global setup_manager
        if setup_manager:
            setup_manager.stop_server()

        # Then check for any remaining vLLM processes and stop them
        result = subprocess.run(
            ['pgrep', '-f', 'vllm.entrypoints.openai.api_server'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            console.print(f"[yellow]Stopping {len(pids)} vLLM server process(es)...[/yellow]")

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
                console.print("[red]Force killing remaining vLLM processes...[/red]")
                subprocess.run(['pkill', '-9', '-f', 'vllm.entrypoints.openai.api_server'], timeout=5)

            console.print("[green]✓ Stopped all vLLM servers[/green]")
        else:
            console.print("[dim]No vLLM server processes found[/dim]")
    except Exception as e:
        console.print(f"[red]Error stopping vLLM server: {e}[/red]")


def show_server_status():
    """Show the status of the vLLM server."""
    global setup_manager

    if not setup_manager:
        setup_manager = create_vllm_setup_manager()

    # First check if we have a server process from this session
    if setup_manager.server_process and setup_manager.server_process.poll() is None:
        console.print("[green]✓ vLLM server is running (managed by this session)[/green]")
        if setup_manager.server_config:
            config = setup_manager.server_config
            console.print(f"  Model: {config.get('model_id', 'unknown')}")
            console.print(f"  Endpoint: {config.get('endpoint_url', 'unknown')}")
            console.print(f"  Port: {config.get('port', 'unknown')}")
        return

    # Check if there's a vLLM server running by checking common endpoints
    import subprocess
    import os
    from pathlib import Path

    # Load model config to get endpoint info
    config_dir = get_config_dir()
    model_config_file = config_dir / "model_config.yaml"

    server_found = False
    if model_config_file.exists():
        try:
            import yaml
            with open(model_config_file, 'r') as f:
                model_config = yaml.safe_load(f)

            if model_config and 'model' in model_config:
                endpoint_url = model_config['model'].get('endpoint_url')
                if endpoint_url:
                    # Try to check if the endpoint is accessible (quiet check)
                    if _check_vllm_endpoint_quietly(endpoint_url):
                        console.print("[green]✓ vLLM server is running[/green]")
                        model_id = model_config['model'].get('model_id', 'unknown')
                        console.print(f"  Model: {model_id}")
                        console.print(f"  Endpoint: {endpoint_url}")
                        port = endpoint_url.split(':')[-1].split('/')[0] if ':' in endpoint_url else 'unknown'
                        console.print(f"  Port: {port}")
                        server_found = True
        except Exception:
            pass

    # Check for running vLLM processes
    if not server_found:
        try:
            # Check for python processes running vllm
            result = subprocess.run(
                ['pgrep', '-f', 'vllm.entrypoints.openai.api_server'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                console.print("[green]✓ vLLM server process is running[/green]")
                console.print("  (Server started by another session or method)")
                server_found = True
        except Exception:
            pass

    if not server_found:
        console.print("[red]✗ vLLM server is not running[/red]")
        console.print("[dim]  Use 'nemo-vllm setup' to start a server[/dim]")


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive vLLM setup for NeMo Evaluator")
    parser.add_argument(
        "action",
        choices=["setup", "start", "stop", "status"],
        help="Action to perform"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to model configuration file to update"
    )

    args = parser.parse_args()

    try:
        if args.action in ["setup", "start"]:
            success = setup_vllm_interactive(args.config)
            sys.exit(0 if success else 1)
        elif args.action == "stop":
            stop_vllm_server()
        elif args.action == "status":
            show_server_status()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        stop_vllm_server()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        stop_vllm_server()
        sys.exit(1)


if __name__ == "__main__":
    main()
