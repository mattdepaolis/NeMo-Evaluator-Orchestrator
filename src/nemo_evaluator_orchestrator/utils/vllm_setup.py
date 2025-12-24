"""
vLLM setup utilities for NeMo Evaluator Orchestrator.
Provides interactive setup and automatic vLLM server management.
"""
import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    import dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

from .vllm_utils import create_vllm_optimizer

console = Console()


class VLLMSetupManager:
    """Manages interactive vLLM setup and server lifecycle."""

    def __init__(self):
        self.console = console
        self.server_process = None
        self.server_config = {}

    def interactive_model_setup(self) -> Dict[str, any]:
        """
        Interactive setup for vLLM model configuration.

        Returns:
            Dictionary containing model configuration
        """
        console.print("\n[bold cyan]🚀 vLLM Model Setup[/bold cyan]")
        console.print("Let's configure your vLLM server for evaluation.\n")
        
        # Load environment variables from .env file if it exists
        env_file = Path.cwd() / ".env"
        if env_file.exists():
            if HAS_DOTENV:
                console.print("✓ Loading environment variables from .env file")
                dotenv.load_dotenv(env_file)
                console.print("[dim]  Environment variables loaded[/dim]")
            else:
                console.print("[yellow]⚠ python-dotenv not installed, skipping .env loading[/yellow]")
                console.print("[dim]  Install with: pip install python-dotenv[/dim]")
        else:
            console.print("[dim]No .env file found, using existing environment variables[/dim]")
        
        console.print()
        
        # Model selection
        console.print("[yellow]Step 1: Choose a Model[/yellow]")
        console.print("Popular options (⚠ = may require HuggingFace authentication):")
        models_table = Table(show_header=True, header_style="bold blue")
        models_table.add_column("Model", style="cyan", width=35)
        models_table.add_column("Size", style="green", width=8)
        models_table.add_column("Use Case", style="yellow", width=25)
        
        popular_models = [
            # Models that don't require authentication
            ("microsoft/DialoGPT-medium", "117M", "Conversational AI (no auth needed)"),
            ("distilbert-base-uncased", "66M", "Lightweight BERT (no auth needed)"),
            ("google/flan-t5-base", "250M", "Text-to-text (no auth needed)"),
            ("facebook/opt-1.3b", "1.3B", "General purpose (no auth needed)"),
            # Models that may require authentication
            ("meta-llama/Llama-3.2-3B-Instruct", "3B", "General purpose, fast ⚠"),
            ("meta-llama/Llama-3.2-7B-Instruct", "7B", "Balanced performance ⚠"),
            ("meta-llama/Llama-3.1-8B-Instruct", "8B", "High performance ⚠"),
            ("mistralai/Mistral-7B-Instruct-v0.1", "7B", "Fast, efficient ⚠"),
        ]
        
        for model, size, use_case in popular_models:
            models_table.add_row(model, size, use_case)
        
        console.print(models_table)
        
        # Model input with validation
        while True:
            model_id = Prompt.ask("\nEnter HuggingFace model ID").strip()
        
            if not model_id:
                console.print("[red]Model ID cannot be empty. Please try again.[/red]")
                continue
        
            # Basic validation - check if it looks like a valid HF model ID
            if "/" not in model_id:
                console.print("[yellow]Model ID should be in format 'organization/model-name'[/yellow]")
                console.print("[dim]Example: meta-llama/Llama-3.2-3B-Instruct[/dim]")
                if not Confirm.ask("Continue anyway?"):
                    continue
        
            # Check for potential authentication requirements
            # Only specific gated models require authentication
            gated_models = [
                'meta-llama', 'mistralai',  # Most models from these orgs are gated
                'microsoft/wizardlm', 'microsoft/wizard',  # WizardLM series
            ]
            requires_auth = any(pattern in model_id.lower() for pattern in gated_models)
        
            if requires_auth:
                console.print(f"[yellow]⚠ Model '{model_id}' requires HuggingFace authentication.[/yellow]")
                has_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
                if has_token:
                    console.print("[green]✓ HuggingFace token found in environment![/green]")
                    console.print(f"[dim]  Token: {has_token[:8]}...{has_token[-4:] if len(has_token) > 12 else has_token}[/dim]")
                else:
                    console.print("[red]No HuggingFace token found in environment variables.[/red]")
                    console.print("")

                    # Offer interactive token input
                    if Confirm.ask("Would you like to enter your HuggingFace token now?", default=False):
                        token = Prompt.ask("Enter your HuggingFace token", password=True)

                        if token.strip():
                            # Set environment variable
                            os.environ['HF_TOKEN'] = token
                            console.print("[green]✓ Token set in environment![/green]")

                            # Optionally save to .env file
                            if Confirm.ask("Save token to .env file for future use?", default=True):
                                try:
                                    env_file = Path(".env")
                                    existing_content = ""
                                    if env_file.exists():
                                        existing_content = env_file.read_text()

                                    # Remove existing HF_TOKEN lines
                                    lines = existing_content.split('\n')
                                    lines = [line for line in lines if not line.startswith('HF_TOKEN=')]

                                    # Add new token
                                    lines.append(f"HF_TOKEN={token}")

                                    # Write back
                                    env_file.write_text('\n'.join(lines) + '\n')
                                    console.print("[green]✓ Token saved to .env file![/green]")
                                except Exception as e:
                                    console.print(f"[yellow]⚠ Could not save to .env file: {e}[/yellow]")
                                    console.print("[dim]Token is still set in environment for this session.[/dim]")

                            # Try to login with huggingface-cli
                            try:
                                console.print("[yellow]Logging in with HuggingFace...[/yellow]")
                                result = subprocess.run(
                                    ["huggingface-cli", "login", "--token", token],
                                    capture_output=True, text=True, timeout=30
                                )
                                if result.returncode == 0:
                                    console.print("[green]✓ Successfully logged in to HuggingFace![/green]")
                                else:
                                    console.print(f"[yellow]⚠ HuggingFace login may have issues: {result.stderr}[/yellow]")
                            except Exception as e:
                                console.print(f"[yellow]⚠ Could not run huggingface-cli login: {e}[/yellow]")
                                console.print("[dim]You may need to run 'huggingface-cli login' manually.[/dim]")
                        else:
                            console.print("[yellow]No token entered. Continuing without authentication...[/yellow]")
                    else:
                        console.print("You can set up authentication later by:")
                        console.print("  1. Running: [cyan]huggingface-cli login[/cyan]")
                        console.print("  2. Setting: [cyan]export HF_TOKEN=your_token_here[/cyan]")
                        console.print("  3. Adding to .env file: [cyan]HF_TOKEN=your_token_here[/cyan]")
                        console.print("")
                        console.print("[yellow]Try one of these models that don't require authentication:[/yellow]")
                        console.print("[cyan]• microsoft/DialoGPT-medium (conversational AI)[/cyan]")
                        console.print("[cyan]• distilbert-base-uncased (lightweight BERT)[/cyan]")
                        console.print("[cyan]• google/flan-t5-base (text-to-text)[/cyan]")
                        console.print("[cyan]• facebook/opt-1.3b (general purpose)[/cyan]")
                        console.print("")
                        if not Confirm.ask("Continue with this model anyway?"):
                            continue

            break

        # GPU configuration
        console.print("\n[yellow]Step 2: GPU Configuration[/yellow]")

        # Check available GPUs
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                gpus = result.stdout.strip().split('\n')
                console.print(f"[green]✓ Found {len(gpus)} GPU(s):[/green]")
                for i, gpu in enumerate(gpus):
                    name, memory = gpu.split(', ')
                    console.print(f"  GPU {i}: {name.strip()} ({memory.strip()} MB)")
        except Exception:
            console.print("[yellow]⚠ Could not detect GPUs. Make sure NVIDIA drivers are installed.[/yellow]")
        
        # Tensor parallel size
        while True:
            tensor_parallel = IntPrompt.ask(
                "\nTensor parallel size (1 for single GPU, 2+ for multi-GPU)",
                default=1
            )
            if tensor_parallel >= 1:
                break
            console.print("[red]Tensor parallel size must be at least 1[/red]")
        
        # GPU memory utilization
        while True:
            gpu_memory_util = IntPrompt.ask(
                "GPU memory utilization percentage (70-95 recommended)",
                default=90
            )
            if 50 <= gpu_memory_util <= 95:
                break
            console.print("[red]GPU memory utilization must be between 50 and 95[/red]")

        # Port configuration
        while True:
            port = IntPrompt.ask(
                "\nPort for vLLM server",
                default=8000
            )
            if 1024 <= port <= 65535:
                break
            console.print("[red]Port must be between 1024 and 65535[/red]")

        # Quantization options
        console.print("\n[yellow]Step 3: Quantization (Optional)[/yellow]")
        console.print("Quantization reduces memory usage but may affect accuracy.")

        quantization_options = ["none", "awq", "gptq", "squeezellm", "marlin"]
        console.print("Options:", ", ".join(quantization_options))

        while True:
            quantization = Prompt.ask(
                "Quantization method (press Enter for none)",
                default="none"
            )
            if quantization in quantization_options:
                break
            console.print(f"[red]Please choose from: {', '.join(quantization_options)}[/red]")

        if quantization == "none":
            quantization = None

        # Host binding
        host = Prompt.ask(
            "\nHost binding (0.0.0.0 for all interfaces, 127.0.0.1 for localhost only)",
            default="0.0.0.0"
        )

        # Advanced options
        console.print("\n[yellow]Step 4: Advanced Options[/yellow]")

        while True:
            max_model_len = IntPrompt.ask(
                "Maximum model length (context window)",
                default=4096
            )
            if max_model_len >= 1024:
                break
            console.print("[red]Maximum model length must be at least 1024[/red]")
        
        # Build configuration
        config = {
            "model_id": model_id,
            "tensor_parallel_size": tensor_parallel,
            "gpu_memory_utilization": gpu_memory_util / 100.0,  # Convert to decimal
            "port": port,
            "host": host,
            "max_model_len": max_model_len,
            "quantization": quantization,
            "endpoint_url": f"http://localhost:{port}/v1/chat/completions"
        }

        # Display configuration summary
        console.print("\n[bold green]✓ Configuration Summary:[/bold green]")
        summary_table = Table(show_header=False)
        summary_table.add_column("Setting", style="cyan", width=25)
        summary_table.add_column("Value", style="yellow")

        summary_table.add_row("Model ID", config["model_id"])
        summary_table.add_row("Tensor Parallel", str(config["tensor_parallel_size"]))
        summary_table.add_row("GPU Memory Util", ".1%")
        summary_table.add_row("Port", str(config["port"]))
        summary_table.add_row("Host", config["host"])
        summary_table.add_row("Max Model Length", str(config["max_model_len"]))
        if config["quantization"]:
            summary_table.add_row("Quantization", config["quantization"])
        summary_table.add_row("Endpoint URL", config["endpoint_url"])

        console.print(summary_table)

        if Confirm.ask("\nProceed with this configuration?", default=True):
            return config
        else:
            console.print("[yellow]Setup cancelled. Please run again to retry.[/yellow]")
            sys.exit(0)

    def generate_vllm_command(self, config: Dict[str, any]) -> list:
        """
        Generate vLLM server command from configuration.

        Args:
        config: vLLM configuration dictionary

        Returns:
            List of command arguments for subprocess
        """
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", config["model_id"],
            "--port", str(config["port"]),
            "--host", config["host"],
            "--tensor-parallel-size", str(config["tensor_parallel_size"]),
            "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
            "--max-model-len", str(config["max_model_len"]),
        ]

        # Add quantization if specified
        if config.get("quantization"):
            cmd.extend(["--quantization", config["quantization"]])

        # Add trust remote code for HuggingFace models
        cmd.append("--trust-remote-code")

        return cmd

    def check_vllm_readiness(self, endpoint_url: str, timeout: int = 600) -> bool:
        """
        Check if vLLM server is ready to accept requests.

        Args:
            endpoint_url: The endpoint URL to check
            timeout: Maximum time to wait in seconds

        Returns:
            True if server is ready, False otherwise
        """
        start_time = time.time()
        console.print(f"[yellow]Waiting for vLLM server to be ready at {endpoint_url}...[/yellow]")

        with Progress() as progress:
            task = progress.add_task("Checking vLLM server readiness...", total=timeout)

            while time.time() - start_time < timeout:
                try:
                    # Try to connect to the health endpoint
                    parsed = endpoint_url.replace("/v1/chat/completions", "")
                    health_url = f"{parsed}/health" if not parsed.endswith("/") else f"{parsed}health"

                    # Try health endpoint first
                    try:
                        request = Request(health_url)
                        with urlopen(request, timeout=5) as response:
                            if response.getcode() == 200:
                                console.print("[green]✓ vLLM server is ready![/green]")
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

                    with urlopen(request, timeout=5) as response:
                        # Even if we get a 4xx error, the server is responding
                        if response.getcode() < 500:
                            console.print("[green]✓ vLLM server is ready![/green]")
                            return True

                except (URLError, HTTPError):
                    pass  # Server not ready yet

                # Update progress
                elapsed = int(time.time() - start_time)
                progress.update(task, completed=min(elapsed, timeout))

                time.sleep(2)

            progress.update(task, completed=timeout)

        console.print("[red]✗ vLLM server failed to start within timeout period[/red]")
        return False

    def start_vllm_server(self, config: Dict[str, any]) -> bool:
        """
        Start vLLM server with the given configuration.

        Args:
            config: vLLM configuration dictionary

        Returns:
            True if server started successfully, False otherwise
        """
        cmd = self.generate_vllm_command(config)

        console.print(f"\n[bold cyan]🚀 Starting vLLM Server[/bold cyan]")
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")

        # Add helpful messaging for large models
        model_id = config.get("model_id", "").lower()
        if any(keyword in model_id for keyword in ["7b", "8b", "13b", "30b", "65b", "70b"]):
            console.print("[yellow]💡 Large model detected - first-time download may take 5-15 minutes[/yellow]")
            console.print("[dim]This is normal for models over 7B parameters. Please be patient...[/dim]\n")
        elif any(keyword in model_id for keyword in ["3b", "mistral", "llama"]):
            console.print("[yellow]💡 Medium model detected - download may take 2-5 minutes[/yellow]")
            console.print("[dim]This is normal for first-time setup. Please be patient...[/dim]\n")
        else:
            console.print("[dim]Starting server...[/dim]\n")

        try:
            # Start the server in background
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )

            # Store config for later use
            self.server_config = config

            # Give the server a moment to start up and check if it's still running
            time.sleep(3)
            if self.server_process.poll() is not None:
                # Server exited early - capture error output
                stdout, stderr = self.server_process.communicate()
                console.print("[red]✗ vLLM server exited immediately[/red]")

                # Show comprehensive error information
                if stderr:
                    console.print("[red]Error output from vLLM:[/red]")
                    console.print("-" * 50)
                    # Show last 20 lines of stderr for more context
                    error_lines = stderr.strip().split('\n')
                    if len(error_lines) > 20:
                        error_lines = error_lines[-20:]
                        console.print("[dim]... (showing last 20 lines)[/dim]")
                    for line in error_lines:
                        if line.strip():
                            console.print(f"  {line}")
                    console.print("-" * 50)

                if stdout:
                    console.print("[yellow]Standard output (last 10 lines):[/yellow]")
                    output_lines = stdout.strip().split('\n')[-10:]
                    for line in output_lines:
                        if line.strip():
                            console.print(f"  {line}")

                # Check for common error patterns
                error_text = (stderr + stdout).lower()
                if "authentication" in error_text or "login" in error_text or "403" in error_text:
                    console.print("\n[yellow]💡 This looks like an authentication issue.[/yellow]")
                    console.print("  Solutions:")
                    console.print("  1. Verify your HF_TOKEN is correct: [cyan]echo $HF_TOKEN[/cyan]")
                    console.print("  2. Login with CLI: [cyan]huggingface-cli login[/cyan]")
                    console.print("  3. Check token permissions for this model")
                elif "cuda" in error_text or "gpu" in error_text:
                    console.print("\n[yellow]💡 This looks like a GPU/CUDA issue.[/yellow]")
                    console.print("  Solutions:")
                    console.print("  1. Check GPU availability: [cyan]nvidia-smi[/cyan]")
                    console.print("  2. Try reducing GPU memory utilization")
                    console.print("  3. Try a smaller model")
                elif "memory" in error_text:
                    console.print("\n[yellow]💡 This looks like a memory issue.[/yellow]")
                    console.print("  Solutions:")
                    console.print("  1. Reduce GPU memory utilization in setup")
                    console.print("  2. Try a smaller model")
                    console.print("  3. Check available GPU memory")
                elif "model" in error_text and "not found" in error_text:
                    console.print("\n[yellow]💡 Model not found or not accessible.[/yellow]")
                    console.print("  Solutions:")
                    console.print("  1. Verify the model name is correct")
                    console.print("  2. Check if you have access to this model")
                    console.print("  3. Try a different model from the list")

                return False

            # Check if server is ready
            if self.check_vllm_readiness(config["endpoint_url"]):
                console.print("[green]✓ vLLM server started successfully![/green]")
                return True
            else:
                console.print("[red]✗ vLLM server failed to become ready[/red]")

                # Check if server process is still running
                if self.server_process and self.server_process.poll() is None:
                    console.print("[yellow]Server process is still running but not responding[/yellow]")
                    console.print("This might indicate model loading issues or port conflicts.")
                else:
                    console.print("[red]Server process has exited[/red]")

                # Try to capture any error output
                try:
                    if self.server_process:
                        # Try to get remaining output
                        stdout, stderr = self.server_process.communicate(timeout=5)

                        if stderr:
                            console.print("[red]Error output from failed server:[/red]")
                            console.print("-" * 50)
                            error_lines = stderr.strip().split('\n')[-15:]  # Last 15 lines
                            for line in error_lines:
                                if line.strip():
                                    console.print(f"  {line}")
                            console.print("-" * 50)

                        if stdout:
                            console.print("[yellow]Last stdout output:[/yellow]")
                            output_lines = stdout.strip().split('\n')[-10:]  # Last 10 lines
                            for line in output_lines:
                                if line.strip():
                                    console.print(f"  {line}")
                except subprocess.TimeoutExpired:
                    console.print("[yellow]Could not capture complete error output (timeout)[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]Could not capture error output: {e}[/yellow]")

                # Provide troubleshooting tips
                console.print("\n[yellow]Troubleshooting tips:[/yellow]")
                console.print("• [bold]Try a smaller model first:[/bold] facebook/opt-1.3b, microsoft/DialoGPT-medium")
                console.print("• Check if the model requires authentication (HF_TOKEN)")
                console.print("• Verify GPU memory availability with: [cyan]nvidia-smi[/cyan]")
                console.print("• Large models (7B+) take 5-15 minutes to download on first run")
                console.print("• Check for port conflicts on the specified port")
                console.print("• Review the error output above for specific issues")
                console.print("• Try reducing GPU memory utilization if you get CUDA errors")

                self.stop_server()
                return False

        except Exception as e:
            console.print(f"[red]✗ Error starting vLLM server: {e}[/red]")
            return False

    def stop_server(self):
        """Stop the running vLLM server."""
        if self.server_process:
            console.print("[yellow]Stopping vLLM server...[/yellow]")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                console.print("[green]✓ vLLM server stopped[/green]")
            except subprocess.TimeoutExpired:
                console.print("[red]Force stopping vLLM server...[/red]")
                self.server_process.kill()
                self.server_process.wait()
            finally:
                self.server_process = None
                self.server_config = {}

    def update_model_config(self, config: Dict[str, any], config_file: Path):
        """
        Update the model configuration file with vLLM settings.

        Args:
            config: vLLM configuration
            config_file: Path to model config file
        """
        # Load existing config or create new one
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    model_config = yaml.safe_load(f) or {}
            except Exception:
                model_config = {}
        else:
            model_config = {}

        # Ensure structure exists
        if "model" not in model_config:
            model_config["model"] = {}
        if "cache" not in model_config:
            model_config["cache"] = {}

        # Update model configuration
        model_config["model"].update({
            "endpoint_url": config["endpoint_url"],
            "model_id": config["model_id"],
            "framework": "vllm",
            "metadata": {
                "model_family": config["model_id"].split("/")[0] if "/" in config["model_id"] else "unknown",
                "model_size": "unknown",  # Could be enhanced to parse from model name
                "quantization": config.get("quantization"),
                "tensor_parallel_size": config["tensor_parallel_size"],
                "max_model_len": config["max_model_len"],
            }
        })

        # Update cache configuration
        model_config["cache"].update({
            "enabled": True,
            "preload_datasets": True,
        })

        # Save updated configuration
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)

        console.print(f"[green]✓ Updated model configuration: {config_file}[/green]")


def create_vllm_setup_manager() -> VLLMSetupManager:
    """Factory function to create a VLLMSetupManager instance."""
    return VLLMSetupManager()
