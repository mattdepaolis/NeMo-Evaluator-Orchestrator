"""
Entry point wrappers for CLI commands.
"""

import sys
from pathlib import Path


def select_benchmarks_main():
    """Entry point for nemo-select."""
    from nemo_evaluator_orchestrator.cli.select_benchmarks import main

    sys.exit(main() if main else 0)


def run_all_benchmarks_main():
    """Entry point for nemo-run."""
    from nemo_evaluator_orchestrator.cli.run_all_benchmarks import main

    sys.exit(main() if main else 0)


def monitor_benchmarks_main():
    """Entry point for nemo-monitor."""
    from nemo_evaluator_orchestrator.cli.monitor_benchmarks import main

    sys.exit(main() if main else 0)


def aggregate_results_main():
    """Entry point for nemo-aggregate."""
    from nemo_evaluator_orchestrator.cli.aggregate_results import main

    sys.exit(main() if main else 0)


def manage_containers_main():
    """Entry point for nemo-containers."""
    from nemo_evaluator_orchestrator.cli.manage_containers import main

    sys.exit(main() if main else 0)


def show_system_status_main():
    """Entry point for nemo-status."""
    from nemo_evaluator_orchestrator.cli.show_system_status import main

    sys.exit(main() if main else 0)


def generate_catalog_main():
    """Entry point for nemo-catalog."""
    from nemo_evaluator_orchestrator.cli.generate_catalog import main

    sys.exit(main() if main else 0)


def manage_datasets_main():
    """Entry point for nemo-datasets."""
    from nemo_evaluator_orchestrator.cli.manage_datasets import main

    sys.exit(main() if main else 0)


def setup_vllm_main():
    """Entry point for nemo-vllm."""
    from nemo_evaluator_orchestrator.cli.setup_vllm import main

    sys.exit(main() if main else 0)


def mlflow_main():
    """Entry point for nemo-mlflow."""
    import argparse
    import subprocess
    import sys
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="MLflow management for NeMo Evaluator")
    parser.add_argument("action", choices=["start", "stop", "status", "ui"], help="Action to perform")
    parser.add_argument("--port", default="5000", help="Port for MLflow server (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host for MLflow server (default: 127.0.0.1)")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.parent
    mlflow_dir = project_root / "output" / "mlflow"
    artifacts_dir = mlflow_dir / "artifacts"

    if args.action == "start":
        # Create directories
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Check if already running
        try:
            import requests
            response = requests.get(f"http://{args.host}:{args.port}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ MLflow server is already running at http://{args.host}:{args.port}")
                print(f"📊 View experiments: http://{args.host}:{args.port}")
                return
        except:
            pass

        print(f"🚀 Starting MLflow server on http://{args.host}:{args.port}...")
        print(f"📁 Backend store: {mlflow_dir}")
        print(f"📦 Artifacts: {artifacts_dir}")
        print("Press Ctrl+C to stop the server")

        try:
            subprocess.run([
                "mlflow", "server",
                "--backend-store-uri", str(mlflow_dir),
                "--default-artifact-root", str(artifacts_dir),
                "--host", args.host,
                "--port", args.port
            ])
        except KeyboardInterrupt:
            print("\n👋 MLflow server stopped")

    elif args.action == "stop":
        print("🛑 Stopping MLflow server...")
        try:
            # Find and kill MLflow processes
            result = subprocess.run(
                ["pgrep", "-f", "mlflow server"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        subprocess.run(["kill", pid.strip()])
                print("✅ MLflow server stopped")
            else:
                print("ℹ️  No MLflow server processes found")
        except Exception as e:
            print(f"❌ Error stopping MLflow: {e}")

    elif args.action == "status":
        try:
            import requests
            response = requests.get(f"http://{args.host}:{args.port}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ MLflow server is running at http://{args.host}:{args.port}")

                # Try to get experiment count
                try:
                    import mlflow
                    mlflow.set_tracking_uri(f"http://{args.host}:{args.port}")
                    experiments = mlflow.search_experiments()
                    print(f"📊 Experiments: {len(experiments)}")

                    # Show recent runs if any
                    if experiments:
                        for exp in experiments[:3]:  # Show first 3 experiments
                            runs = mlflow.search_runs(exp.experiment_id, max_results=1)
                            if not runs.empty:
                                latest_run = runs.iloc[0]
                                print(f"  • {exp.name}: {len(runs)} runs (latest: {latest_run.start_time.strftime('%Y-%m-%d %H:%M')})")
                except Exception as e:
                    print(f"⚠️  Could not fetch experiment details: {e}")

                print(f"🔗 View UI: http://{args.host}:{args.port}")
            else:
                print(f"❌ MLflow server health check failed (status: {response.status_code})")
        except requests.exceptions.RequestException:
            print(f"❌ MLflow server is not running on http://{args.host}:{args.port}")
            print("💡 Start it with: nemo-mlflow start")

    elif args.action == "ui":
        # Open browser or just show URL
        url = f"http://{args.host}:{args.port}"
        print(f"🔗 MLflow UI: {url}")
        print("📊 Open this URL in your browser to view experiments and results")

        # Try to open browser (Linux)
        try:
            subprocess.run(["xdg-open", url], check=False)
        except:
            pass  # Browser open failed, just show URL
