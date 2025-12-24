#!/usr/bin/env python3
"""
Generate benchmark catalog from available NeMo Evaluator tasks.
"""
import sys
from pathlib import Path

from nemo_evaluator_orchestrator.utils.benchmark_utils import parse_benchmark_list, save_benchmark_catalog
from nemo_evaluator_orchestrator.utils.paths import get_config_dir


def main():
    """Main entry point."""
    print("Parsing available benchmarks...")
    benchmarks = parse_benchmark_list()
    
    if not benchmarks:
        print("Error: Could not parse benchmarks. Make sure nemo-evaluator-launcher is installed.")
        sys.exit(1)
    
    catalog_path = get_config_dir() / "benchmark_catalog.yaml"
    save_benchmark_catalog(benchmarks, catalog_path)
    
    print(f"\n✓ Benchmark catalog generated successfully!")
    print(f"  Location: {catalog_path}")


if __name__ == "__main__":
    main()

