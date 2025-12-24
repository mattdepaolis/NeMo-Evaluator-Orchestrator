#!/usr/bin/env python3
"""
Interactive tool for selecting benchmarks and configuring evaluation parameters.
"""
import sys
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Set, Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table
from rich.panel import Panel

from nemo_evaluator_orchestrator.utils.benchmark_utils import load_benchmark_catalog, get_benchmark_full_name
from nemo_evaluator_orchestrator.utils.paths import get_config_dir

console = Console()


def load_suites() -> Dict:
    """Load preset benchmark suites."""
    suites_path = get_config_dir() / "benchmark_suites.yaml"
    with open(suites_path, 'r') as f:
        return yaml.safe_load(f)['suites']


def display_categories(catalog: Dict) -> Dict[str, List[str]]:
    """Display benchmarks organized by category."""
    categories = {}
    for harness_name, harness_data in catalog['harnesses'].items():
        category = harness_data.get('category', 'Other')
        if category not in categories:
            categories[category] = []
        categories[category].append(harness_name)
    
    table = Table(title="Available Benchmark Categories")
    table.add_column("Category", style="cyan")
    table.add_column("Harnesses", style="green")
    table.add_column("Total Benchmarks", style="yellow")
    
    for category, harnesses in sorted(categories.items()):
        total = sum(catalog['harnesses'][h]['task_count'] for h in harnesses)
        table.add_row(category, ", ".join(harnesses[:3]) + ("..." if len(harnesses) > 3 else ""), str(total))
    
    console.print(table)
    return categories


def display_suites(suites: Dict):
    """Display available preset suites."""
    table = Table(title="Preset Benchmark Suites")
    table.add_column("Suite Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Benchmarks", style="yellow")
    table.add_column("Est. Time", style="magenta")
    
    for suite_name, suite_data in suites.items():
        if suite_name == 'custom':
            continue
        bench_count = len(suite_data.get('benchmarks', [])) if isinstance(suite_data.get('benchmarks'), list) else "All"
        table.add_row(
            suite_name,
            suite_data.get('description', ''),
            str(bench_count),
            suite_data.get('estimated_time', 'Unknown')
        )
    
    console.print(table)


def get_benchmarks_from_suite(suite_name: str, suites: Dict, catalog: Dict) -> Set[str]:
    """Get benchmark names from a preset suite."""
    if suite_name not in suites:
        return set()
    
    suite = suites[suite_name]
    benchmarks = suite.get('benchmarks', [])
    
    if benchmarks == "all":
        # Return all benchmarks
        all_benchmarks = set()
        for harness_name, harness_data in catalog['harnesses'].items():
            for task in harness_data['tasks']:
                full_name = get_benchmark_full_name(harness_name, task['name'])
                all_benchmarks.add(full_name)
        return all_benchmarks
    
    return set(benchmarks)


def get_benchmarks_from_category(category: str, catalog: Dict) -> Set[str]:
    """Get all benchmarks from a category."""
    benchmarks = set()
    for harness_name, harness_data in catalog['harnesses'].items():
        if harness_data.get('category') == category:
            for task in harness_data['tasks']:
                full_name = get_benchmark_full_name(harness_name, task['name'])
                benchmarks.add(full_name)
    return benchmarks


def select_individual_benchmarks_interactive(catalog: Dict) -> Set[str]:
    """Interactive benchmark discovery and selection."""
    selected = set()

    while True:
        console.print("\n[bold cyan]🔍 Individual Benchmark Selection[/bold cyan]")

        # Show selection options
        console.print("\n[bold]How would you like to find benchmarks?[/bold]")
        console.print("1. Browse by category")
        console.print("2. Search by name/keyword")
        console.print("3. Show popular benchmarks")
        console.print("4. Enter benchmark names directly")
        console.print("5. Done with individual selection")

        choice = Prompt.ask("\nChoose option", choices=["1", "2", "3", "4", "5"], default="1")

        if choice == "1":
            # Browse by category
            selected.update(browse_benchmarks_by_category(catalog))
        elif choice == "2":
            # Search by name/keyword
            selected.update(search_benchmarks_by_name(catalog))
        elif choice == "3":
            # Show popular benchmarks
            selected.update(select_from_popular_benchmarks(catalog))
        elif choice == "4":
            # Direct entry
            selected.update(enter_benchmark_names_directly())
        elif choice == "5":
            break

        if selected:
            console.print(f"\n[yellow]📋 Currently selected individual benchmarks: {len(selected)}[/yellow]")
            for bench in sorted(selected):
                console.print(f"  • {bench}")

    return selected


def browse_benchmarks_by_category(catalog: Dict) -> Set[str]:
    """Browse and select benchmarks by category."""
    selected = set()

    # Get categories
    categories = {}
    for harness_name, harness_data in catalog['harnesses'].items():
        category = harness_data.get('category', 'Other')
        if category not in categories:
            categories[category] = []
        categories[category].extend([(harness_name, task) for task in harness_data['tasks']])

    # Display categories
    console.print("\n[bold]Available Categories:[/bold]")
    cat_list = list(categories.keys())
    for i, cat in enumerate(cat_list, 1):
        count = len(categories[cat])
        console.print(f"{i}. {cat} ({count} benchmarks)")

    # Select category
    while True:
        cat_choice = Prompt.ask(f"\nChoose category (1-{len(cat_list)}) or 'back'", default="1")
        if cat_choice.lower() == 'back':
            return selected

        try:
            cat_idx = int(cat_choice) - 1
            if 0 <= cat_idx < len(cat_list):
                selected_category = cat_list[cat_idx]
                break
            else:
                console.print(f"[red]Please enter a number between 1 and {len(cat_list)}[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number or 'back'[/red]")

    # Show benchmarks in category
    benchmarks = categories[selected_category]
    console.print(f"\n[bold]{selected_category} Benchmarks:[/bold]")

    # Group by harness for better display
    harness_groups = {}
    for harness_name, task in benchmarks:
        if harness_name not in harness_groups:
            harness_groups[harness_name] = []
        harness_groups[harness_name].append(task)

    current_num = 1
    benchmark_map = {}

    for harness_name, tasks in harness_groups.items():
        console.print(f"\n[dim]{harness_name}:[/dim]")
        for task in tasks:
            benchmark_map[current_num] = f"{harness_name}.{task['name']}"
            desc = task.get('description', '')[:80] + '...' if len(task.get('description', '')) > 80 else task.get('description', '')
            console.print(f"  {current_num}. {task['name']} - {desc}")
            current_num += 1

    # Let user select benchmarks
    console.print(f"\n[bold]Select benchmarks from {selected_category}:[/bold]")
    console.print("Enter numbers separated by commas, or 'all' for all benchmarks")
    console.print("Example: 1,3,5 or 1-5 or all")

    while True:
        selection = Prompt.ask("Your choice", default="")
        if not selection:
            return selected

        if selection.lower() == 'all':
            selected.update(benchmark_map.values())
            console.print(f"[green]✓ Added all {len(benchmark_map)} benchmarks from {selected_category}[/green]")
            return selected

        # Parse selection
        try:
            selected_nums = parse_benchmark_selection(selection, len(benchmark_map))
            for num in selected_nums:
                if num in benchmark_map:
                    bench_name = benchmark_map[num]
                    selected.add(bench_name)
                    console.print(f"[green]✓ Added {bench_name}[/green]")
            break
        except ValueError as e:
            console.print(f"[red]{e}[/red]")

    return selected


def search_benchmarks_by_name(catalog: Dict) -> Set[str]:
    """Search for benchmarks by name or keyword."""
    selected = set()

    search_term = Prompt.ask("\nEnter search term (benchmark name, keyword, or partial match)")
    if not search_term:
        return selected

    # Search through all benchmarks
    matches = []
    search_lower = search_term.lower()

    for harness_name, harness_data in catalog['harnesses'].items():
        for task in harness_data['tasks']:
            task_name = task['name']
            full_name = f"{harness_name}.{task_name}"
            description = task.get('description', '')

            # Check if search term matches
            if (search_lower in task_name.lower() or
                search_lower in full_name.lower() or
                search_lower in description.lower() or
                search_lower in harness_name.lower()):
                matches.append((full_name, task))

    if not matches:
        console.print(f"[yellow]No benchmarks found matching '{search_term}'[/yellow]")
        return selected

    # Display matches
    console.print(f"\n[bold]Found {len(matches)} matching benchmarks:[/bold]")
    for i, (full_name, task) in enumerate(matches[:20], 1):  # Limit to 20 results
        desc = task.get('description', '')[:60] + '...' if len(task.get('description', '')) > 60 else task.get('description', '')
        console.print(f"  {i}. {full_name} - {desc}")

    if len(matches) > 20:
        console.print(f"[dim]... and {len(matches) - 20} more matches[/dim]")

    # Let user select from results
    if len(matches) <= 20:
        console.print("\nEnter numbers separated by commas, or 'all' for all matches")
        selection = Prompt.ask("Your choice", default="")
    else:
        console.print("\nEnter numbers separated by commas (showing first 20)")
        selection = Prompt.ask("Your choice", default="")

    if not selection:
        return selected

    if selection.lower() == 'all':
        selected.update(full_name for full_name, _ in matches)
        console.print(f"[green]✓ Added all {len(matches)} matching benchmarks[/green]")
    else:
        try:
            selected_nums = parse_benchmark_selection(selection, len(matches))
            for num in selected_nums:
                if 1 <= num <= len(matches):
                    full_name, _ = matches[num - 1]
                    selected.add(full_name)
                    console.print(f"[green]✓ Added {full_name}[/green]")
        except ValueError as e:
            console.print(f"[red]{e}[/red]")

    return selected


def select_from_popular_benchmarks(catalog: Dict) -> Set[str]:
    """Show and select from popular benchmark recommendations."""
    selected = set()

    console.print("\n[bold]🔥 Popular Benchmark Recommendations:[/bold]")

    # Define popular benchmarks by category
    popular_benchmarks = {
        "📚 Language Understanding": [
            ("simple_evals.mmlu", "General knowledge across 57 subjects"),
            ("simple_evals.mmlu_pro", "Advanced MMLU with harder questions"),
            ("lm-evaluation-harness.arc_challenge", "Science exam questions"),
            ("simple_evals.gpqa", "Expert-level science questions"),
        ],
        "🔢 Math & Reasoning": [
            ("simple_evals.math_test_500", "Mathematical problem solving"),
            ("simple_evals.AIME_2024", "Olympiad-level math problems"),
            ("lm-evaluation-harness.gsm8k", "Grade school math word problems"),
            ("simple_evals.mgsm", "Multilingual math reasoning"),
        ],
        "💻 Code Generation": [
            ("simple_evals.humaneval", "Python coding problems"),
            ("bigcode-evaluation-harness.mbpp", "General coding tasks"),
            ("livecodebench.codeexecution_v2", "Live coding challenges"),
        ],
        "🛡️ Safety & Ethics": [
            ("garak.garak", "Safety and jailbreak testing"),
            ("safety_eval.aegis_v2", "Content safety evaluation"),
            ("ifeval", "Instruction following evaluation"),
        ],
        "🎨 Multimodal": [
            ("vlmevalkit.mmbench", "Multimodal understanding"),
            ("vlmevalkit.mmbench_cn", "Chinese multimodal tasks"),
        ]
    }

    current_num = 1
    benchmark_map = {}

    for category, benchmarks in popular_benchmarks.items():
        console.print(f"\n[bold]{category}:[/bold]")
        for bench_name, description in benchmarks:
            benchmark_map[current_num] = bench_name
            console.print(f"  {current_num}. {bench_name} - {description}")
            current_num += 1

    console.print(f"\n[bold]Select popular benchmarks:[/bold]")
    console.print("Enter numbers separated by commas, or 'all' for all popular benchmarks")
    console.print("Example: 1,3,5 or 1-5 or all")

    selection = Prompt.ask("Your choice", default="")
    if not selection:
        return selected

    if selection.lower() == 'all':
        selected.update(benchmark_map.values())
        console.print(f"[green]✓ Added all {len(benchmark_map)} popular benchmarks[/green]")
    else:
        try:
            selected_nums = parse_benchmark_selection(selection, len(benchmark_map))
            for num in selected_nums:
                if num in benchmark_map:
                    bench_name = benchmark_map[num]
                    selected.add(bench_name)
                    console.print(f"[green]✓ Added {bench_name}[/green]")
        except ValueError as e:
            console.print(f"[red]{e}[/red]")

    return selected


def enter_benchmark_names_directly() -> Set[str]:
    """Allow direct entry of benchmark names."""
    selected = set()

    console.print("\n[bold]Enter benchmark names directly:[/bold]")
    console.print("Enter one benchmark per line, empty line to finish")
    console.print("Example: simple_evals.mmlu")
    console.print("Example: ifeval")
    console.print("Example: lm-evaluation-harness.gsm8k")

    while True:
        bench_input = Prompt.ask("Benchmark name", default="")
        if not bench_input:
            break
        bench_name = bench_input.strip()
        if bench_name:
            selected.add(bench_name)
            console.print(f"[green]✓ Added {bench_name}[/green]")

    return selected


def parse_benchmark_selection(selection: str, max_num: int) -> List[int]:
    """Parse benchmark selection string like '1,3,5' or '1-5'."""
    selected_nums = set()

    # Handle comma-separated values and ranges
    parts = selection.replace(' ', '').split(',')
    for part in parts:
        if '-' in part:
            # Handle range like "1-5"
            try:
                start, end = map(int, part.split('-'))
                if start > end:
                    start, end = end, start
                for num in range(start, end + 1):
                    if 1 <= num <= max_num:
                        selected_nums.add(num)
            except ValueError:
                raise ValueError(f"Invalid range format: {part}")
        else:
            # Handle single number
            try:
                num = int(part)
                if 1 <= num <= max_num:
                    selected_nums.add(num)
                else:
                    raise ValueError(f"Number {num} out of range (1-{max_num})")
            except ValueError:
                raise ValueError(f"Invalid number: {part}")

    return sorted(list(selected_nums))


def select_multiple_suites(suites: Dict, catalog: Dict) -> Set[str]:
    """Allow user to select multiple suites."""
    selected_benchmarks = set()

    while True:
        display_suites(suites)

        # Show currently selected suites
        if selected_benchmarks:
            current_count = len(selected_benchmarks)
            console.print(f"\n[yellow]Currently selected: {current_count} benchmarks from previous suites[/yellow]")

        suite_choice = Prompt.ask("\nEnter suite name (or 'done' to finish, 'skip' to skip)", default="quick")

        if suite_choice == "done":
            break
        elif suite_choice == "skip":
            if not selected_benchmarks:
                console.print("[dim]No suites selected[/dim]")
            break
        elif suite_choice in suites:
            suite_benchmarks = get_benchmarks_from_suite(suite_choice, suites, catalog)
            if suite_benchmarks:
                selected_benchmarks.update(suite_benchmarks)
                console.print(f"[green]✓ Added {len(suite_benchmarks)} benchmarks from suite '{suite_choice}'[/green]")
                console.print(f"[dim]Total selected: {len(selected_benchmarks)} benchmarks[/dim]")
            else:
                console.print(f"[red]No benchmarks found in suite '{suite_choice}'[/red]")
        else:
            console.print(f"[red]Suite '{suite_choice}' not found. Available suites: {', '.join(suites.keys())}[/red]")

    return selected_benchmarks


def select_benchmarks_interactive(catalog: Dict, suites: Dict) -> List[str]:
    """Interactive benchmark selection."""
    console.print(Panel.fit("[bold cyan]NeMo Evaluator Benchmark Selector[/bold cyan]"))
    
    # Display options
    console.print("\n[bold]Selection Method:[/bold]")
    console.print("1. Select one or more preset suites")
    console.print("2. Select by category")
    console.print("3. Select individual benchmarks")
    console.print("4. Combine all methods (suites + categories + individual benchmarks)")
    
    method = Prompt.ask("\nChoose selection method", choices=["1", "2", "3", "4"], default="1")
    
    selected = set()
    
    if method == "1":
        selected.update(select_multiple_suites(suites, catalog))

    if method == "4":
        # For combine methods, allow multiple suites first
        selected.update(select_multiple_suites(suites, catalog))
    
    if method in ["2", "4"]:
        categories = display_categories(catalog)
        category_choice = Prompt.ask("\nEnter category name (or 'skip' to skip)", default="skip")
        if category_choice != "skip" and category_choice in categories:
            cat_benchmarks = get_benchmarks_from_category(category_choice, catalog)
            selected.update(cat_benchmarks)
            console.print(f"[green]✓ Added {len(cat_benchmarks)} benchmarks from category '{category_choice}'[/green]")
    
    if method in ["3", "4"]:
        selected.update(select_individual_benchmarks_interactive(catalog))
    
    return sorted(list(selected))


def load_eval_params(config_dir: Path) -> Dict:
    """Load current evaluation parameters."""
    params_path = config_dir / "eval_params.yaml"
    if params_path.exists():
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def save_eval_params(params: Dict, config_dir: Path):
    """Save evaluation parameters to configuration file."""
    params_path = config_dir / "eval_params.yaml"
    with open(params_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    console.print(f"[green]✓ Evaluation parameters saved to {params_path}[/green]")


def load_model_config(config_dir: Path) -> Dict:
    """Load current model configuration."""
    model_path = config_dir / "model_config.yaml"
    if model_path.exists():
        with open(model_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def save_model_config(config: Dict, config_dir: Path):
    """Save model configuration to file."""
    model_path = config_dir / "model_config.yaml"
    with open(model_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    console.print(f"[green]✓ Model configuration saved to {model_path}[/green]")


def configure_eval_params_interactive(current_params: Dict, non_interactive: bool = False) -> Dict:
    """Interactive configuration of evaluation parameters."""
    console.print("\n[bold cyan]Evaluation Parameters Configuration[/bold cyan]")

    params = current_params.copy() if current_params else {}
    global_params = params.get('global', {})

    if non_interactive:
        console.print("Using default evaluation parameters (non-interactive mode)")
        console.print("Current settings:")
        console.print(f"  Temperature: {global_params.get('temperature', 0.0)}")
        console.print(f"  Max new tokens: {global_params.get('max_new_tokens', 2048)}")
        console.print(f"  Parallelism: {global_params.get('parallelism', 1)}")
        console.print(f"  Limit samples: {global_params.get('limit_samples', 'None (full evaluation)')}")
        console.print(f"  Request timeout: {global_params.get('request_timeout', 3600)}s")
        return params

    console.print("\nConfigure global evaluation parameters:")

    # Temperature
    current_temp = global_params.get('temperature', 0.0)
    console.print(f"Current temperature: {current_temp}")
    if Confirm.ask("Change temperature?", default=False):
        while True:
            temp = FloatPrompt.ask("Temperature (0.0 for deterministic, higher for more randomness)", default=current_temp)
            if 0.0 <= temp <= 2.0:
                global_params['temperature'] = temp
                break
            console.print("[red]Temperature must be between 0.0 and 2.0[/red]")

    # Max new tokens
    current_max_tokens = global_params.get('max_new_tokens', 2048)
    console.print(f"Current max_new_tokens: {current_max_tokens}")
    if Confirm.ask("Change max_new_tokens?", default=False):
        while True:
            max_tokens = IntPrompt.ask("Maximum new tokens to generate", default=current_max_tokens)
            if max_tokens > 0:
                global_params['max_new_tokens'] = max_tokens
                break
            console.print("[red]Max tokens must be greater than 0[/red]")

    # Parallelism
    current_parallelism = global_params.get('parallelism', 1)
    console.print(f"Current parallelism: {current_parallelism}")
    if Confirm.ask("Change parallelism?", default=False):
        while True:
            parallelism = IntPrompt.ask("Number of parallel requests", default=current_parallelism)
            if parallelism > 0:
                global_params['parallelism'] = parallelism
                break
            console.print("[red]Parallelism must be greater than 0[/red]")

    # Limit samples (for testing)
    current_limit = global_params.get('limit_samples')
    console.print(f"Current limit_samples: {current_limit} (null = full evaluation)")
    if Confirm.ask("Set limit_samples for testing?", default=False):
        limit = IntPrompt.ask("Limit samples (set to 0 for no limit)", default=current_limit or 0)
        global_params['limit_samples'] = limit if limit > 0 else None

    # Request timeout
    current_timeout = global_params.get('request_timeout', 3600)
    console.print(f"Current request_timeout: {current_timeout} seconds")
    if Confirm.ask("Change request timeout?", default=False):
        while True:
            timeout = IntPrompt.ask("Request timeout in seconds", default=current_timeout)
            if timeout > 0:
                global_params['request_timeout'] = timeout
                break
            console.print("[red]Timeout must be greater than 0[/red]")

    params['global'] = global_params
    return params


def configure_model_params_interactive(current_config: Dict, non_interactive: bool = False) -> Dict:
    """Interactive configuration of model parameters."""
    console.print("\n[bold cyan]Model Configuration[/bold cyan]")

    config = current_config.copy() if current_config else {}
    model_config = config.get('model', {})
    cache_config = config.get('cache', {})

    if non_interactive:
        console.print("Using current model configuration (non-interactive mode)")
        console.print("Current settings:")
        console.print(f"  Model: {model_config.get('model_id', 'Not set')}")
        console.print(f"  Endpoint: {model_config.get('endpoint_url', 'Not set')}")
        console.print(f"  Framework: {model_config.get('framework', 'Not set')}")
        console.print(f"  Cache enabled: {cache_config.get('enabled', False)}")
        return config

    console.print("Current model configuration:")
    console.print(f"  Model: {model_config.get('model_id', 'Not set')}")
    console.print(f"  Endpoint: {model_config.get('endpoint_url', 'Not set')}")
    console.print(f"  Framework: {model_config.get('framework', 'Not set')}")
    console.print(f"  Cache enabled: {cache_config.get('enabled', False)}")

    # Cache settings
    if Confirm.ask("\nConfigure caching?", default=False):
        cache_enabled = Confirm.ask("Enable dataset caching?", default=cache_config.get('enabled', True))
        cache_config['enabled'] = cache_enabled

        if cache_enabled:
            preload = Confirm.ask("Preload datasets?", default=cache_config.get('preload_datasets', True))
            cache_config['preload_datasets'] = preload

    config['cache'] = cache_config
    return config


def save_selection(benchmarks: List[str], output_path: Path):
    """Save selected benchmarks to configuration file."""
    config = {
        'selected_benchmarks': benchmarks,
        'benchmark_count': len(benchmarks)
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]✓ Saved {len(benchmarks)} benchmarks to {output_path}[/green]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Select benchmarks and configure evaluation parameters")
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip interactive prompts and use defaults"
    )
    parser.add_argument(
        "--configure-params", action="store_true",
        help="Configure evaluation parameters interactively"
    )
    parser.add_argument(
        "--configure-model", action="store_true",
        help="Configure model parameters interactively"
    )

    args = parser.parse_args()

    config_dir = get_config_dir()

    console.print(Panel.fit("[bold cyan]NeMo Evaluator Configuration Tool[/bold cyan]"))

    # Load current configurations
    eval_params = load_eval_params(config_dir)
    model_config = load_model_config(config_dir)

    # Load catalog and suites
    catalog_path = config_dir / "benchmark_catalog.yaml"
    if not catalog_path.exists():
        console.print(f"[red]Error: Benchmark catalog not found at {catalog_path}[/red]")
        console.print("Run 'python generate_catalog.py' first to generate the catalog.")
        sys.exit(1)

    catalog = load_benchmark_catalog(catalog_path)
    suites = load_suites()

    console.print(f"[green]Loaded catalog with {catalog['total_benchmarks']} benchmarks across {len(catalog['harnesses'])} harnesses[/green]\n")

    # Benchmark selection
    if not args.yes:
        console.print("[bold]Benchmark Selection:[/bold]")
        selected = select_benchmarks_interactive(catalog, suites)

        if not selected:
            console.print("[yellow]No benchmarks selected. Exiting.[/yellow]")
            sys.exit(0)

        # Display selection summary
        console.print(f"\n[bold]Selected {len(selected)} benchmarks:[/bold]")
        for i, bench in enumerate(selected[:20], 1):
            console.print(f"  {i}. {bench}")
        if len(selected) > 20:
            console.print(f"  ... and {len(selected) - 20} more")

        # Save benchmarks
        if Confirm.ask("\nSave benchmark selection?"):
            output_path = config_dir / "selected_benchmarks.yaml"
            save_selection(selected, output_path)
    else:
        # Load existing selection or create default
        selected_path = config_dir / "selected_benchmarks.yaml"
        if selected_path.exists():
            with open(selected_path, 'r') as f:
                selected_config = yaml.safe_load(f)
                selected = selected_config.get('selected_benchmarks', [])
                console.print(f"[green]✓ Loaded {len(selected)} existing benchmarks[/green]")
        else:
            console.print("[yellow]No existing benchmark selection found. Run without --yes to select benchmarks.[/yellow]")
            selected = []

    # Evaluation parameters configuration
    configure_eval = args.configure_params or (not args.yes and Confirm.ask("\nConfigure evaluation parameters?", default=False))
    if configure_eval:
        eval_params = configure_eval_params_interactive(eval_params, args.yes)
        if not args.yes:  # Only save if we actually configured (not in non-interactive mode)
            save_eval_params(eval_params, config_dir)
        else:
            save_eval_params(eval_params, config_dir)

    # Model configuration
    configure_model = args.configure_model or (not args.yes and Confirm.ask("\nConfigure model parameters?", default=False))
    if configure_model:
        model_config = configure_model_params_interactive(model_config, args.yes)
        if not args.yes:  # Only save if we actually configured (not in non-interactive mode)
            save_model_config(model_config, config_dir)
        else:
            save_model_config(model_config, config_dir)

    console.print(f"\n[bold green]✓ Configuration complete![/bold green]")
    console.print("Next step: Run 'nemo-run' to execute evaluations")

    if not selected and not args.yes:
        console.print("\n[yellow]Note: No benchmarks selected. Run 'nemo-select' again to choose benchmarks.[/yellow]")


if __name__ == "__main__":
    main()

