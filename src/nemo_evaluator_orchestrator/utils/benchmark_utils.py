"""
Utility functions for managing and organizing NeMo Evaluator benchmarks.
"""
import subprocess
import yaml
import json
from typing import Dict, List, Tuple
from pathlib import Path


def parse_benchmark_list() -> Dict[str, Dict]:
    """
    Parse the output of 'nemo-evaluator-launcher ls tasks' to extract
    all available benchmarks organized by harness/container.
    
    Returns:
        Dictionary mapping harness names to benchmark information
    """
    try:
        result = subprocess.run(
            ["nemo-evaluator-launcher", "ls", "tasks"],
            capture_output=True,
            text=True,
            check=True
        )
        
        benchmarks = {}
        current_harness = None
        current_container = None
        
        for line in result.stdout.split('\n'):
            line = line.strip()
            
            # Parse harness line
            if line.startswith('harness:'):
                current_harness = line.split(':', 1)[1].strip()
                benchmarks[current_harness] = {
                    'container': None,
                    'tasks': []
                }
            
            # Parse container line
            elif line.startswith('container:'):
                if current_harness:
                    benchmarks[current_harness]['container'] = line.split(':', 1)[1].strip()
            
            # Parse task line - format: "task_name (type, full_name) - description..."
            elif line and current_harness and not line.startswith('-') and not line.startswith('=') and len(line) > 10:
                # Skip header lines
                if line.startswith('task') or line.startswith('---'):
                    continue
                
                # Extract task name (first word before space or parenthesis)
                task_name = line.split()[0] if line.split() else ""
                
                # Extract description (after dash)
                description = ""
                if ' - ' in line:
                    description = line.split(' - ', 1)[1].strip()
                elif ')' in line and '(' in line:
                    # Extract from parentheses
                    desc_start = line.find(')')
                    if desc_start > 0 and desc_start < len(line) - 1:
                        rest = line[desc_start+1:].strip()
                        if rest.startswith('-'):
                            description = rest[1:].strip()
                
                # Extract full name from parentheses if available
                full_name = task_name
                if '(' in line and ')' in line:
                    paren_start = line.find('(')
                    paren_end = line.find(')', paren_start)
                    if paren_end > paren_start:
                        content = line[paren_start+1:paren_end]
                        if ',' in content:
                            full_name = content.split(',')[1].strip()
                        else:
                            full_name = content.strip()
                
                if task_name:
                    benchmarks[current_harness]['tasks'].append({
                        'name': task_name,
                        'full_name': full_name,
                        'description': description
                    })
        
        return benchmarks
    
    except subprocess.CalledProcessError as e:
        print(f"Error running nemo-evaluator-launcher: {e}")
        return {}
    except Exception as e:
        print(f"Error parsing benchmark list: {e}")
        return {}


def categorize_benchmarks(benchmarks: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Categorize benchmarks into logical groups based on harness names.
    
    Args:
        benchmarks: Dictionary of benchmarks from parse_benchmark_list()
    
    Returns:
        Dictionary mapping categories to lists of harness names
    """
    categories = {
        'Language Models': [],
        'Code Generation': [],
        'Vision-Language': [],
        'Safety & Security': [],
        'Specialized Tools': [],
        'Efficiency': [],
        'Other': []
    }
    
    # Map harness names to categories
    category_mapping = {
        'lm-evaluation-harness': 'Language Models',
        'simple_evals': 'Language Models',
        'mtbench': 'Language Models',
        'mmath': 'Language Models',
        'helm': 'Language Models',
        'bigcode-evaluation-harness': 'Code Generation',
        'livecodebench': 'Code Generation',
        'scicode': 'Code Generation',
        'hle': 'Code Generation',
        'bfcl': 'Code Generation',
        'vlmevalkit': 'Vision-Language',
        'safety_eval': 'Safety & Security',
        'garak': 'Safety & Security',
        'ifbench': 'Specialized Tools',
        'tooltalk': 'Specialized Tools',
        'nemo_skills': 'Specialized Tools',
        'profbench': 'Specialized Tools',
        'genai_perf_eval': 'Efficiency'
    }
    
    for harness_name in benchmarks.keys():
        category = category_mapping.get(harness_name, 'Other')
        categories[category].append(harness_name)
    
    return categories


def save_benchmark_catalog(benchmarks: Dict, output_path: Path):
    """
    Save benchmark catalog to YAML file.
    
    Args:
        benchmarks: Dictionary of benchmarks
        output_path: Path to output YAML file
    """
    catalog = {
        'version': '1.0',
        'total_benchmarks': sum(len(h['tasks']) for h in benchmarks.values()),
        'harnesses': {}
    }
    
    categories = categorize_benchmarks(benchmarks)
    
    for harness_name, harness_data in benchmarks.items():
        catalog['harnesses'][harness_name] = {
            'container': harness_data['container'],
            'category': next((cat for cat, harnesses in categories.items() if harness_name in harnesses), 'Other'),
            'task_count': len(harness_data['tasks']),
            'tasks': harness_data['tasks']
        }
    
    with open(output_path, 'w') as f:
        yaml.dump(catalog, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Benchmark catalog saved to {output_path}")
    print(f"Total harnesses: {len(benchmarks)}")
    print(f"Total benchmarks: {catalog['total_benchmarks']}")


def load_benchmark_catalog(catalog_path: Path) -> Dict:
    """
    Load benchmark catalog from YAML file.
    
    Args:
        catalog_path: Path to catalog YAML file
    
    Returns:
        Dictionary containing benchmark catalog
    """
    with open(catalog_path, 'r') as f:
        return yaml.safe_load(f)


def get_benchmark_full_name(harness: str, task_name: str) -> str:
    """
    Get the full benchmark name for use in config files.
    
    Args:
        harness: Harness name (e.g., 'simple_evals')
        task_name: Task name (e.g., 'mmlu')
    
    Returns:
        Full benchmark name (e.g., 'simple_evals.mmlu')
    """
    if '.' in task_name:
        return task_name
    return f"{harness}.{task_name}"

