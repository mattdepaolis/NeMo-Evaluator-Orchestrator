"""
Utility functions for managing project paths.
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Get the project root directory (where the repository is located).

    Returns:
        Path to the project root directory
    """
    # This file is in src/nemo_evaluator/utils/
    # Project root is 3 levels up
    return Path(__file__).parent.parent.parent.parent


def get_config_dir() -> Path:
    """
    Get the configuration directory path.

    Returns:
        Path to the config directory
    """
    return get_project_root() / "config"


def get_output_dir() -> Path:
    """
    Get the output directory path.

    Returns:
        Path to the output directory
    """
    return get_project_root() / "output"


def get_cache_dir() -> Path:
    """
    Get the cache directory path.

    Returns:
        Path to the cache directory
    """
    return get_project_root() / "cache"
