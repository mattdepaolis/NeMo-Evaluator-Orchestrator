"""
Utility functions for checking Docker container health and service readiness.
"""
import subprocess
import time
import json
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


# Cache for health status to avoid excessive checks
_health_cache: Dict[str, Tuple[bool, datetime]] = {}
_cache_ttl_seconds = 5


def get_container_health_status(container_name: str, use_cache: bool = True) -> Dict[str, str]:
    """
    Get comprehensive health status for a Docker container.
    
    Args:
        container_name: Name or ID of the Docker container
        use_cache: Whether to use cached results (default: True)
        
    Returns:
        Dictionary with keys:
        - status: Container state (running, stopped, etc.)
        - health: Health status (healthy, unhealthy, starting, none)
        - uptime: Container uptime in human-readable format
        - started_at: Container start timestamp
    """
    # Check cache first
    if use_cache:
        cache_key = f"health_{container_name}"
        if cache_key in _health_cache:
            is_healthy, cached_time = _health_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=_cache_ttl_seconds):
                # Return cached status (simplified)
                return {
                    "status": "running" if is_healthy else "unknown",
                    "health": "healthy" if is_healthy else "unknown",
                    "uptime": "N/A",
                    "started_at": "N/A",
                }
    
    try:
        # Get container state and health
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .State}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode != 0:
            return {
                "status": "not_found",
                "health": "unknown",
                "uptime": "N/A",
                "started_at": "N/A",
            }
        
        state_data = json.loads(result.stdout)
        status = state_data.get("Status", "unknown")
        health = state_data.get("Health", {}).get("Status", "none")
        started_at = state_data.get("StartedAt", "")
        
        # Calculate uptime
        uptime = "N/A"
        if started_at:
            try:
                # Parse ISO format timestamp
                start_time = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                now = datetime.now(start_time.tzinfo) if start_time.tzinfo else datetime.now()
                uptime_delta = now - start_time
                
                total_seconds = int(uptime_delta.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                
                if hours > 0:
                    uptime = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    uptime = f"{minutes}m {seconds}s"
                else:
                    uptime = f"{seconds}s"
            except Exception:
                uptime = started_at[:19] if len(started_at) > 19 else started_at
        
        # Update cache
        is_healthy = health == "healthy"
        if use_cache:
            _health_cache[f"health_{container_name}"] = (is_healthy, datetime.now())
        
        return {
            "status": status,
            "health": health,
            "uptime": uptime,
            "started_at": started_at[:19] if started_at else "N/A",
        }
    except Exception as e:
        return {
            "status": "error",
            "health": "unknown",
            "uptime": "N/A",
            "started_at": "N/A",
            "error": str(e),
        }


def check_service_readiness(
    url: str, timeout: int = 5, expected_status: int = 200
) -> Tuple[bool, Optional[str]]:
    """
    Check if a service endpoint is ready by making an HTTP request.
    
    Args:
        url: The health check URL
        timeout: Request timeout in seconds
        expected_status: Expected HTTP status code (default: 200)
        
    Returns:
        Tuple of (is_ready, error_message)
        If ready, returns (True, None)
        If not ready, returns (False, error_message)
    """
    try:
        request = Request(url)
        request.add_header("User-Agent", "NeMo-Evaluator-Orchestrator/1.0")
        
        with urlopen(request, timeout=timeout) as response:
            status_code = response.getcode()
            if status_code == expected_status:
                return True, None
            else:
                return False, f"Unexpected status code: {status_code}"
    except HTTPError as e:
        return False, f"HTTP error: {e.code} {e.reason}"
    except URLError as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def get_container_port_mapping(container_name: str, internal_port: int) -> Optional[int]:
    """
    Get the host port mapped to a container's internal port.
    
    Args:
        container_name: Name or ID of the Docker container
        internal_port: Internal port number
        
    Returns:
        Host port number if found, None otherwise
    """
    try:
        result = subprocess.run(
            [
                "docker",
                "port",
                container_name,
                str(internal_port),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # Output format: "0.0.0.0:3825" or "[::]:3825"
            # Extract port number
            parts = result.stdout.strip().split(":")
            if len(parts) >= 2:
                return int(parts[-1])
    except Exception:
        pass
    
    return None


def wait_for_container_health(
    container_name: str,
    timeout: int = 60,
    check_interval: int = 2,
    target_health: str = "healthy",
) -> Tuple[bool, str]:
    """
    Wait for a container to reach a target health status.
    
    Args:
        container_name: Name or ID of the Docker container
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        target_health: Target health status (healthy, running, etc.)
        
    Returns:
        Tuple of (success, final_status)
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        health_info = get_container_health_status(container_name, use_cache=False)
        current_status = health_info.get("status", "unknown")
        current_health = health_info.get("health", "none")
        
        # Check if target is reached
        if target_health == "healthy":
            if current_health == "healthy":
                return True, "healthy"
        elif target_health == "running":
            if current_status == "running":
                return True, "running"
        
        # Check if container is in a failed state
        if current_status in ("exited", "dead"):
            return False, f"Container {current_status}"
        
        time.sleep(check_interval)
    
    # Timeout
    final_health = get_container_health_status(container_name, use_cache=False)
    return False, f"Timeout waiting for {target_health}. Current: {final_health.get('health', 'unknown')}"


def wait_for_service_ready(
    url: str,
    timeout: int = 60,
    check_interval: int = 2,
    max_retries: int = 3,
) -> Tuple[bool, Optional[str]]:
    """
    Wait for a service endpoint to become ready.
    
    Args:
        url: The health check URL
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        max_retries: Maximum number of consecutive failures before giving up
        
    Returns:
        Tuple of (is_ready, error_message)
    """
    start_time = time.time()
    consecutive_failures = 0
    
    while time.time() - start_time < timeout:
        is_ready, error = check_service_readiness(url, timeout=check_interval)
        
        if is_ready:
            return True, None
        
        consecutive_failures += 1
        if consecutive_failures >= max_retries:
            return False, f"Service not ready after {consecutive_failures} attempts: {error}"
        
        time.sleep(check_interval)
    
    return False, f"Timeout waiting for service at {url}"


def get_container_resource_usage(container_name: str) -> Optional[Dict[str, str]]:
    """
    Get resource usage statistics for a container.
    
    Args:
        container_name: Name or ID of the Docker container
        
    Returns:
        Dictionary with CPU and memory usage, or None if unavailable
    """
    try:
        result = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split("|")
            if len(parts) >= 3:
                return {
                    "cpu_percent": parts[0].strip(),
                    "memory_usage": parts[1].strip(),
                    "memory_percent": parts[2].strip(),
                }
    except Exception:
        pass
    
    return None


def clear_health_cache():
    """Clear the health status cache."""
    global _health_cache
    _health_cache.clear()

