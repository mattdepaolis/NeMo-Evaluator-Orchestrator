"""
Utility functions for endpoint URL handling and Docker compatibility.
"""
import platform
import json
from typing import Optional, Tuple, Dict
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def should_convert_endpoint() -> bool:
    """
    Determine if endpoint URLs should be converted for Docker compatibility.
    
    On Linux, Docker containers cannot reach 'localhost' on the host machine.
    They need to use the Docker bridge gateway IP (typically 172.17.0.1).
    
    Returns:
        True if running on Linux, False otherwise
    """
    return platform.system() == "Linux"


def convert_localhost_for_docker(url: str) -> str:
    """
    Convert localhost URLs to Docker bridge IP for container compatibility.
    
    On Linux, converts:
    - localhost -> 172.17.0.1
    - 127.0.0.1 -> 172.17.0.1
    
    On other platforms, returns the URL unchanged.
    
    Args:
        url: The endpoint URL to convert
        
    Returns:
        Converted URL if on Linux, original URL otherwise
    """
    if not should_convert_endpoint():
        return url
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Check if hostname is localhost or 127.0.0.1
    hostname = parsed.hostname
    if hostname in ("localhost", "127.0.0.1"):
        # Replace with Docker bridge gateway IP
        new_hostname = "172.17.0.1"
        
        # Reconstruct URL with new hostname
        netloc = f"{new_hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        if parsed.username or parsed.password:
            auth = ""
            if parsed.username:
                auth = parsed.username
            if parsed.password:
                auth += f":{parsed.password}"
            netloc = f"{auth}@{netloc}"
        
        new_parsed = parsed._replace(netloc=netloc)
        return urlunparse(new_parsed)
    
    return url


def validate_endpoint_url(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate an endpoint URL format.
    
    Args:
        url: The URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, returns (True, None)
        If invalid, returns (False, error_message)
    """
    if not url:
        return False, "URL is empty"
    
    try:
        parsed = urlparse(url)
        
        if not parsed.scheme:
            return False, "URL must include a scheme (http:// or https://)"
        
        if parsed.scheme not in ("http", "https"):
            return False, f"Unsupported scheme: {parsed.scheme}. Use http:// or https://"
        
        if not parsed.hostname:
            return False, "URL must include a hostname"
        
        return True, None
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


def detect_service_endpoint(container_name: str, port: int, path: str = "/health") -> Optional[str]:
    """
    Detect service endpoint from a running Docker container.
    
    Args:
        container_name: Name of the Docker container
        port: Port number the service is listening on
        path: Health check path (default: /health)
        
    Returns:
        Full endpoint URL if container is running, None otherwise
    """
    import subprocess
    
    try:
        # Check if container is running
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode != 0 or result.stdout.strip() != "true":
            return None
        
        # Get container IP or use localhost
        # For services exposed to host, use localhost
        # For internal services, we'd need to get the container IP
        # For simplicity, assume services are exposed on host
        if should_convert_endpoint():
            host = "172.17.0.1"
        else:
            host = "localhost"
        
        return f"http://{host}:{port}{path}"
    except Exception:
        return None


def validate_model_endpoint(endpoint_url: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Validate that a model endpoint is accessible and responding.
    
    Args:
        endpoint_url: The model endpoint URL (e.g., http://localhost:8000/v1/chat/completions)
        timeout: Request timeout in seconds (default: 10)
        
    Returns:
        Tuple of (is_accessible, error_message)
        If accessible, returns (True, None)
        If not accessible, returns (False, error_message)
    """
    try:
        # Create a simple test request
        test_payload = {
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
        }
        
        data = json.dumps(test_payload).encode("utf-8")
        request = Request(endpoint_url, data=data)
        request.add_header("Content-Type", "application/json")
        request.add_header("User-Agent", "NeMo-Evaluator-Orchestrator/1.0")
        
        with urlopen(request, timeout=timeout) as response:
            status_code = response.getcode()
            # Even if we get an error response, the endpoint is accessible
            # 400/404 might mean model doesn't exist, but endpoint works
            if status_code < 500:
                return True, None
            else:
                return False, f"Endpoint returned server error: {status_code}"
                
    except HTTPError as e:
        # HTTP errors mean the endpoint is accessible, but there might be an issue
        status_code = e.code
        if status_code < 500:
            # 4xx errors mean endpoint is accessible but request is invalid
            # This is actually OK for validation - endpoint is working
            return True, None
        else:
            return False, f"Endpoint returned server error: {status_code} {e.reason}"
    except URLError as e:
        return False, f"Cannot connect to endpoint: {str(e)}"
    except Exception as e:
        return False, f"Error validating endpoint: {str(e)}"


def check_model_exists(endpoint_url: str, model_id: str, timeout: int = 10, fetch_available: bool = True) -> Tuple[bool, Optional[str], Optional[list]]:
    """
    Check if a specific model exists at the endpoint.
    
    Args:
        endpoint_url: The model endpoint URL (e.g., http://localhost:8000/v1/chat/completions)
        model_id: The model ID to check (e.g., "meta/llama-3.2-3b-instruct")
        timeout: Request timeout in seconds (default: 10)
        fetch_available: Whether to fetch available models if model not found (default: True)
        
    Returns:
        Tuple of (model_exists, error_message, available_models)
        If model exists, returns (True, None, None)
        If model doesn't exist, returns (False, error_message, available_models_list)
    """
    try:
        # Create a test request with the specific model
        test_payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
        }
        
        data = json.dumps(test_payload).encode("utf-8")
        request = Request(endpoint_url, data=data)
        request.add_header("Content-Type", "application/json")
        request.add_header("User-Agent", "NeMo-Evaluator-Orchestrator/1.0")
        
        with urlopen(request, timeout=timeout) as response:
            # If we get a 200 response, model exists
            status_code = response.getcode()
            if status_code == 200:
                return True, None, None
            else:
                return False, f"Unexpected status code: {status_code}", None
                
    except HTTPError as e:
        status_code = e.code
        error_body = None
        
        # Try to read error response body
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            pass
        
        # 404 means model not found
        if status_code == 404:
            error_msg = f"Model '{model_id}' not found at endpoint"
            if error_body:
                try:
                    error_json = json.loads(error_body)
                    if "error" in error_json and "message" in error_json["error"]:
                        error_msg = error_json["error"]["message"]
                except Exception:
                    pass
            
            # Try to fetch available models to suggest alternatives
            available_models = None
            if fetch_available:
                try:
                    success, models, _ = get_available_models(endpoint_url, timeout=timeout)
                    if success and models:
                        available_models = models
                except Exception:
                    pass  # Non-critical, just continue without suggestions
            
            return False, error_msg, available_models
        
        # Other 4xx errors
        if 400 <= status_code < 500:
            error_msg = f"Client error {status_code}: {e.reason}"
            if error_body:
                try:
                    error_json = json.loads(error_body)
                    if "error" in error_json and "message" in error_json["error"]:
                        error_msg = error_json["error"]["message"]
                except Exception:
                    pass
            return False, error_msg, None
        
        # 5xx errors - server error, but endpoint is accessible
        return False, f"Server error {status_code}: {e.reason}", None
        
    except URLError as e:
        return False, f"Cannot connect to endpoint: {str(e)}", None
    except Exception as e:
        return False, f"Error checking model: {str(e)}", None


def get_available_models(endpoint_url: str, timeout: int = 10) -> Tuple[bool, Optional[list], Optional[str]]:
    """
    Attempt to get list of available models from the endpoint.
    
    Note: Not all endpoints support model listing. This is a best-effort attempt.
    
    Args:
        endpoint_url: The base endpoint URL
        timeout: Request timeout in seconds (default: 10)
        
    Returns:
        Tuple of (success, models_list, error_message)
        If successful, returns (True, list_of_models, None)
        If failed, returns (False, None, error_message)
    """
    try:
        # Try to get models from /v1/models endpoint (OpenAI-compatible)
        parsed = urlparse(endpoint_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        models_url = f"{base_url}/v1/models"
        
        request = Request(models_url)
        request.add_header("User-Agent", "NeMo-Evaluator-Orchestrator/1.0")
        
        with urlopen(request, timeout=timeout) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode("utf-8"))
                if "data" in data:
                    models = [model.get("id") for model in data["data"] if "id" in model]
                    return True, models, None
                return False, None, "Unexpected response format"
            else:
                return False, None, f"Endpoint returned status {response.getcode()}"
                
    except HTTPError as e:
        # 404 means endpoint doesn't support model listing - this is OK
        if e.code == 404:
            return False, None, "Model listing not supported by endpoint"
        return False, None, f"HTTP error {e.code}: {e.reason}"
    except URLError as e:
        return False, None, f"Cannot connect: {str(e)}"
    except Exception as e:
        return False, None, f"Error getting models: {str(e)}"
