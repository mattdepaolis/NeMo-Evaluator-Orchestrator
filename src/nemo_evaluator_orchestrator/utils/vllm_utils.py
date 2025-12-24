"""
vLLM-specific utilities for NeMo Evaluator Orchestrator.
Provides optimizations and validation for vLLM endpoints.
"""
import json
import time
from typing import Dict, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from rich.console import Console

from .endpoint_utils import validate_endpoint_url, validate_model_endpoint, check_model_exists


class VLLMOptimizer:
    """Provides vLLM-specific optimizations and configurations."""

    def __init__(self, endpoint_url: str, model_id: str, console: Optional[Console] = None):
        self.endpoint_url = endpoint_url
        self.model_id = model_id
        self.console = console or Console()
        self.vllm_info = None

    def detect_vllm_capabilities(self) -> Dict[str, any]:
        """
        Detect vLLM server capabilities and configuration.

        Returns:
            Dictionary containing vLLM server information and capabilities.
        """
        capabilities = {
            "is_vllm": False,
            "version": None,
            "gpu_memory": None,
            "max_model_len": None,
            "tensor_parallel_size": None,
            "gpu_memory_utilization": None,
            "enforce_eager": None,
            "quantization": None,
            "dtype": None,
            "optimizations": []
        }

        try:
            # Try to get vLLM server info
            import urllib.parse
            parsed = urllib.parse.urlparse(self.endpoint_url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            # Check for vLLM-specific endpoints
            vllm_info_url = f"{base_url}/v1/info"
            request = Request(vllm_info_url)
            request.add_header("User-Agent", "NeMo-Evaluator-Orchestrator/1.0")

            with urlopen(request, timeout=10) as response:
                if response.getcode() == 200:
                    info = json.loads(response.read().decode("utf-8"))
                    capabilities["is_vllm"] = True
                    capabilities["version"] = info.get("version", "unknown")

                    # Extract model information
                    model_info = info.get("model_info", {})
                    capabilities["max_model_len"] = model_info.get("max_model_len")
                    capabilities["tensor_parallel_size"] = info.get("tensor_parallel_size")
                    capabilities["gpu_memory_utilization"] = info.get("gpu_memory_utilization")
                    capabilities["enforce_eager"] = info.get("enforce_eager")
                    capabilities["quantization"] = model_info.get("quantization")
                    capabilities["dtype"] = model_info.get("dtype")

                    # Calculate available GPU memory if possible
                    gpu_count = info.get("gpu_count", 1)
                    if capabilities["gpu_memory_utilization"]:
                        # This is approximate - actual available memory depends on many factors
                        capabilities["gpu_memory"] = f"~{(1.0 - capabilities['gpu_memory_utilization']) * 100:.1f}% free"

                    self.vllm_info = capabilities

        except (URLError, HTTPError, json.JSONDecodeError, KeyError):
            # Not a vLLM server or endpoint not available
            pass

        return capabilities

    def get_vllm_optimized_config(self, base_config: Dict) -> Dict:
        """
        Generate vLLM-optimized evaluation configuration.

        Args:
            base_config: Base evaluation configuration

        Returns:
            Optimized configuration for vLLM
        """
        if not self.vllm_info or not self.vllm_info["is_vllm"]:
            return base_config

        optimized_config = base_config.copy()

        # vLLM-specific optimizations
        params = optimized_config.get("evaluation", {}).get("nemo_evaluator_config", {}).get("config", {}).get("params", {})

        # Optimize parallelism based on vLLM capabilities
        if self.vllm_info.get("tensor_parallel_size"):
            # For tensor parallel models, reduce parallelism to avoid overhead
            current_parallelism = params.get("parallelism", 1)
            if current_parallelism > 2:
                params["parallelism"] = min(current_parallelism, 2)
                if self.console:
                    self.console.print(f"[yellow]⚡ vLLM optimization: Reduced parallelism to {params['parallelism']} for tensor parallel model[/yellow]")

        # Optimize batch size based on available GPU memory
        if self.vllm_info.get("gpu_memory_utilization", 0) > 0.8:
            # High memory utilization - reduce batch size
            current_parallelism = params.get("parallelism", 1)
            if current_parallelism > 1:
                params["parallelism"] = 1
                if self.console:
                    self.console.print("[yellow]⚡ vLLM optimization: Reduced parallelism due to high GPU memory utilization[/yellow]")

        # Set optimal timeout for vLLM
        if "request_timeout" not in params:
            params["request_timeout"] = 600  # 10 minutes for vLLM

        return optimized_config

    def validate_vllm_setup(self) -> Tuple[bool, str]:
        """
        Validate vLLM setup and provide recommendations.

        Returns:
            Tuple of (is_valid, message)
        """
        issues = []
        recommendations = []

        # Check if endpoint is accessible
        is_accessible, access_error = validate_model_endpoint(self.endpoint_url, timeout=15)
        if not is_accessible:
            issues.append(f"Endpoint not accessible: {access_error}")
            return False, "; ".join(issues)

        # Check if model exists
        model_exists, model_error, available_models = check_model_exists(
            self.endpoint_url, self.model_id, timeout=15, fetch_available=True
        )
        if not model_exists:
            issues.append(f"Model not found: {model_error}")
            if available_models:
                recommendations.append(f"Available models: {', '.join(available_models[:3])}")
            return False, "; ".join(issues + recommendations)

        # Detect vLLM capabilities
        capabilities = self.detect_vllm_capabilities()

        if not capabilities["is_vllm"]:
            # Not vLLM but still accessible - provide general recommendations
            recommendations.append("Endpoint accessible but vLLM-specific optimizations not available")
        else:
            # vLLM detected - check for optimizations
            if capabilities["version"]:
                recommendations.append(f"vLLM version: {capabilities['version']}")

            if capabilities["gpu_memory"]:
                recommendations.append(f"GPU memory: {capabilities['gpu_memory']}")

            if capabilities["max_model_len"]:
                recommendations.append(f"Max model length: {capabilities['max_model_len']}")

            # Performance recommendations
            if capabilities.get("gpu_memory_utilization", 0) > 0.9:
                issues.append("GPU memory utilization is very high - consider reducing parallelism")

            if capabilities.get("tensor_parallel_size", 1) > 1:
                recommendations.append("Tensor parallel model detected - using conservative parallelism settings")

        if issues:
            return False, "; ".join(issues + recommendations)
        elif recommendations:
            return True, "; ".join(recommendations)
        else:
            return True, "vLLM setup validated successfully"


def create_vllm_optimizer(endpoint_url: str, model_id: str, console: Optional[Console] = None) -> VLLMOptimizer:
    """Factory function to create a VLLMOptimizer instance."""
    return VLLMOptimizer(endpoint_url, model_id, console)
