"""
Utility functions for retry logic with exponential backoff and error classification.
"""
import time
import random
import logging
from typing import Callable, Optional, Type, Tuple, Any
from functools import wraps

logger = logging.getLogger(__name__)


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.
    
    Retryable errors: 5xx server errors, connection errors, timeouts
    Non-retryable errors: 4xx client errors (400-499), model not found errors
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # First, check for non-retryable 4xx client errors
    # These indicate permanent issues like model not found, authentication failures, etc.
    
    # Check for 4xx status codes in error message
    if "404" in error_str or "not found" in error_str:
        # Check if it's a model not found error (non-retryable)
        if "model" in error_str and ("does not exist" in error_str or "notfounderror" in error_str):
            return False
        # Generic 404 is also non-retryable
        if "404" in error_str:
            return False
    
    # Check for other 4xx errors
    for code in ["400", "401", "403", "405", "406", "407", "408", "409", "410", "411", "412", "413", "414", "415", "416", "417", "418", "421", "422", "423", "424", "425", "426", "428", "429", "431", "451"]:
        if code in error_str:
            return False
    
    # Check for "bad request", "unauthorized", "forbidden" - all 4xx errors
    if "bad request" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
        return False
    
    # Extract status code from error attributes if available
    status_code = None
    if hasattr(error, "status"):
        status_code = error.status
    elif hasattr(error, "code"):
        status_code = error.code
    elif hasattr(error, "status_code"):
        status_code = error.status_code
    
    # Explicitly exclude 4xx errors (400-499)
    if status_code is not None:
        if 400 <= status_code < 500:
            return False
        # Only retry 5xx errors
        if status_code >= 500:
            return True
    
    # Check for ClientResponseError (aiohttp) - extract status code
    if "ClientResponseError" in error_type:
        if hasattr(error, "status"):
            status = error.status
            # Explicitly exclude 4xx errors
            if 400 <= status < 500:
                return False
            # Only retry 5xx errors
            if status >= 500:
                return True
    
    # Check for retryable 5xx server errors
    if "500" in error_str or "internal server error" in error_str:
        return True
    if "502" in error_str or "bad gateway" in error_str:
        return True
    if "503" in error_str or "service unavailable" in error_str:
        return True
    if "504" in error_str or "gateway timeout" in error_str:
        return True
    
    # Check for connection errors (retryable)
    if "connection" in error_str and "error" in error_str and "refused" not in error_str:
        return True
    if "timeout" in error_str and "408" not in error_str:  # 408 is 4xx, already excluded
        return True
    if "network" in error_str and "error" in error_str:
        return True
    
    # Check exception types (only retryable ones)
    retryable_types = (
        "ConnectionError",
        "TimeoutError",
        "URLError",
    )
    
    if error_type in retryable_types:
        return True
    
    # HTTPError can be retryable or not depending on status code
    # We already checked status codes above, so if we get here with HTTPError
    # and no status code was found, be conservative and don't retry
    if error_type == "HTTPError":
        return False  # Conservative: don't retry if we can't determine status
    
    return False


def calculate_backoff_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True) -> float:
    """
    Calculate exponential backoff delay with optional jitter.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        jitter: Whether to add random jitter (default: True)
        
    Returns:
        Delay in seconds
    """
    # Exponential backoff: 2^attempt * base_delay
    delay = min(base_delay * (2 ** attempt), max_delay)
    
    if jitter:
        # Add ±20% jitter to prevent thundering herd
        jitter_amount = delay * 0.2
        delay = delay + random.uniform(-jitter_amount, jitter_amount)
        delay = max(0.1, delay)  # Ensure minimum delay
    
    return delay


def retry_with_backoff(
    func: Callable,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_errors: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    log_retries: bool = True,
) -> Any:
    """
    Execute a function with exponential backoff retry logic.
    
    Args:
        func: The function to execute
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        jitter: Whether to add random jitter (default: True)
        retryable_errors: Tuple of exception types that should be retried
        on_retry: Optional callback function called on each retry (exception, attempt_number)
        log_retries: Whether to log retry attempts (default: True)
        
    Returns:
        Result of the function call
        
    Raises:
        The last exception if all retries are exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            
            # Check if error is retryable
            is_retryable = False
            
            if retryable_errors:
                is_retryable = isinstance(e, retryable_errors)
            else:
                is_retryable = is_retryable_error(e)
            
            # If not retryable or max retries reached, raise
            if not is_retryable or attempt >= max_retries:
                if log_retries and attempt > 0:
                    logger.error(
                        f"Function {func.__name__} failed after {attempt} retries: {e}"
                    )
                raise
            
            # Calculate delay
            delay = calculate_backoff_delay(attempt, base_delay, max_delay, jitter)
            
            if log_retries:
                logger.warning(
                    f"Retry attempt {attempt + 1}/{max_retries} due to: {type(e).__name__}: {e}"
                )
            
            # Call retry callback if provided
            if on_retry:
                try:
                    on_retry(e, attempt + 1)
                except Exception:
                    pass  # Don't let callback errors break retry logic
            
            # Wait before retrying
            time.sleep(delay)
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic exhausted without exception")


def retryable(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_errors: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    log_retries: bool = True,
):
    """
    Decorator for adding retry logic with exponential backoff to functions.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        jitter: Whether to add random jitter (default: True)
        retryable_errors: Tuple of exception types that should be retried
        on_retry: Optional callback function called on each retry (exception, attempt_number)
        log_retries: Whether to log retry attempts (default: True)
        
    Example:
        @retryable(max_retries=3, base_delay=2.0)
        def my_function():
            # This function will be retried on retryable errors
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            def call_func():
                return func(*args, **kwargs)
            
            return retry_with_backoff(
                call_func,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter=jitter,
                retryable_errors=retryable_errors,
                on_retry=on_retry,
                log_retries=log_retries,
            )
        
        return wrapper
    return decorator


def get_error_context(error: Exception, url: Optional[str] = None) -> dict:
    """
    Extract detailed error context for logging and debugging.
    
    Args:
        error: The exception that occurred
        url: Optional URL that was being accessed
        
    Returns:
        Dictionary with error context information
    """
    context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    
    if url:
        context["url"] = url
    
    # Extract status code if available
    if hasattr(error, "status"):
        context["status_code"] = error.status
    elif hasattr(error, "code"):
        context["status_code"] = error.code
    
    # Extract response body if available
    if hasattr(error, "response"):
        try:
            if hasattr(error.response, "text"):
                response_text = error.response.text
                # Limit response text length
                if len(response_text) > 500:
                    response_text = response_text[:500] + "..."
                context["response_body"] = response_text
        except Exception:
            pass
    
    # Extract request info if available
    if hasattr(error, "request"):
        try:
            if hasattr(error.request, "url"):
                context["request_url"] = error.request.url
            if hasattr(error.request, "method"):
                context["request_method"] = error.request.method
        except Exception:
            pass
    
    return context

