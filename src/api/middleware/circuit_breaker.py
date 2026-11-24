"""
Circuit Breaker pattern implementation.

Prevents cascading failures by detecting repeated failures and
temporarily disabling calls to the failing service.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject calls
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit is open and calls are rejected."""
    pass


class CircuitBreaker:
    """
    State-machine circuit breaker.

    - CLOSED → OPEN: after `failure_threshold` consecutive failures
    - OPEN → HALF_OPEN: after `recovery_timeout` seconds
    - HALF_OPEN → CLOSED: on first success
    - HALF_OPEN → OPEN: on first failure
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        success_threshold: int = 2,
        name: str = "default",
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._total_calls = 0
        self._total_failures = 0

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for automatic OPEN → HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit '%s' transitioning to HALF_OPEN", self.name)
        return self._state

    def _on_success(self) -> None:
        """Handle successful call."""
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                logger.info("Circuit '%s' recovered → CLOSED", self.name)

    def _on_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1
        self._total_failures += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit '%s' failed during HALF_OPEN → OPEN", self.name)
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit '%s' opened after %d failures",
                self.name,
                self._failure_count,
            )

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function with circuit breaker protection.

        Raises CircuitBreakerError if the circuit is open.
        """
        current_state = self.state
        if current_state == CircuitState.OPEN:
            logger.warning("Circuit '%s' is OPEN, rejecting call", self.name)
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Retry after {self.recovery_timeout}s."
            )

        self._total_calls += 1
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    @property
    def stats(self) -> dict:
        """Return circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "failure_rate": self._total_failures / max(self._total_calls, 1),
        }

    def record_success(self) -> None:
        """Report a success for callers that cannot use ``call()``.

        A streaming response is consumed lazily, so it cannot be wrapped by
        ``call()`` — those callers report their outcome directly.
        """
        self._on_success()

    def record_failure(self) -> None:
        """Report a failure for callers that cannot use ``call()``."""
        self._on_failure()

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info("Circuit '%s' manually reset to CLOSED", self.name)