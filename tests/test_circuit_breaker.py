"""
tests/test_circuit_breaker.py — Circuit Breaker Pattern Tests (v15.0.0).
"""
import time
import pytest

from backend.services.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)


def test_circuit_breaker_trip_and_fallback():
    breaker = CircuitBreaker("test_api", failure_threshold=3, reset_timeout_sec=0.2)

    def failing_service():
        raise ConnectionError("Service unreachable")

    # 3 failures trip the circuit
    assert breaker.call(failing_service, fallback="fallback_val") == "fallback_val"
    assert breaker.call(failing_service, fallback="fallback_val") == "fallback_val"
    assert breaker.call(failing_service, fallback="fallback_val") == "fallback_val"

    assert breaker.state == CircuitState.OPEN

    # 4th call fails fast without executing
    assert breaker.call(failing_service, fallback="fallback_fast") == "fallback_fast"

    # Wait for reset timeout -> transitions to HALF_OPEN
    time.sleep(0.25)
    assert breaker.state == CircuitState.HALF_OPEN

    # Successful call resets circuit to CLOSED
    def healthy_service():
        return "success"

    assert breaker.call(healthy_service, fallback="fallback_val") == "success"
    assert breaker.state == CircuitState.CLOSED
