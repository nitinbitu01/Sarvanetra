"""
backend/services/circuit_breaker.py — Circuit Breaker Pattern for Graceful Service Degradation (v15.0.0)

Closes Gap DD: Graceful degradation when external services (VAHAN, CAD, ANPR) are unavailable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = auto()     # Normal operations: requests flow through
    OPEN = auto()       # Tripped: requests fail fast without attempting
    HALF_OPEN = auto()  # Probing: one request allowed to test recovery


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    reset_timeout_sec: float = 30.0

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)

    @property
    def state(self) -> CircuitState:
        if (
            self._state == CircuitState.OPEN
            and (time.monotonic() - self._last_failure_time) >= self.reset_timeout_sec
        ):
            self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, fn: Callable[[], T], fallback: T) -> T:
        s = self.state
        if s == CircuitState.OPEN:
            return fallback

        try:
            result = fn()
            if s == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
            return result
        except Exception:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
            if s == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            return fallback


# ── Pre-configured system circuit breakers ──────────────────────────────
VAHAN_BREAKER = CircuitBreaker("vahan_api", failure_threshold=3, reset_timeout_sec=60.0)
CAD_BREAKER = CircuitBreaker("cad_dispatch", failure_threshold=5, reset_timeout_sec=30.0)
ANPR_BREAKER = CircuitBreaker("anpr_model", failure_threshold=3, reset_timeout_sec=20.0)
