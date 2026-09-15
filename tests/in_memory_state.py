"""
tests/in_memory_state.py — dict-backed stand-in for BehaviorState.

backend/services/state.py is explicitly designed for this ("Detector classes
never import `redis` directly or hold a client — every access goes through
this module, so the backend is swappable (e.g. for a test fake...)"). This is
that fake.

WHAT IT IS NOT
──────────────
It is NOT a substitute for the eval harnesses' own real-Redis requirement.
Both harnesses deliberately hard-refuse to run without a real Redis, because
a fake cannot catch a genuine Redis-semantics mistake — a wrong
ZREMRANGEBYSCORE bound, a pipeline that isn't actually atomic, a key that
outlives its TTL. Those bugs would sail straight through this file.

What running the clips against this fake DOES establish is narrower and still
worth having: that the detection math, the alert-firing paths, and anything
newly hooked into them still behave as the labeled clips expect. Use it to
catch a regression, not to claim the Day 8/9 acceptance criteria were re-met.

TTL semantics: TTLs are recorded but never expire entries. That matches what
real Redis does over an eval run, which completes in well under a second of
wall-clock while the TTLs in play are 10-120 seconds — so nothing would have
expired there either. It is NOT a correct model of TTLs over longer runs.
"""
from __future__ import annotations

import json
from typing import Any


def _parse_bound(raw: float | str) -> tuple[float, bool]:
    """Parse a Redis score bound. Returns (value, exclusive).

    Redis spells an exclusive bound with a leading '(' — e.g. '(1234.5'.
    '-inf'/'+inf' are accepted as-is.
    """
    s = str(raw)
    exclusive = s.startswith("(")
    if exclusive:
        s = s[1:]
    return float(s), exclusive


class InMemoryBehaviorState:
    """Same public surface as BehaviorState, backed by plain dicts."""

    def __init__(self) -> None:
        self._z: dict[str, dict[str, float]] = {}
        self._kv: dict[str, str] = {}
        self._hash: dict[str, dict[str, str]] = {}

    async def ping(self) -> bool:
        return True

    async def delete_matching(self, pattern: str) -> int:
        """Glob-delete, for interface parity with BehaviorState.

        A fresh instance starts empty, so this is usually a no-op here — but
        the eval harnesses call it, and a fake that lacks a method the real
        backend has is a fake that hides call-site errors.
        """
        import fnmatch

        deleted = 0
        for store in (self._z, self._kv, self._hash):
            for key in [k for k in store if fnmatch.fnmatch(k, pattern)]:
                del store[key]
                deleted += 1
        return deleted

    # ── Sorted sets ──────────────────────────────────────────────────────
    async def zadd_point(self, key: str, score: float, member: str, ttl_seconds: int) -> bool:
        self._z.setdefault(key, {})[member] = score
        return True

    async def zremrangebyscore(self, key, min_score, max_score) -> bool:
        """Remove members whose score falls in the given range.

        Mirrors Redis: the range is inclusive unless a bound is prefixed
        with '('. Both bounds are honoured rather than assuming the callers'
        current '-inf' lower bound, so this stays correct if a caller
        changes.
        """
        if key not in self._z:
            return True
        lo, lo_excl = _parse_bound(min_score)
        hi, hi_excl = _parse_bound(max_score)

        def in_range(s: float) -> bool:
            above = s > lo if lo_excl else s >= lo
            below = s < hi if hi_excl else s <= hi
            return above and below

        self._z[key] = {m: s for m, s in self._z[key].items() if not in_range(s)}
        return True

    async def zrange_withscores(self, key: str):
        return sorted(self._z.get(key, {}).items(), key=lambda kv: kv[1])

    # ── Simple keys ──────────────────────────────────────────────────────
    async def set_simple(self, key: str, value: str, ttl_seconds: int) -> bool:
        self._kv[key] = value
        return True

    async def get_simple(self, key: str):
        return self._kv.get(key)

    async def exists(self, key: str) -> bool:
        return key in self._kv

    async def delete(self, key: str) -> bool:
        self._kv.pop(key, None)
        return True

    # ── Flags ────────────────────────────────────────────────────────────
    async def get_flag(self, key: str):
        raw = self._kv.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set_flag(self, key: str, value: dict[str, Any], ttl_seconds: int) -> bool:
        self._kv[key] = json.dumps(value)
        return True

    # ── Object-track hashes (Day 9) ──────────────────────────────────────
    async def hset_object_track(self, object_track_id: str, fields: dict[str, Any],
                                 ttl_seconds: int) -> bool:
        key = f"object_track:{object_track_id}"
        # Same string coercion real Redis hashes force on the caller.
        coerced = {
            k: (json.dumps(v) if isinstance(v, (list, dict)) else str(v))
            for k, v in fields.items()
        }
        self._hash.setdefault(key, {}).update(coerced)
        return True

    async def hget_object_track(self, object_track_id: str):
        key = f"object_track:{object_track_id}"
        raw = self._hash.get(key)
        if not raw:
            return {}   # matches BehaviorState: {} = reachable but absent
        result: dict[str, Any] = dict(raw)
        if "position_history" in result:
            try:
                result["position_history"] = json.loads(result["position_history"])
            except (json.JSONDecodeError, TypeError):
                result["position_history"] = []
        return result

    async def expire_object_track(self, object_track_id: str, ttl_seconds: int) -> bool:
        return f"object_track:{object_track_id}" in self._hash

    # ── Proximity flags (Day 9) ──────────────────────────────────────────
    async def set_proximity_flag(
        self, object_track_id: str, ttl_seconds: int, video_time: float | None = None
    ) -> bool:
        value = "1" if video_time is None else repr(float(video_time))
        return await self.set_simple(f"object_proximity:{object_track_id}", value, ttl_seconds)

    async def clear_proximity_flag(self, object_track_id: str) -> bool:
        return await self.delete(f"object_proximity:{object_track_id}")

    async def get_proximity_flag(self, object_track_id: str) -> bool:
        return await self.exists(f"object_proximity:{object_track_id}")

    async def get_proximity_video_time(self, object_track_id: str) -> float | None:
        raw = await self.get_simple(f"object_proximity:{object_track_id}")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None


def install() -> InMemoryBehaviorState:
    """Swap the fake in as the process-wide BehaviorState singleton.

    Patches both the singleton and the module-level accessor, because the
    detectors resolve `get_behavior_state` from the module at call time.
    """
    from backend.services import state as state_module

    fake = InMemoryBehaviorState()
    state_module._state_instance = fake
    state_module.get_behavior_state = lambda: fake  # type: ignore[assignment]
    return fake
