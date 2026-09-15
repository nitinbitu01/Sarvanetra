"""
backend/services/state.py — Redis-backed shared state for the Day 8 behavior
engine (loitering + crowd anomaly detection) and Day 9 abandoned-object
detection.

Why Redis instead of in-process deques/ring buffers:
  - Survives process restarts — a loitering window mid-accumulation isn't
    lost if the backend is killed and restarted.
  - Correct if the pipeline ever scales across multiple camera-worker
    processes (each process would otherwise have its own, inconsistent
    in-memory view of the same camera's state).

Failure mode (deliberate choice for this pass): if Redis is unreachable,
every method here returns None/False rather than raising or buffering in
memory. Callers (loitering_detector.py, crowd_detector.py,
abandoned_object_detector.py) treat that as "skip this tick, log it, try
again next tick" — never crash the detection pipeline over a state-backend
outage. This is the safer default: silently buffering in memory and resyncing
later risks a resync that fires a wall of stale alerts all at once, or
double-counts a window Redis already expired. A dropped tick or two of
loitering/crowd/abandonment tracking is an acceptable cost; a burst of
false/duplicate CRITICAL-adjacent alerts is not.

Detector classes never import `redis` directly or hold a client — every
access goes through this module, so the backend is swappable (e.g. for a
test fake, or a different store entirely) without touching detector logic.

Key namespaces used by callers (documented here, not enforced — callers own
their own key construction):
  loiter:{identity_key}                    sorted set, score=video_time,
                                            member="{video_time}:{cx},{cy}"
  crowd:baseline:{camera_id}               sorted set, score=video_time,
                                            member="{video_time}:{count}"
  crowd:cooldown:{camera_id}               simple key, TTL = cooldown period
  crowd:surge_start:{camera_id}            simple key, value=video_time
  behavior:active:{flag_type}:{identity}   simple key, JSON value
                                            {started_at, last_fired_at}
  object_track:{object_track_id}           hash, fields:
                                            camera_id, class, first_seen_time,
                                            last_seen_time, position_history
                                            (small JSON array, capped),
                                            candidate_owner_global_id,
                                            is_baseline, status
  object_proximity:{object_track_id}       simple key, "1" while any person
                                            is within OWNERSHIP_RADIUS_PX;
                                            TTL cleared by presence logic
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Rate-limit the "Redis unreachable" warning so a full outage doesn't spam
# one WARNING line per frame per camera — log at most once per this many
# seconds regardless of how many detector ticks hit the failure.
_WARN_INTERVAL_S = 30.0


class BehaviorState:
    """Thin async wrapper around a redis.asyncio client.

    Every method catches its own exceptions and returns a "no-op" value
    (None/False/[]) on failure rather than raising — see module docstring
    for why. `available` reflects whether the last operation succeeded.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client: Any = None
        self._connect_failed = False
        self._last_warn_at = 0.0
        # Guards client construction. Without it, concurrent detector
        # coroutines all reach _get_client before any of them assigns
        # self._client, and each builds its OWN client. With a real server
        # that is merely wasteful; with the in-memory fallback it is a
        # correctness bug - every caller gets a SEPARATE store, so a
        # loitering window accumulates points across several disconnected
        # instances and never fills. Observed as a burst of identical
        # "Redis unreachable" warnings, one per racing call.
        self._client_lock: Any = None

    async def _get_client(self) -> Any:
        """Lazily create the redis.asyncio client. Returns None if the
        `redis` package isn't installed or the URL can't be parsed —
        both are treated the same as "unreachable" by callers."""
        if self._client is not None:
            return self._client
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        async with self._client_lock:
            # Re-check: another coroutine may have built it while we waited.
            if self._client is not None:
                return self._client
            return await self._build_client()

    async def _build_client(self) -> Any:
        try:
            import redis.asyncio as redis_asyncio  # type: ignore[import]

            client = redis_asyncio.from_url(
                self._redis_url, decode_responses=True, socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            try:
                await client.ping()
            except Exception:
                fallback = self._in_memory_fallback()
                if fallback is None:
                    raise
                self._client = fallback
                return self._client
            self._client = client
            return self._client
        except ImportError:
            self._warn_once("redis package not installed — behavior engine state disabled.")
            return None
        except Exception as exc:
            self._warn_once(f"Failed to construct Redis client: {exc}")
            return None

    def _in_memory_fallback(self) -> Any:
        """Substitute an in-process Redis when no server is reachable.

        WHY THIS EXISTS
          Every behaviour detector - loitering, crowd, abandoned-object - reads
          and writes its window through this class, and each one returns
          "SKIPPED_NO_REDIS" when the client is unavailable. With no Redis
          server running, that silently disabled the entire behaviour engine:
          the alerts table held 72 rows, every one an ANPR result, and not a
          single behavioural detection in the system's history. Nothing was
          logged as broken because skipping is the designed response.

        THE TRADE-OFF, STATED PLAINLY
          The module docstring rejects in-memory buffering for a real reason:
          with several API workers, each process would hold its own view of a
          camera's window and they would disagree. That reasoning is sound for
          a multi-worker deployment and this fallback does NOT fix it.

          It is for the single-process case - a demo, a replay, a local
          pipeline run - where the choice is not "shared state vs local state"
          but "local state vs no detections at all". Approximate windows that
          fire beat perfect windows that never run.

          Run a real Redis and this is never constructed; the ping above only
          falls through when the server is genuinely unreachable.
        """
        try:
            import fakeredis.aioredis  # type: ignore[import]
        except ImportError:
            self._warn_once("Redis unreachable and fakeredis not installed — "
                            "behavior engine disabled.")
            return None
        logger.warning(
            "[behavior-state] Redis unreachable at %s — using IN-PROCESS state. "
            "Behaviour detectors will run, but windows are NOT shared between "
            "processes. Start a Redis server for multi-worker deployments.",
            self._redis_url)
        return fakeredis.aioredis.FakeRedis(decode_responses=True)

    def _warn_once(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warn_at >= _WARN_INTERVAL_S:
            logger.warning("[behavior-state] %s (further warnings suppressed for %.0fs)",
                            message, _WARN_INTERVAL_S)
            self._last_warn_at = now
        from backend.services.metrics import counters
        counters.incr("redis_errors_total")

    async def ping(self) -> bool:
        """Explicit connectivity check — used at startup logging only."""
        client = await self._get_client()
        if client is None:
            return False
        try:
            await client.ping()
            return True
        except Exception as exc:
            self._warn_once(f"Redis ping failed: {exc}")
            return False

    # ── Sorted sets (loitering window, crowd baseline) ──────────────────────

    async def zadd_point(self, key: str, score: float, member: str, ttl_seconds: int) -> bool:
        """ZADD one point and refresh the key's TTL. Returns False on any failure."""
        client = await self._get_client()
        if client is None:
            return False
        try:
            async with client.pipeline(transaction=True) as pipe:
                pipe.zadd(key, {member: score})
                pipe.expire(key, ttl_seconds)
                await pipe.execute()
            return True
        except Exception as exc:
            self._warn_once(f"zadd_point failed for key={key}: {exc}")
            return False

    async def zremrangebyscore(self, key: str, min_score: float | str, max_score: float | str) -> bool:
        client = await self._get_client()
        if client is None:
            return False
        try:
            await client.zremrangebyscore(key, min_score, max_score)
            return True
        except Exception as exc:
            self._warn_once(f"zremrangebyscore failed for key={key}: {exc}")
            return False

    async def zrange_withscores(self, key: str) -> list[tuple[str, float]] | None:
        """Full sorted-set contents, ascending by score. None on failure
        (distinct from [] which means 'reachable but genuinely empty')."""
        client = await self._get_client()
        if client is None:
            return None
        try:
            raw = await client.zrange(key, 0, -1, withscores=True)
            return [(member, float(score)) for member, score in raw]
        except Exception as exc:
            self._warn_once(f"zrange_withscores failed for key={key}: {exc}")
            return None

    # ── Simple keys (cooldown, surge-start marker) ───────────────────────────

    async def set_simple(self, key: str, value: str, ttl_seconds: int) -> bool:
        client = await self._get_client()
        if client is None:
            return False
        try:
            await client.set(key, value, ex=ttl_seconds)
            return True
        except Exception as exc:
            self._warn_once(f"set_simple failed for key={key}: {exc}")
            return False

    async def get_simple(self, key: str) -> str | None:
        client = await self._get_client()
        if client is None:
            return None
        try:
            return await client.get(key)
        except Exception as exc:
            self._warn_once(f"get_simple failed for key={key}: {exc}")
            return None

    async def exists(self, key: str) -> bool:
        client = await self._get_client()
        if client is None:
            return False
        try:
            return bool(await client.exists(key))
        except Exception as exc:
            self._warn_once(f"exists failed for key={key}: {exc}")
            return False

    async def delete(self, key: str) -> bool:
        client = await self._get_client()
        if client is None:
            return False
        try:
            await client.delete(key)
            return True
        except Exception as exc:
            self._warn_once(f"delete failed for key={key}: {exc}")
            return False

    # ── Maintenance / test-isolation helper ─────────────────────────────────

    async def delete_matching(self, pattern: str) -> int:
        """Delete every key matching a glob pattern. Returns the count deleted.

        Exists for TEST ISOLATION, not for detector logic — no detector calls
        this, and none should. The eval harnesses use it to clear their own
        `*EVAL-CAM-*` keyspace before a run, because the debounce flags
        (behavior:active:*) and crowd cooldown keys are specifically designed
        to survive across ticks, which also means they survive across RUNS: a
        second run against the same Redis would find every alert already
        debounced and report recall 0.00, looking exactly like a detector
        regression when nothing had changed.

        Uses SCAN, not KEYS, so it does not block the server on a large
        keyspace. Callers must pass a specific pattern — deliberately no
        default, and no FLUSHDB anywhere, so this can never wipe a database
        someone pointed at real data by mistake.
        """
        client = await self._get_client()
        if client is None:
            return 0
        try:
            deleted = 0
            async for key in client.scan_iter(match=pattern, count=500):
                await client.delete(key)
                deleted += 1
            return deleted
        except Exception as exc:
            self._warn_once(f"delete_matching failed for pattern={pattern}: {exc}")
            return 0

    # ── Active-flag helper (debounce state machine replacement) ─────────────

    async def get_flag(self, key: str) -> dict[str, Any] | None:
        raw = await self.get_simple(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set_flag(self, key: str, value: dict[str, Any], ttl_seconds: int) -> bool:
        return await self.set_simple(key, json.dumps(value), ttl_seconds)

    # ── Object track hash (Day 9 abandoned-object detector) ───────────────

    async def hset_object_track(self, object_track_id: str, fields: dict[str, Any],
                                 ttl_seconds: int) -> bool:
        """Set one or more fields on the object_track:{id} hash and refresh TTL.

        'position_history' should be passed as a JSON-serialized list by the
        caller. Other fields are stored as their str() representation.
        Returns False on any failure.
        """
        client = await self._get_client()
        if client is None:
            return False
        key = f"object_track:{object_track_id}"
        # Coerce all values to strings for Redis hash storage.
        str_fields = {k: (json.dumps(v) if isinstance(v, (list, dict)) else str(v))
                      for k, v in fields.items()}
        try:
            async with client.pipeline(transaction=True) as pipe:
                pipe.hset(key, mapping=str_fields)
                pipe.expire(key, ttl_seconds)
                await pipe.execute()
            return True
        except Exception as exc:
            self._warn_once(f"hset_object_track failed for id={object_track_id}: {exc}")
            return False

    async def hget_object_track(self, object_track_id: str) -> dict[str, Any] | None:
        """Return all fields of the object_track:{id} hash, or None on failure
        (distinct from {} which means 'reachable but key does not exist').

        'position_history' is returned as a Python list (JSON-parsed).
        """
        client = await self._get_client()
        if client is None:
            return None
        key = f"object_track:{object_track_id}"
        try:
            raw = await client.hgetall(key)
            if not raw:
                return {}   # key not found or empty hash
            result: dict[str, Any] = dict(raw)
            # Deserialize position_history back to list
            if "position_history" in result:
                try:
                    result["position_history"] = json.loads(result["position_history"])
                except (json.JSONDecodeError, TypeError):
                    result["position_history"] = []
            return result
        except Exception as exc:
            self._warn_once(f"hget_object_track failed for id={object_track_id}: {exc}")
            return None

    async def expire_object_track(self, object_track_id: str, ttl_seconds: int) -> bool:
        """Refresh the TTL on an existing object_track key. Returns False on failure."""
        client = await self._get_client()
        if client is None:
            return False
        key = f"object_track:{object_track_id}"
        try:
            await client.expire(key, ttl_seconds)
            return True
        except Exception as exc:
            self._warn_once(f"expire_object_track failed for id={object_track_id}: {exc}")
            return False

    # ── Proximity flag (Day 9 abandoned-object detector) ──────────────────

    async def set_proximity_flag(
        self, object_track_id: str, ttl_seconds: int, video_time: float | None = None
    ) -> bool:
        """Record that a person was within OWNERSHIP_RADIUS_PX of this object.

        The stored VALUE is the video_time of that sighting. That value — not
        the key's TTL — is what the detector compares against to decide
        whether the object is still attended.

        `ttl_seconds` is now garbage collection only: it stops the key
        outliving the object track forever. It must be set generously. It is
        deliberately NOT the semantic clock any more, because it is wall-clock
        and the detector's state machine is video_time (see
        abandoned_object_detector.on_object_update).

        Passing video_time=None writes the legacy "1" sentinel, which readers
        treat as "nearby, time unknown".
        """
        value = "1" if video_time is None else repr(float(video_time))
        return await self.set_simple(f"object_proximity:{object_track_id}", value, ttl_seconds)

    async def clear_proximity_flag(self, object_track_id: str) -> bool:
        """Explicitly clear the proximity flag (e.g. all persons have left frame)."""
        return await self.delete(f"object_proximity:{object_track_id}")

    async def get_proximity_flag(self, object_track_id: str) -> bool:
        """True if a proximity record exists at all, regardless of its age.

        Kept for callers that only need existence. Note this does NOT answer
        "is the object attended right now" — use get_proximity_video_time()
        and compare against the current video_time for that.
        """
        return await self.exists(f"object_proximity:{object_track_id}")

    async def get_proximity_video_time(self, object_track_id: str) -> float | None:
        """video_time of the most recent nearby-person sighting, or None.

        Returns None when there is no record, on a Redis failure, or when the
        stored value is the legacy "1" sentinel — the caller decides what a
        time-less record means rather than having a fabricated timestamp
        invented here.
        """
        raw = await self.get_simple(f"object_proximity:{object_track_id}")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None


_state_instance: BehaviorState | None = None


def get_behavior_state() -> BehaviorState:
    """Return the singleton BehaviorState (constructed once, lazily connects)."""
    global _state_instance
    if _state_instance is None:
        from backend.core.config import settings
        _state_instance = BehaviorState(settings.REDIS_URL)
    return _state_instance
