# Watchlist Matching & Real-Time Alerting — the core challenge requirement

The official hackathon brief's "Expected Solution Approach" states the challenge in one
sentence: *"demonstrate how live video streams are integrated with a searchable database
of watchlist records ... enabling continuous AI-powered analysis and automated alert
generation whenever a match is detected."* This document covers that pipeline
specifically — not a model (1-4), the actual thing judges test on live camera feeds.

Same discipline as every other doc here: file:line citations to code that runs, and
every fix below was reproduced broken, then verified fixed, not assumed.

---

## Summary

| What the brief asks for | Status before this pass | Status now |
|---|---|---|
| Live feed → watchlist match → alert | **Broken**: three disconnected implementations, none complete | ✅ Unified, verified end-to-end |
| Searchable, editable watchlist database | **No UI or API existed to edit it at all** | ✅ Add/list/deactivate/delete, live |
| Alert appears in real time | **Not pushed live** — DB-only, next-refresh visibility | ✅ Broadcast on hit |
| Plate search (`GET /plate-search`) | **Completely unreachable** — router never registered | ✅ Registered, working |

---

## 1. What was actually broken, precisely

Found by tracing every call site of `WatchlistService.check_and_alert` and every
`Alert(alert_type="STOLEN_VEHICLE_WATCHLIST_HIT", ...)` construction, not by inspecting
one function in isolation.

**Two watchlists, only one of them reachable.** `backend/services/watchlist_service.py`
matched plates against a standalone `watchlist.json` file (or a hardcoded
`DEFAULT_WATCHLIST` dict) that nothing in the app's UI or API could read or write.
Meanwhile `GET /plate-search` and the journey-lookup endpoints already queried a real
table, `watchlist_plates`, for the exact same question — "is this plate on a watchlist."
A plate added the only way an operator actually could (there was no way) would never
have been the one the live ANPR pipeline matched against anyway.

**The live pipeline's real alert never broadcast.** `backend/services/
live_24x7_pipeline.py` (confirmed the actual live-processing pipeline — see §3) creates
a genuine `Alert(alert_type="STOLEN_VEHICLE_WATCHLIST_HIT", ...)` row, correctly
triggers CAD auto-dispatch to the nearest patrol unit, and queues evidence-clip capture.
But no `dashboard_ws.broadcast` or Redis `publish_event` call existed anywhere in that
file — an alert reached the database and CAD dispatch correctly, but a control-room
operator would only see it on their next manual refresh, not live.

**The alert that did land was unlabeled.** The `Alert` row above never set
`subject_label`. `AlertFeed.jsx`'s primary display line reads
`{alert.subject_label || 'Unknown subject'}` — so even after a refresh, the single alert
type this platform is evaluated on rendered as *"Unknown subject."* It also had no entry
in `TYPE_ICONS`, falling back to a generic 🔔 with no severity styling — the least
visually distinctive alert in the feed, for what should be the most.

**`GET /plate-search` was dead code on both ends.** `backend/routers/v1/plate_search.py`
was never passed to `app.include_router()` anywhere in `backend/main.py` — confirmed by
grep, not inference. The frontend never called it either. A pre-existing broken parameter
on `search_plate` (`request_obj: Optional[Request] = None`, unused, invalid for FastAPI's
route-schema generation) would have crashed the entire app at startup the moment anyone
tried to register it — which is very likely *why* it was never registered.

## 2. What changed

### Data source — `WatchlistService` now reads `watchlist_plates`

`backend/services/watchlist_service.py::_load_from_db()` queries the real
`watchlist_plates` table, refreshed on a 5-second TTL (`WATCHLIST_DB_RELOAD_INTERVAL_S`)
rather than per-check — this runs on every confirmed plate lock, and a DB round trip
there is unnecessary when a few seconds of staleness is immaterial to whether a match
fires. `watchlist.json`/`DEFAULT_WATCHLIST` remain **only** as a fallback for when the DB
is genuinely unreachable (e.g. a standalone harvesting script with no DB configured) —
not a second intended path. The proven fuzzy-match algorithm itself
(1-character tolerance, measured 86.5% recall / 0% false-positive rate on 74 held-out
vehicles against a 10,000-entry watchlist) is untouched — only its data source changed.

A real bug surfaced and fixed while building this: the constructor originally read
`self._load_from_db() or _load_watchlist(...)`, and Python's `or` cannot distinguish "DB
query failed" (should fall back) from "DB query succeeded, watchlist is genuinely empty"
(should NOT fall back) — both are falsy. Fixed to an explicit `is not None` check;
regression-tested in `tests/test_watchlist_alerting_pipeline.py::
test_load_from_db_empty_result_does_not_silently_fall_back_to_json`.

### Editable, searchable watchlist — new endpoints

`backend/routers/v1/plate_search.py` (extended, not a new router file — it already
imported and queried `WatchlistPlate`):

| Method | Path | |
|---|---|---|
| GET | `/plate-search/watchlist` | List (`?include_inactive=true` for retired entries) |
| POST | `/plate-search/watchlist` | Add — reactivates a retired entry rather than 409ing |
| PATCH | `/plate-search/watchlist/{plate}` | Update reason/category, or retire (`active: false`) |
| DELETE | `/plate-search/watchlist/{plate}` | Hard delete (for correcting a mis-entered plate) |

All mutations require admin and are audit-logged (`WATCHLIST_PLATE_ADDED`,
`_REACTIVATED`, `_UPDATED`, `_DEACTIVATED`, `_DELETED`). A new `WatchlistPlate.active`
column (migration `d17_watchlist_plate_active`) lets a resolved case be retired without
deleting the row — a past `Alert` may still reference that plate by text.

`frontend/src/components/WatchlistManager.jsx` — new UI, reachable from the admin nav
("🚨 Watchlist"): add a plate with a reason and category, see it live within the
5-second TTL in the running pipeline, deactivate or delete it. **This is the control
surface for the hackathon's live "designated vehicle" test** — add the plate here before
the drive-past, not by editing a file on the server.

### `GET /plate-search` — registered, and its blocking bug fixed

`search_plate`'s dead `request_obj: Optional[Request] = None` parameter removed;
`app.include_router(plate_search_router, prefix="/api/v1")` added to `backend/main.py`.
Verified live: `GET /plate-search?plate=X` now returns real search results with
disclosed measured accuracy (precision 0.92, recall 0.896) and explicit "not available"
fields (vehicle colour, direction of travel, FASTag) rather than inventing them.

### Live broadcast on a real hit

`live_24x7_pipeline.py`'s stolen-vehicle section now sets `subject_label` on the `Alert`
row and calls `backend.cache.redis_cache.publish_event()` after creating it — the same
Redis pub/sub channel (`sentinel:broadcast`) `main.py`'s existing
`_pubsub_broadcast_task` already subscribes to and forwards into `ws_broadcast()`
(`main.py:593-610`, pre-existing, unmodified). `publish_event()` itself falls back to an
in-process queue when Redis is unavailable (pre-existing behavior in
`backend/cache/redis_cache.py`, unmodified) — so a same-process deployment
(`SENTINEL_PIPELINE_AUTOSTART=1`) still gets the broadcast even without Redis running;
the documented, recommended separate-process deployment (`python -m
backend.scripts.run_pipeline` alongside `uvicorn`) needs Redis for the cross-process
hop, same as every other Redis-dependent feature already in this codebase. This call
site is synchronous (a reader-thread loop, not asyncio) — bridged with `asyncio.run()`,
wrapped in try/except so a broadcast failure can never take down the ingestion loop that
already did the work that matters (the Alert row, CAD dispatch, and evidence queue all
happen first and are unaffected by a broadcast failure).

### Alert feed — the vehicle hit finally looks like what it is

`frontend/src/components/AlertFeed.jsx`: `STOLEN_VEHICLE_WATCHLIST_HIT` added to
`TYPE_ICONS` (🚔, `badge-critical`) and to the `isWatchlistMatch` check that drives the
red-bordered "critical" row styling — previously indistinguishable from a generic 🔔
alert with no special treatment at all.

## 3. Which pipeline is actually live, and why that matters

Two candidate pipelines call ANPR: `backend/workers/ai_worker.py` (`AIWorker`, reads
frames from a Redis queue at `redis://redis:6379` — a Docker-Compose-style hostname) and
`backend/services/live_24x7_pipeline.py`. Only the second is the one this deployment
actually runs: `ai_worker.py` requires a reachable Redis queue to even start pulling
frames, and Redis is not deployed in this environment (`main.py`'s own startup log:
*"Redis unreachable at redis://localhost:6379/0 — using IN-PROCESS state"*) — with no
Redis, `AIWorker.run()` cannot connect and never processes a frame. `live_24x7_pipeline.py`
is the one `backend/scripts/run_pipeline.py` starts, with an explicit, measured reason in
its own docstring for running it as a separate process from the API (running it in-process
made `/docs` and `/analytics/calibration/cameras` time out under load — 6.7GB RSS, 1,067s
CPU held by the pipeline). Both pipelines funnel plate reads through the same
`anpr_engine.process_vehicle_track()`, so the watchlist-data-source fix in §2 benefits
either one — but the missing-broadcast fix was specifically applied to
`live_24x7_pipeline.py`'s alert-creation code because that is the pipeline that is
actually reachable and running.

## 4. What the brief's watchlist categories map to today

The brief names: *stolen vehicles, wanted persons, missing persons, blacklisted
vehicles, suspect watchlists, or other entities of interest.*

- **Vehicles** (`WatchlistPlate.category`, free-text): `stolen`, `wanted`, `suspect`,
  `blacklisted`, `other` all work end-to-end today via the endpoints in §2 — the category
  field was already free-text, nothing forced only "stolen."
- **Persons** (`WatchlistPerson`, face-matching): a separate, already fully-wired
  pipeline (`backend/services/face_watchlist_matcher.py`) — real DB match, real Alert,
  real broadcast, already correct before this pass (used as the reference pattern for
  fixing the vehicle path). **Known, disclosed gap, not fixed in this pass**: the seeded
  demo `WatchlistPerson` rows carry `np.random.rand(512)` as their face embedding
  (`backend/db/seed_data.py:110-111`) — random noise, not a real face vector — so a live
  match against the seeded demo persons is not currently possible. This needs re-seeding
  with genuine reference-photo embeddings before a person-watchlist demo would work; the
  vehicle-plate path in this document does not have this problem, since the fix here
  routes through real, operator-entered plate text rather than a computed embedding.
  A watchlist-person management UI equivalent to `WatchlistManager.jsx` (add/list/retire
  a wanted/missing person with a reference photo) does not exist yet either — flagged as
  the next gap in this exact area, not silently left unaddressed.
