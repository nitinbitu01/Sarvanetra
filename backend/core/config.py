"""
backend/core/config.py — Centralized settings via pydantic-settings.

Day 6 additions: REID_* settings for threshold, stale-appearance, and retention.

All config values come from .env (or environment variables).
Never hardcode secrets, DB paths, or origins elsewhere — import `settings`.

Switching DATABASE_URL from SQLite to PostgreSQL is the only change needed
to move to Postgres — no code changes, only this string and running Alembic.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    DEMO_MODE: bool = True
    SENTINEL_DEMO_MODE: bool = True
    PTZ_RECOVERY_ETA_SECONDS: int = 10
    LOITERING_THRESHOLD_SECONDS: int = 8

    # Database — SQLite default, Postgres-ready
    DATABASE_URL: str = "sqlite:///./output/sentinel.db"

    # Auth
    JWT_SECRET: str = "CHANGE_ME_use_openssl_rand_hex_32"
    JWT_EXPIRE_MINUTES: int = 60
    JWT_ALGORITHM: str = "HS256"

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    AUTH_RATE_LIMIT_PER_MINUTE: int = 10

    # ReID — Day 6 (all overridable via .env, never hardcoded in service logic)
    # Cosine-similarity threshold above which two tracks are auto-merged as the same person.
    # Values below REID_REVIEW_LOWER_THRESHOLD create a new identity.
    # Values in the band go to the human review queue.
    # These defaults MUST be validated by running scripts/reid_validate.py on real footage
    # before treating auto-merge decisions as ground truth in production.
    REID_AUTO_MERGE_THRESHOLD: float = 0.85
    REID_REVIEW_LOWER_THRESHOLD: float = 0.65
    # Hours after which an appearance-based match is flagged with a confidence caveat
    # even on auto-merge (clothing/context may have changed over long gaps).
    REID_APPEARANCE_STALE_HOURS: int = 6
    # Days after which a GlobalPerson with no new sightings is soft-deleted.
    # IMPORTANT: 90 is a placeholder — confirm legally-appropriate value with
    # whoever owns compliance for this deployment (DPDP Act, MHA guidelines).
    REID_RETENTION_DAYS: int = 90
    # Fine-tuned OSNet-IBN checkpoint overlaid onto the ImageNet-pretrained
    # backbone at load time (backend/services/reid_embedder.py). Empty string
    # means "use torchreid's own default pretrained weights only", which is
    # always a safe fallback.
    #
    # The bundled default was trained on Market-1501 (backend/scripts/
    # train_reid_market1501.py, 250 epochs, Rank-1 91.3% / mAP 76.0% on
    # Market-1501's OWN held-out split). That number describes fit to
    # Market-1501's 6 Tsinghua-campus cameras, not to any Gujarat camera —
    # ReID domain-generalization research is consistent that a checkpoint
    # tuned on one city's cameras does not reliably transfer to another's
    # better than strong generic pretrained weights. Treat this as "the
    # training pipeline produced a real checkpoint" evidence, not "the
    # deployed system is more accurate now" evidence, until it is re-measured
    # against real Gujarat footage.
    REID_CHECKPOINT_PATH: str = (
        "output/reid_train/osnet_ibn_market1501/model/model.pth.tar-250"
    )

    # Face watchlist matching — Day 7
    # High-stakes threshold: a false positive here can trigger a wrongful
    # stop/detention, unlike ReID's journey-building false positives (lower
    # stakes — worst case is a mislabeled trail). Deliberately NOT reused from
    # REID_AUTO_MERGE_THRESHOLD (0.85) — that value was tuned for a different
    # error profile. MUST be validated via `scripts/reid_validate.py --mode
    # face` before trusting this in anything resembling real deployment.
    WATCHLIST_FACE_MATCH_THRESHOLD: float = 0.80

    # Camera heartbeat — Day 7
    CAMERA_HEARTBEAT_INTERVAL_SECONDS: int = 30
    # Consecutive failed checks required before flipping ONLINE -> OFFLINE.
    # Guards against flapping status on one transient hiccup, at the cost of
    # up to one extra cycle of detection latency (worst case ~2x interval).
    CAMERA_HEARTBEAT_FAILURE_THRESHOLD: int = 2

    # ── Behavior engine — Day 8 (loitering + crowd anomaly) ──────────────────
    # Redis backend for shared/durable detector state (see services/state.py).
    # Falls back to skip-and-log if unreachable — see state.py module docstring
    # for why "skip" (not "buffer in memory") is the chosen failure mode.
    REDIS_URL: str = "redis://localhost:6379/0"

    # Loitering: a person must stay within LOITER_RADIUS_METERS of their own
    # rolling centroid for LOITER_DURATION_SEC of continuous video_time before
    # an alert fires. Real-world units, converted to pixels per-camera via
    # camera_calibration (services/camera_calibration.py) — NOT a flat pixel
    # constant, so the same behavior definition holds across cameras with
    # different fields of view.
    LOITER_DURATION_SEC: float = 60.0
    LOITER_RADIUS_METERS: float = 3.0
    # Fallback px-per-meter for a camera with no calibration row. Deliberately
    # conservative (assumes a fairly zoomed-in view) — alerts computed with
    # this default are tagged calibration_method='uncalibrated' downstream so
    # they're never silently mixed with calibrated ones. See camera_calibration.py.
    DEFAULT_PX_PER_METER: float = 40.0

    # Crowd anomaly: current on-screen person count vs a rolling baseline
    # average over CROWD_BASELINE_WINDOW_SEC of video_time.
    CROWD_BASELINE_WINDOW_SEC: float = 300.0       # 5 min rolling baseline
    CROWD_SURGE_MULTIPLIER: float = 2.0            # count >= baseline_avg * this -> candidate surge
    # Guards a tiny baseline (e.g. 1->3 people) from registering as a "3x
    # surge" — not in the original spec verbatim, but a real false-positive
    # risk with a ratio-only trigger on a near-empty baseline.
    CROWD_MIN_ABSOLUTE_COUNT: int = 4
    # A candidate surge must sustain for this long before firing — this is
    # what makes a "brief group photo, 10s" a non-event while a genuine
    # gathering still fires within roughly one COOLDOWN cycle.
    CROWD_MIN_DURATION_SEC: float = 15.0
    CROWD_COOLDOWN_SEC: float = 180.0
    # Drift check (observability, §4): warn if the rolling baseline itself
    # jumps by this multiple between consecutive computations — usually a
    # sign of a degraded/miscounting detector upstream, not a real crowd.
    CROWD_DRIFT_WARN_MULTIPLIER: float = 2.0

    # ── Retention policy — Day 9 (§3) ───────────────────────────────────────
    # IMPORTANT: these defaults are engineering placeholders. The actual
    # retention periods for video surveillance data and audit logs must be
    # confirmed with whoever owns compliance for this deployment (DPDP Act,
    # MHA guidelines, applicable State police IT policy) before go-live.
    # Changing these values here is sufficient — no code changes needed.
    JOURNEY_RETENTION_DAYS: int = 30        # journey rows older than this are hard-deleted
    AUDIT_RETENTION_DAYS: int = 365         # journey_query_log rows kept for 1 year
    # Audit trail intentionally outlives the data it audits: a journey row is
    # deleted after 30 days but the log of WHO looked at it is kept for a year
    # so any audit review of access can still reconstruct the question
    # "who queried this person's movement history, and why" even after the
    # underlying journey data itself is gone.

    # ── Abandoned-object detector — Day 9 (§4/§5) ──────────────────────────
    # These three constants carry the highest false-positive risk in the
    # system. Re-run tests/abandoned_object_eval/eval_runner.py whenever any
    # of them is changed — see the Day 9 prompt §5 for the full rationale.
    OWNERSHIP_RADIUS_METERS: float = 2.0    # person must stay within this radius of the object
    ABANDON_DURATION_SEC: float = 60.0      # object must be static for this long without owner
    STATIC_MOVEMENT_PX_THRESHOLD: float = 10.0  # max centroid drift (px) to count as "static"
    # How long AFTER the last frame in which a person was within the ownership
    # radius the object still counts as attended. Measured in VIDEO_TIME, like
    # every other duration in that detector (ABANDON_DURATION_SEC, span_sec).
    # This replaced a wall-clock Redis TTL that answered the same question on a
    # different clock from the rest of the state machine — which suppressed
    # genuine abandonment on any feed processed faster than real-time (e.g.
    # config.yaml's `loop_video: true`, or catch-up after a backlog) and
    # expired too early on a stalled feed.
    #
    # Sizing: this is a tolerance for MISSED PERSON DETECTIONS, not a grace
    # period for the owner. At config.yaml's processing.fps=5, 2.0 seconds is
    # 10 consecutive processed frames in which the person could go undetected
    # (brief occlusion, bad angle, a dropped detection) without the object
    # being declared unattended. Anything much larger stops being a detection
    # tolerance and silently becomes a delay added to EVERY abandonment alert,
    # since the unattended clock only starts once this window lapses.
    # Re-derive this if processing.fps changes, and re-run
    # tests/abandoned_object_eval/eval_runner.py — the Day 9 note about
    # re-running on constant changes applies to this constant too.
    ABANDON_PROXIMITY_WINDOW_SEC: float = 2.0
    # STATIC_MOVEMENT_PX_THRESHOLD is the one constant not converted through calibration —
    # it guards against the camera/object jitter case (small pixel wobble of a truly
    # static object) and intentionally stays in pixel space. The ownership radius IS
    # converted per-camera via camera_calibration (same as loitering), so the
    # "is the owner still near the object" question is physically meaningful.

    # ── SENTINEL IQ scoring engine — Day 10 ─────────────────────────────────
    # Night multiplier window, as local-clock hours [start, end) wrapping
    # midnight. 20/6 means "20:00 through 05:59 counts as night".
    IQ_NIGHT_START_HOUR: int = 20
    IQ_NIGHT_END_HOUR: int = 6
    # Alert timestamps are stored in UTC (datetime.utcnow throughout this
    # codebase), but "is it night" is a question about LOCAL time. Gujarat is
    # IST = UTC+5:30. Without this offset the night window would be evaluated
    # 5.5 hours out of phase — 20:00-06:00 UTC is 01:30-11:30 IST, i.e. the
    # multiplier would apply through the local morning and not at all through
    # the local evening. Change this if deploying outside IST.
    IQ_TIMEZONE_OFFSET_HOURS: float = 5.5
    # Cross-zone travel: how far back to look for sightings of the same
    # global identity in a DIFFERENT Camera.zone.
    IQ_CROSS_ZONE_WINDOW_MINUTES: int = 30

    # ── Alert fatigue management — Day 11 ───────────────────────────────────
    # All four values are .env-overridable, never hardcoded in service logic.
    # These are the trigger thresholds for dedup and zone-incident clustering;
    # change them here and they propagate everywhere without touching code.

    # Dedup window: two alerts for the same global_id + alert_type within this
    # many seconds of each other are treated as one real-world event. The loser
    # gets merged_into_alert_id set; the row is never deleted.
    # Only applies when both alerts have a non-null global_id (see Step 0 note:
    # until the Track-persistence fix lands, effectively inert).
    DEDUP_WINDOW_SEC: int = 60

    # Zone incident: geographic radius (haversine, using Camera.gps_lat/gps_lon)
    # within which alerts are considered to be at the same physical location.
    # 100m is the right order of magnitude for a single intersection or courtyard;
    # increase it for wider zones (e.g. train station concourse).
    ZONE_INCIDENT_RADIUS_METERS: float = 100.0

    # Zone incident: lookback window for counting nearby alerts. 120s (2 min)
    # matches the design brief — a burst in a short window is a genuine multi-
    # alert incident; a slow trickle over an hour is not.
    ZONE_INCIDENT_WINDOW_SEC: int = 120

    # Minimum number of Alert rows within the radius+window required to open
    # a ZoneIncident. Default 5 matches the design brief. Given the Step 0
    # finding that one physical location has up to 4 duplicate camera_ids,
    # haversine clustering collapses them to one point, so the threshold is
    # calibrated to real events, not camera-count inflation.
    ZONE_INCIDENT_MIN_ALERTS: int = 5

    # ── Push Notifications — Day 15 ──────────────────────────────────────────
    # Generate with: python -m backend.scripts.gen_vapid_keys
    # Override BOTH in .env. The defaults below are development-only and are
    # committed, which means they are public — treat them as compromised and
    # never use them anywhere real.
    VAPID_PUBLIC_KEY: str = (
        "BEXsYoY0-GnTN3qNOCCMADWV3PXd8XKthSQRLio9Urm7iq4nYkJs8DgAumt_WIGIUVlxBpXehU5IrLD1TEAXnPM"
    )
    VAPID_PRIVATE_KEY: str = "NQiNkT5QQsZJzQlG0lNZTcOS8aj4kHzztP578FcNCHs"
    # Required by every push service (FCM, Mozilla) as an abuse contact.
    # Omitting it produces a 400 whose error text points at signing, not at
    # the missing contact — an hour of debugging the wrong thing.
    VAPID_CONTACT_EMAIL: str = "admin@sentineliq.local"

    # Push TTL tracks the ACK window: a push that lands after the alert has
    # already escalated or been acknowledged is worse than no push — it sends
    # an officer to something already handled. Set at send time from
    # ACK_TIMEOUT_SECONDS (see push/service.py) rather than duplicated here,
    # so the two cannot drift.
    #
    # Per-subscription send cap. A stuck detection loop or a compromised
    # camera must not be able to vibrate every officer's phone continuously.
    PUSH_RATE_LIMIT_PER_SUB_PER_MINUTE: int = 3
    # Subscriptions with no successful send in this many days are deleted by
    # the daily cleanup job — almost always a replaced device or uninstalled
    # PWA holding an endpoint that will never resolve again.
    PUSH_SUBSCRIPTION_STALE_DAYS: int = 30

    # ── Face-match dispatch interlock ────────────────────────────────────────
    # When True, a WATCHLIST_FACE_MATCH does NOT auto-dispatch an officer
    # unless the matching threshold has been validated against real labelled
    # data (see services/threshold_validation.py). The alert still fires, is
    # still scored, still captures evidence and still reaches the review
    # queue — a human can dispatch from there. What is withheld is the
    # AUTOMATIC part: sending a named officer to a physical location on the
    # strength of a threshold nobody has measured.
    #
    # Set False only with a deliberate, recorded decision to accept that risk.
    REQUIRE_VALIDATED_FACE_THRESHOLD_FOR_DISPATCH: bool = True

    # ── Alert Routing — Day 14 ───────────────────────────────────────────────
    # Seconds an assigned officer has to ACK before the alert escalates.
    # Read at escalation-tick time, never cached at startup, so changing it
    # and restarting takes effect with no code change. Demo runs: 15-20.
    ACK_TIMEOUT_SECONDS: int = 15
    # How often the escalation tick runs.
    ESCALATION_TICK_SECONDS: int = 5

    # ── Evidence Lock — Day 13 ───────────────────────────────────────────────
    # Clip window around a CRITICAL alert's event time.
    EVIDENCE_PRE_SECONDS: float = 30.0
    EVIDENCE_POST_SECONDS: float = 60.0
    # Extra headroom on the frame ring buffer. The buffer must still contain
    # the PRE window at the moment we slice, which happens POST seconds AFTER
    # the event — so it has to hold PRE+POST plus slack for jitter. The buffer
    # size is DERIVED from these three (see frame_buffer.py), never configured
    # independently, so it cannot drift into being too small to hold its own
    # pre-roll.
    EVIDENCE_BUFFER_MARGIN_SECONDS: float = 15.0
    # JPEG quality for buffered frames. Raw frames are not viable: 640x480x3
    # is ~921KB, and 105s at 5fps would be ~480MB PER CAMERA. At q85 a frame
    # is ~40KB, so the same window is ~21MB. Lower this before lowering the
    # window if memory is tight.
    EVIDENCE_JPEG_QUALITY: int = 85
    # Downscale frames to at most this width before JPEG-encoding them into
    # the evidence ring buffer. 0 = keep native resolution.
    #
    # This is a THROUGHPUT setting, not a storage one. add_frame() encodes on
    # the detection thread for every processed frame, and 1080p imencode was
    # measured at 16.2 ms - more than the YOLO detect+track pass itself
    # (11.7 ms). It was the largest single limit on concurrent cameras while
    # presenting as a GPU limit. At 960px encoding costs 4.3 ms.
    #
    # 960 keeps evidence clips clearly legible (faces/plates are cropped from
    # the live frame at full resolution elsewhere, not from this buffer).
    # Set to 0 if a deployment needs native-resolution evidence and can spare
    # the per-frame cost.
    EVIDENCE_BUFFER_MAX_WIDTH: int = 960
    # Codec for the written clip. avc1 (H.264) is the default because the
    # dashboard plays evidence in a browser and Chrome will not play mp4v
    # (MPEG-4 Part 2). If the avc1 encoder is unavailable in the installed
    # OpenCV build, evidence_capture falls back to mp4v and records which
    # codec was actually used — a clip that exists but won't play is a
    # failure mode worth being able to diagnose after the fact.
    EVIDENCE_VIDEO_CODEC: str = "avc1"
    EVIDENCE_VIDEO_CODEC_FALLBACK: str = "mp4v"
    # Root for evidence/{alert_id}/. Relative paths resolve from project root.
    EVIDENCE_DIR: str = "evidence"
    # Alert types that always count as CRITICAL regardless of score.
    EVIDENCE_CRITICAL_ALERT_TYPES: str = "WATCHLIST_FACE_MATCH,watchlist_vehicle"
    # SENTINEL IQ contribution at or above which an alert is CRITICAL even if
    # its type isn't in the list above.
    EVIDENCE_CRITICAL_IQ_THRESHOLD: float = 7.0

    # ── Feedback Loop (Day 16) ────────────────────────────────────────────────────
    FALSE_ALARM_FLAG_THRESHOLD: int = 3
    # Zones with >= this many false alarms today show the System Learning banner.
    # COSMETIC ONLY — never mutates detector thresholds or any config value.
    # Read fresh at submit time, never cached at startup.

    # ── HLS Streaming (Day 16) ───────────────────────────────────────────────────
    HLS_SEGMENT_DURATION: int = 2      # seconds per .ts segment
    HLS_LIST_SIZE:        int = 3      # segments retained in manifest + on disk
    HLS_DIR:              str = "hls"  # relative to project root

    # ── Dial-112 CAD Emergency Dispatch ──────────────────────────────────────────
    CAD_URBAN_SPEED_KMH: float = 30.0
    CAD_GPS_STALE_THRESHOLD_SEC: float = 60.0

    @property
    def evidence_critical_types(self) -> List[str]:
        return [t.strip() for t in self.EVIDENCE_CRITICAL_ALERT_TYPES.split(",") if t.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS_ORIGINS into a list."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()


# Module-level alias — `from backend.core.config import settings`
settings: Settings = get_settings()
