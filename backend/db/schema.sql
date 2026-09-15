-- backend/db/schema.sql
-- Full schema v12.0.0

-- ── Main violations table ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS traffic_violations (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_uuid          VARCHAR(64)   UNIQUE NOT NULL,
    violation_type          VARCHAR(32)   NOT NULL DEFAULT 'TRIPLE_RIDING',
    camera_id               VARCHAR(32)   NOT NULL,
    track_id                INTEGER       NOT NULL,
    vehicle_class           VARCHAR(32)   NOT NULL,

    -- Rider count evidence
    rider_count             INTEGER       NOT NULL DEFAULT 1,
    majority_ratio          FLOAT         NOT NULL DEFAULT 1.0,
    counting_method         VARCHAR(64)   NOT NULL DEFAULT 'direct',

    -- Speed
    speed_kmh               FLOAT         NOT NULL,

    -- ANPR
    license_plate           VARCHAR(32)   NULL,
    plate_confidence        FLOAT         NULL,
    plate_format_valid      BOOLEAN       NOT NULL DEFAULT 0,

    -- Helmet
    helmet_results          JSON          NULL,

    -- Fine
    violation_breakdown     JSON          NULL,
    challan_amount_inr      INTEGER       NOT NULL DEFAULT 0,

    -- Evidence
    crop_path               VARCHAR(512)  NOT NULL,
    plate_crop_path         VARCHAR(512)  NULL,
    composite_evidence_path VARCHAR(512)  NULL,
    evidence_panel_hashes   JSON          NULL,
    composite_hash          VARCHAR(64)   NULL,

    -- Environmental context
    weather_regime          VARCHAR(16)   NULL,
    camera_angle_deg        FLOAT         NULL,
    michelson_contrast      FLOAT         NULL,

    -- Review workflow
    challan_status          VARCHAR(16)   NOT NULL DEFAULT 'pending_review',
        -- pending_review | issued | contested | dismissed
    reviewed_by             VARCHAR(64)   NULL,
    reviewed_at             TIMESTAMP     NULL,
    dismiss_reason          TEXT          NULL,

    created_at              TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Deduplication: one row per (camera, track) per 30s window
    CONSTRAINT uq_violation_window UNIQUE (
        camera_id, track_id,
        CAST(strftime('%s', created_at) AS INTEGER) / 30
    )
);

CREATE INDEX IF NOT EXISTS idx_viol_status
    ON traffic_violations(challan_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_viol_plate
    ON traffic_violations(license_plate);
CREATE INDEX IF NOT EXISTS idx_viol_ratio
    ON traffic_violations(majority_ratio DESC, challan_status);
CREATE INDEX IF NOT EXISTS idx_viol_camera
    ON traffic_violations(camera_id, created_at DESC);

-- ── Shadow violations (pre-live validation) ───────────────────────────
CREATE TABLE IF NOT EXISTS shadow_violations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_uuid  TEXT    UNIQUE NOT NULL,
    camera_id       TEXT    NOT NULL,
    track_id        INTEGER NOT NULL,
    rider_count     INTEGER NOT NULL,
    majority_ratio  REAL    NOT NULL,
    speed_kmh       REAL    NOT NULL,
    weather_regime  TEXT    NOT NULL,
    ground_truth    TEXT    NOT NULL DEFAULT 'pending',
        -- pending | true_positive | false_positive
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Reviewer audit log (append-only, never UPDATE) ────────────────────
CREATE TABLE IF NOT EXISTS review_audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_id INTEGER     NOT NULL REFERENCES traffic_violations(id),
    action       VARCHAR(16) NOT NULL,   -- confirmed | dismissed | escalated
    reviewer_id  VARCHAR(64) NOT NULL,   -- from JWT token
    reviewer_ip  VARCHAR(45) NULL,
    reason       TEXT        NULL,
    timestamp    TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_vid
    ON review_audit_log(violation_id, timestamp DESC);

-- ── Retraining queue ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retraining_items (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_id      INTEGER NOT NULL,
    crop_path         TEXT    NOT NULL,
    label             TEXT    NOT NULL,   -- true_positive | false_positive
    rider_count       INTEGER NOT NULL,
    majority_ratio    REAL    NOT NULL,
    camera_id         TEXT    NOT NULL,
    weather_regime    TEXT    NOT NULL,
    used_in_training  BOOLEAN NOT NULL DEFAULT 0,
    added_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Camera calibrations ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS camera_calibrations (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id            VARCHAR(32) NOT NULL,
    homography_matrix    TEXT        NOT NULL,
    reprojection_error_m FLOAT       NOT NULL,
    camera_angle_deg     FLOAT       NOT NULL,
    calibrated_by        VARCHAR(64) NOT NULL,
    calibrated_on        DATE        NOT NULL,
    is_active            BOOLEAN     NOT NULL DEFAULT 1,
    created_at           TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
);
