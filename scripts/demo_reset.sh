#!/bin/bash
# scripts/demo_reset.sh
set -e

echo "🔄 Sentinel IQ Demo Reset..."

# 1. Kill orphaned ffmpeg
pkill -f "ffmpeg.*hls" 2>/dev/null || true
sleep 2

# 2. Clear HLS directories
rm -rf hls/ && mkdir -p hls/

# 3. Clear evidence files (keep directory structure)
find evidence/ -name "*.mp4" -o -name "*.jpg" -o -name "*.pdf" \
  2>/dev/null | xargs rm -f 2>/dev/null || true

# Determine DB path(s)
for DB_FILE in "sentinel.db" "output/sentinel.db"; do
  if [ -f "$DB_FILE" ]; then
    sqlite3 "$DB_FILE" << 'SQL'
-- Ensure demo officers exist
INSERT OR IGNORE INTO officers (id, name, lat, lng, status, current_alert_id, last_updated)
VALUES (1, 'Officer Chen', 37.7749, -122.4194, 'AVAILABLE', NULL, datetime('now')),
       (2, 'Officer Park', 37.7751, -122.4180, 'AVAILABLE', NULL, datetime('now')),
       (3, 'Officer Ramirez', 37.7740, -122.4210, 'AVAILABLE', NULL, datetime('now')),
       (4, 'Officer Thompson', 37.7760, -122.4220, 'OFFLINE', NULL, datetime('now'));

-- Officers: reset to AVAILABLE (keep OFFLINE seed entries)
UPDATE officers
SET status='AVAILABLE', current_alert_id=NULL, last_updated=datetime('now')
WHERE name != 'Officer Thompson';  -- Thompson stays OFFLINE by seed design

-- Clear all generated data (keep seed: cameras, officers, demo config)
DELETE FROM routed_alerts;
DELETE FROM alerts;
CREATE TABLE IF NOT EXISTS alert_feedback (id INTEGER PRIMARY KEY, alert_id INTEGER, verdict TEXT, created_at DATETIME);
DELETE FROM alert_feedback;
CREATE TABLE IF NOT EXISTS feedback_flag_log (id INTEGER PRIMARY KEY, alert_id INTEGER, created_at DATETIME);
DELETE FROM feedback_flag_log;
DELETE FROM push_delivery_log;
CREATE TABLE IF NOT EXISTS officer_connectivity_log (id INTEGER PRIMARY KEY, officer_id INTEGER, status TEXT, created_at DATETIME);
DELETE FROM officer_connectivity_log;
CREATE TABLE IF NOT EXISTS client_error_log (id INTEGER PRIMARY KEY, error TEXT, created_at DATETIME);
DELETE FROM client_error_log;

-- Keep push_subscriptions: re-subscribing on phone is slow.
-- Only clear if push notification tests are failing (see Known Risk 7).
-- DELETE FROM push_subscriptions;  -- uncomment only if Scene 5 push fails

-- Verify:
SELECT '  Officers after reset:' AS info;
SELECT name, status FROM officers;
SQL
  fi
done

echo ""
echo "⚠️  MANUAL BROWSER STEPS (open DevTools → Application → Local Storage):"
echo "  Delete key: sentineliq.pending_acks"
echo "  Delete key: sentineliq.dismissed_unrouted"
echo "  Delete key: sentineliq.sync_lock"
echo "  Then: Ctrl+R to reload dashboard"
echo ""
echo "Running verification..."
bash scripts/verify_demo_ready.sh || python3 scripts/verify_demo_ready.py || python scripts/verify_demo_ready.py
