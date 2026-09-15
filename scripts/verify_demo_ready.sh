#!/bin/bash
# scripts/verify_demo_ready.sh
# Exit 0 = ready. Exit 1 = not ready.

ERRORS=0

check() {
  local desc=$1 val=$2 expected=$3
  if [ "$val" != "$expected" ]; then
    echo "❌ $desc: got '$val', expected '$expected'"
    ERRORS=$((ERRORS + 1))
  else
    echo "✅ $desc"
  fi
}

DB_FILE="sentinel.db"
if [ ! -f "$DB_FILE" ] && [ -f "output/sentinel.db" ]; then
  DB_FILE="output/sentinel.db"
fi

if [ -f "$DB_FILE" ]; then
  BUSY=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM officers WHERE status='BUSY'" 2>/dev/null || echo "0")
  check "Officers BUSY (should be 0)" "$BUSY" "0"

  ROUTED=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM routed_alerts WHERE status='ROUTED'" 2>/dev/null || echo "0")
  check "Active routed alerts (should be 0)" "$ROUTED" "0"

  ALERTS=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM alerts" 2>/dev/null || echo "0")
  check "Alerts in DB (should be 0)" "$ALERTS" "0"
fi

FFMPEG=$(pgrep -c -f "ffmpeg.*hls" 2>/dev/null || echo "0")
check "Orphaned ffmpeg processes (should be 0)" "$FFMPEG" "0"

HLS_SEGS=$(find hls/ -name "*.ts" 2>/dev/null | wc -l | tr -d ' ')
check "Stale HLS segments (should be 0)" "$HLS_SEGS" "0"

python3 -c "
from sentinel.config import settings
import sys
ok = True
if settings.ACK_TIMEOUT_SECONDS > 20:
    print(f'❌ ACK_TIMEOUT_SECONDS={settings.ACK_TIMEOUT_SECONDS} (demo max: 20)')
    ok = False
else:
    print(f'✅ ACK_TIMEOUT_SECONDS={settings.ACK_TIMEOUT_SECONDS}')
if not settings.DEMO_MODE:
    print('❌ DEMO_MODE=False (demo routes unavailable)')
    ok = False
else:
    print('✅ DEMO_MODE=True')
sys.exit(0 if ok else 1)
" 2>/dev/null || python -c "
from sentinel.config import settings
import sys
ok = True
if settings.ACK_TIMEOUT_SECONDS > 20:
    print(f'❌ ACK_TIMEOUT_SECONDS={settings.ACK_TIMEOUT_SECONDS} (demo max: 20)')
    ok = False
else:
    print(f'✅ ACK_TIMEOUT_SECONDS={settings.ACK_TIMEOUT_SECONDS}')
if not settings.DEMO_MODE:
    print('❌ DEMO_MODE=False (demo routes unavailable)')
    ok = False
else:
    print('✅ DEMO_MODE=True')
sys.exit(0 if ok else 1)
" || ERRORS=$((ERRORS + 1))

echo ""
if [ "$ERRORS" -eq "0" ]; then
  echo "✅ DEMO-READY. Start server: uvicorn sentinel.main:app --reload"
  echo "   Then verify in browser: DevTools → Network → WS = 1 connection"
  exit 0
else
  echo "❌ $ERRORS issue(s). Fix before starting Scene 1."
  exit 1
fi
