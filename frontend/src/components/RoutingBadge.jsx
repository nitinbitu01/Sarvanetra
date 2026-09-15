// frontend/src/components/RoutingBadge.jsx
//
// Dispatch state for a CRITICAL alert: which officer, how long they have
// left, and an ACK button.
//
// COUNTDOWN TRUTH MODEL
//   The server is the only source of truth for elapsed time. On mount (and
//   on reconnect) we seed from GET /routing/active's `seconds_elapsed`, then
//   a local setInterval interpolates between polls purely for display. That
//   distinction is what makes a page reload resume at the right position
//   instead of restarting the countdown at full — the interval is a
//   rendering detail, never the clock.
import { useState, useEffect, useRef, useCallback } from 'react';
import { useAuth, API } from '../context/AuthContext';
import { useConnectionState } from '../context/WebSocketContext';
import { useAckQueueItem } from '../hooks/useAckQueue';
import { enqueueAck, dismiss as dismissAckQueueItem, flush as flushAckQueue } from '../utils/ackQueue';

const ACTIVE_STATES = ['ROUTED', 'ESCALATED_ROUTED'];
// No officer was ever assigned, so there is nothing to acknowledge. The ACK
// button is hidden entirely rather than disabled — a disabled button implies
// "not yet", which is the wrong story for a permanent terminal state.
const NO_OFFICER_STATES = ['UNROUTED', 'ESCALATED_UNROUTED'];

function fmtRemaining(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function fmtClock(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString();
}

function CountdownRing({ remaining, total }) {
  const pct = total > 0 ? Math.max(0, Math.min(100, (remaining / total) * 100)) : 0;
  const color = pct > 50 ? '#f59e0b' : pct > 20 ? '#f97316' : '#ef4444';
  return (
    <span
      title={`${fmtRemaining(remaining)} remaining`}
      style={{
        width: 18, height: 18, borderRadius: '50%', display: 'inline-block',
        background: `conic-gradient(${color} ${pct}%, #e5e7eb ${pct}%)`,
        flexShrink: 0,
      }}
    />
  );
}

export default function RoutingBadge({ alertId, routing, onRoutingChange }) {
  const { authFetch } = useAuth();
  const connectionState = useConnectionState();
  const queueItem = useAckQueueItem(alertId);
  const [elapsed, setElapsed] = useState(null);   // null until server-seeded
  const [acking, setAcking] = useState(false);
  const [queued, setQueued] = useState(false);
  const timerRef = useRef(null);

  const status = routing?.status || 'PENDING';
  const timeout = routing?.timeout_seconds ?? 120;
  const isActive = ACTIVE_STATES.includes(status);

  // Seed from server truth whenever the routing record changes identity or
  // the server hands us a fresh seconds_elapsed (mount, reload, reconnect).
  useEffect(() => {
    if (routing && typeof routing.seconds_elapsed === 'number') {
      setElapsed(routing.seconds_elapsed);
    }
  }, [routing?.routed_alert_id, routing?.seconds_elapsed, routing?.assigned_at]);

  // Display-only interpolation. Stopped the moment the alert leaves an
  // active state, so an ACK'd or escalated card is not still ticking.
  useEffect(() => {
    if (!isActive || elapsed === null) {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      return undefined;
    }
    timerRef.current = setInterval(() => setElapsed(e => (e === null ? e : e + 1)), 1000);
    return () => {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    };
  }, [isActive, elapsed === null]);

  const ack = useCallback(async () => {
    setAcking(true);
    try {
      const res = await authFetch(`${API}/alerts/${alertId}/ack`, { method: 'POST' });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        // Both 'acknowledged' and 'already_acknowledged' are successes —
        // the alert is acknowledged either way.
        if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
        onRoutingChange?.({ ...routing, status: 'ACKNOWLEDGED' });
      } else if (res.status === 409) {
        // Should be unreachable: the button is hidden for the states that
        // return 409. Defensive no-op rather than a user-facing error.
        console.warn('[RoutingBadge] unexpected 409 on ACK:', data);
      } else {
        // 5xx — same non-network-drop-but-still-lost-tap case as below.
        enqueueAck(alertId);
        setQueued(true);
      }
    } catch (err) {
      // Connection dropped — queue instead of the tap silently vanishing.
      console.warn('[RoutingBadge] ACK failed, queued for retry:', err);
      enqueueAck(alertId);
      setQueued(true);
    } finally {
      setAcking(false);
    }
  }, [alertId, authFetch, routing, onRoutingChange]);

  // Fold the outbox's outcome in — covers this badge mounting fresh onto an
  // alert that was queued from a previous mount/tab session too.
  useEffect(() => {
    if (!queueItem) return;
    if (queueItem.status === 'synced') {
      setQueued(false);
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      onRoutingChange?.({ ...routing, status: 'ACKNOWLEDGED' });
      dismissAckQueueItem(queueItem.id);
    } else if (queueItem.status === 'failed') {
      setQueued(false);
      console.warn('[RoutingBadge] queued ACK could not sync:', queueItem.lastError);
      dismissAckQueueItem(queueItem.id);
    } else {
      setQueued(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- onRoutingChange/routing
    // intentionally excluded: this must react to the QUEUE's state, not
    // re-fire whenever the parent hands down a new routing object reference.
  }, [queueItem?.status, queueItem?.id, queueItem?.lastError]);

  useEffect(() => {
    if (connectionState === 'CONNECTED') flushAckQueue();
  }, [connectionState]);

  if (!routing) return null;

  const pill = (bg, fg, border, extra = {}) => ({
    display: 'inline-flex', alignItems: 'center', gap: 7,
    background: bg, color: fg, border: `1px solid ${border}`,
    borderRadius: 999, padding: '3px 11px', fontSize: 12, fontWeight: 600,
    ...extra,
  });

  if (status === 'PENDING') {
    return <span style={pill('#f3f4f6', '#6b7280', '#d1d5db')}>Routing…</span>;
  }

  if (status === 'ACKNOWLEDGED') {
    return (
      <span style={pill('rgba(16,185,129,0.12)', '#10b981', 'rgba(16,185,129,0.4)')}>
        ✓ ACK'd{routing.officer_name ? ` by ${routing.officer_name}` : ''}
        {routing.ack_at ? ` at ${fmtClock(routing.ack_at)}` : ''}
      </span>
    );
  }

  if (NO_OFFICER_STATES.includes(status)) {
    const escalated = status === 'ESCALATED_UNROUTED';
    return (
      <span
        className={escalated ? 'routing-pulse' : undefined}
        style={pill('rgba(239,68,68,0.12)', '#ef4444', 'rgba(239,68,68,0.45)')}
      >
        ⚠ {escalated ? 'Escalated — No officer available' : 'No officer available'}
      </span>
    );
  }

  // ROUTED / ESCALATED_ROUTED
  const remaining = Math.max(0, timeout - (elapsed ?? 0));
  const escalated = status === 'ESCALATED_ROUTED';
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
      <span style={pill(
        escalated ? 'rgba(249,115,22,0.12)' : 'rgba(245,158,11,0.12)',
        escalated ? '#f97316' : '#f59e0b',
        escalated ? 'rgba(249,115,22,0.45)' : 'rgba(245,158,11,0.45)',
      )}>
        <CountdownRing remaining={remaining} total={timeout} />
        {escalated ? 'Escalated → ' : ''}{routing.officer_name || `Officer #${routing.officer_id}`}
        {elapsed === null ? '' : ` · ${fmtRemaining(remaining)} remaining`}
      </span>
      <button
        onClick={ack}
        disabled={acking || queued}
        title={queued ? 'Queued — will send automatically when reconnected' : undefined}
        style={{
          padding: '4px 12px', borderRadius: 6, border: '1px solid var(--border)',
          background: queued ? 'var(--accent-yellow, #f59e0b)'
            : acking ? 'var(--bg-elevated)' : 'var(--accent-green)',
          color: acking && !queued ? 'var(--text-muted)' : '#fff',
          fontSize: 11, fontWeight: 700,
          cursor: (acking || queued) ? 'not-allowed' : 'pointer',
        }}
      >
        {queued ? '⏳ Queued' : acking ? '…' : 'ACK'}
      </button>
    </span>
  );
}
