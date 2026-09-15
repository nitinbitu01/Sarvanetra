// frontend/src/components/AlertDetailCard.jsx
//
// Full-screen CRITICAL alert, opened by a notification tap or a foreground
// push. Designed for a phone held at arm's length: one photo, one score, one
// button.
//
// All data comes from ONE call to /alerts/{id}/detail. This card frequently
// opens on a cold start over a mobile connection; three parallel fetches
// would mean three chances to hang and a card that renders in pieces.
import { useState, useEffect, useCallback } from 'react';
import { useAuth, API } from '../context/AuthContext';
import { useConnectionState } from '../context/WebSocketContext';
import { useAckQueueItem } from '../hooks/useAckQueue';
import { enqueueAck, dismiss as dismissAckQueueItem, flush as flushAckQueue } from '../utils/ackQueue';
import { speakCriticalAlert, flushPendingTTS } from '../utils/tts';
import SafeHlsPlayer from './SafeHlsPlayer';
import FeedbackButtons from './FeedbackButtons';
import SystemLearningBanner from './SystemLearningBanner';
import ProofClipModal from './ProofClipModal';

export default function AlertDetailCard({ alertId, onClose }) {
  const { authFetch } = useAuth();
  const connectionState = useConnectionState();
  const queueItem = useAckQueueItem(alertId);
  const [alert, setAlert] = useState(null);
  const [error, setError] = useState(null);
  const [ackState, setAckState] = useState('idle');
  const [queueError, setQueueError] = useState(null);
  const [showProofModal, setShowProofModal] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await authFetch(`${API}/alerts/${alertId}/detail`);
        if (cancelled) return;
        if (!res.ok) {
          setError(res.status === 404 ? 'Alert not found.' : 'Could not load alert.');
          return;
        }
        const data = await res.json();
        setAlert(data);
        if (data.routing?.ack_at) setAckState('already_acknowledged');
        // Fired after data lands so the camera name can be spoken. On iOS
        // this is likely to be blocked (no user gesture yet on a
        // notification-launched app) — tts.js queues it for the first tap
        // rather than blocking the card.
        speakCriticalAlert(data.camera_name);
      } catch {
        if (!cancelled) setError('Network error loading alert.');
      }
    })();
    return () => { cancelled = true; };
  }, [alertId, authFetch]);

  const isAcked = ackState === 'acknowledged' || ackState === 'already_acknowledged';
  const isQueued = ackState === 'queued';

  const handleAck = useCallback(async () => {
    if (isAcked || isQueued || ackState === 'loading') return;
    setAckState('loading');
    try {
      const res = await authFetch(`${API}/alerts/${alertId}/ack`, { method: 'POST' });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        // Day 14 returns 'acknowledged' or 'already_acknowledged'; both mean
        // the alert is handled, so both land in the same terminal UI state.
        setAckState(data.status === 'already_acknowledged'
          ? 'already_acknowledged' : 'acknowledged');
      } else if (res.status === 409) {
        // A live response saying no — never routed, nothing to acknowledge.
        // Distinct from a dropped connection: this is a real-time answer
        // from the server, not silence, so it must not be queued and retried.
        setAckState('cannot_ack');
      } else {
        // 5xx or similar — same treatment as a network failure below: the
        // tap must not just vanish back to an unclicked button.
        enqueueAck(alertId);
        setAckState('queued');
      }
    } catch {
      // fetch() threw — the connection is down. This is the case Day 17
      // exists for: queue the tap instead of silently discarding it.
      enqueueAck(alertId);
      setAckState('queued');
    }
  }, [alertId, authFetch, isAcked, isQueued, ackState]);

  // Fold the outbox's outcome into ackState. Runs even if THIS mount never
  // called handleAck — e.g. the officer tapped ACKNOWLEDGE, closed the card
  // while offline, and reopened it before the queue flushed; the item is
  // still in utils/ackQueue's in-memory (and, where supported, IndexedDB)
  // store and this picks it straight up rather than showing a blank button
  // that invites a second, duplicate tap.
  useEffect(() => {
    if (!queueItem) return;
    if (queueItem.status === 'synced') {
      setAckState('acknowledged');
      dismissAckQueueItem(queueItem.id);
    } else if (queueItem.status === 'failed') {
      setQueueError(queueItem.lastError);
      setAckState('queue_failed');
      dismissAckQueueItem(queueItem.id);
    } else {
      setAckState('queued');
    }
  }, [queueItem?.status, queueItem?.id, queueItem?.lastError]);

  // Opportunistic flush on top of ackQueue's own 'online'/visibilitychange
  // backstops — catches a WebSocket reconnect on a connection flaky enough
  // that the browser never actually fired a native 'offline'/'online' pair.
  useEffect(() => {
    if (connectionState === 'CONNECTED') flushAckQueue();
  }, [connectionState]);

  // Any tap replays a TTS utterance the iOS gesture rule blocked earlier.
  const onCardClick = () => flushPendingTTS();

  const overlay = {
    position: 'fixed', inset: 0, zIndex: 9999,
    background: 'var(--bg-primary, #0f172a)',
    display: 'flex', flexDirection: 'column',
    padding: 'env(safe-area-inset-top, 16px) 16px env(safe-area-inset-bottom, 16px)',
    overflowY: 'auto',
  };

  if (error) {
    return (
      <div style={overlay} onClick={onCardClick}>
        <div style={{ color: 'var(--accent-red)', fontSize: 15, marginTop: 40 }}>{error}</div>
        <button onClick={onClose} style={btn('var(--bg-elevated)', 'var(--text-primary)')}>
          Close
        </button>
      </div>
    );
  }

  if (!alert) {
    return (
      <div style={overlay}>
        <div style={{ color: 'var(--text-muted)', marginTop: 40 }}>Loading alert…</div>
      </div>
    );
  }

  const assigned = alert.routing?.officer_name;

  return (
    <div style={overlay} onClick={onCardClick}>
      <SystemLearningBanner />

      <div style={{
        background: 'var(--accent-red)', color: '#fff', fontWeight: 800,
        fontSize: 16, letterSpacing: 0.5, padding: '12px 14px',
        borderRadius: 10, textAlign: 'center', marginBottom: 14,
      }}>
        ⚠ CRITICAL ALERT
      </div>

      {String(alert.severity || '').toUpperCase() === 'CRITICAL' && alert.camera_id ? (
        <div style={{ marginBottom: 14 }}>
          <SafeHlsPlayer cameraId={alert.camera_id} autoPlay={true} />
        </div>
      ) : alert.crop_url ? (
        <img
          src={alert.crop_url}
          alt="Detection"
          style={{
            width: '100%', maxHeight: '38vh', objectFit: 'cover',
            borderRadius: 10, border: '1px solid var(--border)', marginBottom: 14,
          }}
        />
      ) : (
        <div style={{
          height: 140, borderRadius: 10, marginBottom: 14,
          border: '1px dashed var(--border-bright)', display: 'flex',
          alignItems: 'center', justifyContent: 'center',
          color: 'var(--text-muted)', fontSize: 13,
        }}>
          No image available
        </div>
      )}

      <div style={{ marginBottom: 6, fontSize: 22, fontWeight: 800 }}>
        {alert.camera_name || 'Unknown camera'}
      </div>
      {(alert.location_label || alert.camera_location) && (
        <div style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 10 }}>
          📍 {alert.location_label || alert.camera_location}
        </div>
      )}

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 10 }}>
        {alert.score != null && (
          <div style={{ fontSize: 14 }}>
            Score: <strong style={{ color: 'var(--accent-red)' }}>{alert.score}</strong>
          </div>
        )}
        <div style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
          {alert.created_at ? new Date(alert.created_at).toLocaleString() : ''}
        </div>
      </div>

      {alert.subject_label && (
        <div style={{ fontSize: 15, fontWeight: 700, color: '#f8fafc', marginBottom: 6 }}>
          🎯 {alert.subject_label}
        </div>
      )}

      {alert.description && (
        <div style={{
          fontSize: 13, color: '#cbd5e1', lineHeight: 1.5,
          background: 'rgba(30, 41, 59, 0.6)', border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: 8, padding: '10px 12px', marginBottom: 12,
        }}>
          <strong style={{ color: '#93c5fd' }}>Reason / FIR Context:</strong> {alert.description}
        </div>
      )}

      <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
        {assigned
          ? <>Assigned to <strong>{alert.routing.officer_name}</strong></>
          : <span style={{ color: 'var(--accent-yellow)' }}>
              ⚠ Not assigned to an officer
            </span>}
      </div>

      {/* Feedback buttons */}
      <div style={{ marginTop: 10, marginBottom: 14 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Officer Feedback / Ground Truth:</div>
        <FeedbackButtons
          alertId={alertId}
          initialVerdict={alert.feedback?.verdict ?? null}
          onFeedback={(payload) => {
            setAlert(prev => prev ? { ...prev, feedback: { ...prev.feedback, verdict: payload.verdict } } : prev);
          }}
        />
      </div>

      <button
        onClick={handleAck}
        disabled={isAcked || isQueued || ackState === 'loading' || ackState === 'cannot_ack'}
        style={{
          ...btn(
            isAcked ? 'var(--accent-green)'
              : isQueued ? 'var(--accent-yellow, #f59e0b)'
              : 'var(--accent-red)',
            '#fff',
          ),
          fontSize: 18, padding: '18px 0', fontWeight: 800,
          opacity: ackState === 'cannot_ack' ? 0.5 : 1,
        }}
      >
        {isAcked ? '✓ Acknowledged'
          : isQueued ? '⏳ Queued — will send when back online'
          : ackState === 'loading' ? 'Sending…'
          : ackState === 'cannot_ack' ? 'Cannot acknowledge — not routed'
          : 'ACKNOWLEDGE'}
      </button>

      {ackState === 'queue_failed' && (
        // Terminal, not transient: the server gave a real 409 once the
        // queued tap finally reached it (e.g. the alert was reassigned while
        // this device was offline). Retrying would just get the same
        // rejection, so this offers Close, not "try again".
        <div style={{
          marginTop: 10, padding: '10px 12px', borderRadius: 8,
          background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.4)',
          color: 'var(--accent-red)', fontSize: 13,
        }}>
          Could not sync your acknowledgement: {queueError || 'alert state changed while offline.'}
        </div>
      )}

      <button
        onClick={() => setShowProofModal(true)}
        style={{
          ...btn('rgba(59, 130, 246, 0.15)', '#60a5fa'),
          marginTop: 12, fontSize: 14, fontWeight: 700,
          border: '1px solid rgba(59, 130, 246, 0.35)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
        }}
      >
        <span>🎬</span> View CCTV Video Proof Clip
      </button>

      <button
        onClick={onClose}
        style={{ ...btn('transparent', 'var(--text-muted)'), marginTop: 14, fontSize: 13 }}
      >
        Close
      </button>

      {showProofModal && (
        <ProofClipModal alert={alert} alertId={alertId} onClose={() => setShowProofModal(false)} />
      )}
    </div>
  );
}

function btn(bg, color) {
  return {
    width: '100%', border: 'none', borderRadius: 10, background: bg,
    color, cursor: 'pointer', padding: '12px 0',
  };
}
