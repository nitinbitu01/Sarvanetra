// frontend/src/components/ReviewQueueBadge.jsx
//
// Pending human-review count. Seeded from the server, nudged live.
import { useState, useEffect, useCallback } from 'react';
import { useAuth, API } from '../context/AuthContext';
import { useWebSocketEvent } from '../context/WebSocketContext';
import { PanelSkeleton } from './PanelErrorBoundary';

export default function ReviewQueueBadge({ onOpen }) {
  const { authFetch } = useAuth();
  const [pending, setPending] = useState(null);
  const [failed, setFailed] = useState(false);

  const load = useCallback(async () => {
    try {
      const res = await authFetch(`${API}/reid/review-queue/count`);
      if (!res.ok) throw new Error('count');
      const d = await res.json();
      setPending(d.pending ?? 0);
      setFailed(false);
    } catch {
      setFailed(true);
    }
  }, [authFetch]);

  useEffect(() => { load(); }, [load]);

  // Re-fetch rather than incrementing locally. The badge is a single integer
  // that is cheap to read, and an approve/reject from another operator's tab
  // decrements it — a local counter would drift out of sync with no way to
  // notice. The event is the trigger; the server is the value.
  useWebSocketEvent('reid_review_created', useCallback(() => { load(); }, [load]));

  if (failed) throw new Error('Review queue count unavailable');
  if (pending === null) return <PanelSkeleton lines={2} />;

  const urgent = pending > 0;

  return (
    <div
      className="panel-content"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        padding: '12px 14px',
        height: '100%',
      }}
    >
      <div
        style={{
          fontFamily: 'var(--font-mono, monospace)',
          fontSize: 40,
          fontWeight: 600,
          lineHeight: 1,
          color: urgent ? 'var(--status-warning, #F59E0B)' : 'var(--text-primary, #F9FAFB)',
        }}
      >
        {pending}
      </div>
      <div
        style={{
          fontFamily: 'var(--font-sans, sans-serif)',
          fontSize: 11,
          color: 'var(--text-secondary, #9CA3AF)',
          textAlign: 'center',
          fontWeight: 500,
        }}
      >
        {pending === 1 ? 'match awaiting review' : 'matches awaiting review'}
      </div>
      {urgent && (
        <button
          onClick={() => onOpen?.()}
          style={{
            marginTop: 4,
            padding: '4px 12px',
            borderRadius: 'var(--radius-sm, 2px)',
            fontSize: 11,
            fontWeight: 600,
            border: 'none',
            background: 'var(--status-warning, #F59E0B)',
            color: '#000000',
            cursor: 'pointer',
          }}
        >
          Open Queue
        </button>
      )}
      <div
        style={{
          fontSize: 10,
          color: 'var(--text-muted, #6B7280)',
          textAlign: 'center',
          marginTop: 2,
          lineHeight: 1.3,
        }}
      >
        Officer review gate (65–85% confidence)
      </div>
    </div>
  );
}
