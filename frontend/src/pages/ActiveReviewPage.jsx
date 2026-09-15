// frontend/src/pages/ActiveReviewPage.jsx
// Full-page wrapper for the officer HITL review workflow.
// Lists pending review items from GET /review/queue and opens
// OfficerReviewModal when an officer clicks "Begin Review".
// Uses existing AuthContext — no Zustand.

import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { useWebSocketEvent } from '../context/WebSocketContext';
import OfficerReviewModal from '../components/review/OfficerReviewModal';

const API = import.meta.env.VITE_API_URL || '/api/v1';

function QueueCard({ item, onBeginReview }) {
  const ageMin = item.age_seconds
    ? Math.round(item.age_seconds / 60)
    : null;

  const urgency = item.similarity_score < 0.65
    ? { color: '#ef4444', label: '🔴 Low Confidence', bg: 'rgba(239,68,68,0.08)', border: '#7f1d1d' }
    : item.similarity_score < 0.85
    ? { color: '#f59e0b', label: '🟡 Borderline',    bg: 'rgba(245,158,11,0.08)', border: '#92400e' }
    : { color: '#22c55e', label: '🟢 High Confidence', bg: 'rgba(34,197,94,0.08)', border: '#166534' };

  return (
    <div style={{
      background: 'var(--bg-elevated, #1f2937)',
      border: `1px solid ${urgency.border}`,
      borderRadius: 14,
      padding: '16px 20px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: 14,
      flexWrap: 'wrap',
    }}
      role="listitem"
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 700, fontSize: 14 }}>Alert #{item.alert_id}</span>
          <span style={{
            fontSize: 11, fontWeight: 600, padding: '1px 8px', borderRadius: 20,
            background: urgency.bg, color: urgency.color, border: `1px solid ${urgency.border}`,
          }}>
            {urgency.label}
          </span>
          {item.environmental_condition && (
            <span style={{
              fontSize: 11, padding: '1px 8px', borderRadius: 20,
              background: 'rgba(99,102,241,0.1)', color: '#818cf8', border: '1px solid #4338ca',
            }}>
              {item.environmental_condition.replace('_', ' ')}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', gap: 16, fontSize: 12, color: 'var(--text-muted)' }}>
          {item.camera_name && <span>📷 {item.camera_name}</span>}
          {item.zone && <span>📍 {item.zone}</span>}
          {ageMin !== null && <span>⏱ {ageMin}m old</span>}
        </div>
      </div>

      <button
        onClick={() => onBeginReview(item)}
        id={`review-btn-${item.alert_id}`}
        style={{
          padding: '10px 20px',
          background: 'rgba(99,102,241,0.15)',
          border: '1px solid #4338ca',
          borderRadius: 10, color: '#818cf8',
          fontWeight: 700, fontSize: 13,
          cursor: 'pointer',
          whiteSpace: 'nowrap',
        }}
      >
        🔍 Begin Review
      </button>
    </div>
  );
}

export default function ActiveReviewPage() {
  const { authFetch } = useAuth();
  const [queue, setQueue]               = useState([]);
  const [loadState, setLoadState]       = useState('loading');
  const [activeAlert, setActiveAlert]   = useState(null);
  const [modalOpen, setModalOpen]       = useState(false);
  const [lastUpdated, setLastUpdated]   = useState(null);
  const pollRef                         = useRef(null);

  const fetchQueue = useCallback(async () => {
    try {
      const res = await authFetch(`${API}/review/queue`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setQueue(Array.isArray(data) ? data : data?.items ?? []);
      setLoadState('success');
      setLastUpdated(new Date());
    } catch {
      if (loadState !== 'success') setLoadState('error');
    }
  }, [authFetch]);

  useEffect(() => {
    fetchQueue();
    pollRef.current = setInterval(fetchQueue, 30_000);
    return () => clearInterval(pollRef.current);
  }, [fetchQueue]);

  useWebSocketEvent('reid_review_created', () => fetchQueue());
  useWebSocketEvent('review_submitted', () => fetchQueue());

  const handleBeginReview = (item) => {
    setActiveAlert(item);
    setModalOpen(true);
  };

  const handleModalClose = () => {
    setModalOpen(false);
    setActiveAlert(null);
  };

  const handleReviewSubmitted = () => {
    // Remove completed item from queue optimistically
    if (activeAlert) {
      setQueue(prev => prev.filter(i => i.alert_id !== activeAlert.alert_id));
    }
    fetchQueue(); // Sync with server
  };

  const pending = queue.filter(i => i.status !== 'COMPLETED');

  return (
    <div style={{ padding: 24, maxWidth: 900, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>🔍 Active Review Queue</h1>
          {loadState === 'success' && (
            <span style={{
              padding: '2px 12px', borderRadius: 20, fontSize: 12, fontWeight: 600,
              background: pending.length > 0 ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.15)',
              color: pending.length > 0 ? '#ef4444' : '#22c55e',
              border: `1px solid ${pending.length > 0 ? '#991b1b' : '#166534'}`,
            }}>
              {pending.length > 0 ? `${pending.length} Pending` : '✓ All Clear'}
            </span>
          )}
        </div>
        {lastUpdated && (
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            Updated {lastUpdated.toLocaleTimeString('en-IN')}
          </span>
        )}
      </div>

      {/* Instructions card */}
      <div style={{
        padding: 16, background: 'rgba(99,102,241,0.06)', border: '1px solid #4338ca',
        borderRadius: 12, marginBottom: 20, fontSize: 13, color: 'var(--text-muted)',
        lineHeight: 1.6,
      }}>
        <strong style={{ color: '#818cf8' }}>How the review gate works:</strong>
        {' '}Click "Begin Review" to open a secure review session.
        You must view both images for at least <strong>4 seconds</strong> before
        the decision buttons unlock — this prevents rubber-stamping.
        The AI confidence score is hidden until after you decide.
      </div>

      {/* Loading */}
      {loadState === 'loading' && (
        <div role="status" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {[1, 2, 3].map(i => (
            <div key={i} style={{ height: 80, background: '#1f2937', borderRadius: 14, animation: 'pulse 1.5s infinite' }} />
          ))}
        </div>
      )}

      {/* Error */}
      {loadState === 'error' && (
        <div role="alert" style={{ textAlign: 'center', padding: 60 }}>
          <p style={{ color: '#ef4444', fontWeight: 600, fontSize: 16 }}>⚠️ Failed to load review queue</p>
          <button className="btn-primary" onClick={fetchQueue} style={{ marginTop: 14 }}>Retry</button>
        </div>
      )}

      {/* Empty */}
      {loadState === 'success' && pending.length === 0 && (
        <div style={{ textAlign: 'center', padding: 80 }}>
          <div style={{ fontSize: 64, marginBottom: 14 }}>✅</div>
          <p style={{ color: '#f9fafb', fontWeight: 700, fontSize: 20, margin: '0 0 8px' }}>
            No Pending Reviews
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: 14, margin: 0 }}>
            All borderline detections have been reviewed.
          </p>
        </div>
      )}

      {/* Queue Items */}
      {pending.length > 0 && (
        <div role="list" aria-label="Pending review items" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {pending.map(item => (
            <QueueCard
              key={item.alert_id ?? item.id}
              item={item}
              onBeginReview={handleBeginReview}
            />
          ))}
        </div>
      )}

      {/* Officer Review Modal */}
      <OfficerReviewModal
        alert={activeAlert}
        isOpen={modalOpen}
        onClose={handleModalClose}
        onReviewSubmitted={handleReviewSubmitted}
      />
    </div>
  );
}
