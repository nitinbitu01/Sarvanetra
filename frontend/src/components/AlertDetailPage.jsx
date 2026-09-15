// frontend/src/components/AlertDetailPage.jsx
import React, { useState, useEffect } from 'react';
import { getToken } from '../utils/auth';
import { SafeHlsPlayer } from './SafeHlsPlayer';
import { FeedbackButtons } from './FeedbackButtons';
import { SystemLearningBanner } from './SystemLearningBanner';

const API = import.meta.env.VITE_API_URL || '/api/v1';

export function AlertDetailPage({ alertId, onBack }) {
  const [alert, setAlert] = useState(null);
  const [zoneSummary, setZoneSummary] = useState(null);

  useEffect(() => {
    if (!alertId) return;
    const token = getToken();
    fetch(`${API}/alerts/${alertId}/detail`, {
      headers: token ? { 'Authorization': `Bearer ${token}` } : {},
    })
      .then(r => r.ok ? r.json() : null)
      .then(data => setAlert(data))
      .catch(() => {});
  }, [alertId]);

  function handleFeedback(payload) {
    setZoneSummary(payload.zone_summary);
    setAlert(prev => prev ? { ...prev, feedback: { ...prev.feedback, verdict: payload.verdict } } : prev);
  }

  if (!alert) return <div style={{ padding: 24, color: 'var(--text-muted)' }}>Loading alert #{alertId}…</div>;

  return (
    <div className="alert-detail-page" style={{ padding: 20, maxWidth: 640, margin: '0 auto' }}>
      {onBack && (
        <button
          onClick={onBack}
          style={{
            background: 'transparent',
            border: 'none',
            color: 'var(--accent-blue, #3b82f6)',
            cursor: 'pointer',
            padding: 0,
            marginBottom: 14,
            fontSize: 13,
          }}
        >
          ← Back to alerts
        </button>
      )}

      <SystemLearningBanner />

      {/* Stream — for CRITICAL alerts with a configured camera */}
      {String(alert.severity || '').toUpperCase() === 'CRITICAL' && alert.camera_id && (
        <div style={{ marginBottom: 16 }}>
          <SafeHlsPlayer
            cameraId={alert.camera_id}
            autoPlay={true}
          />
        </div>
      )}

      {/* Alert details */}
      <h2 style={{ margin: '0 0 8px 0', fontSize: 20 }}>
        {alert.alert_type} — <span style={{ color: 'var(--accent-red)' }}>{alert.severity}</span>
      </h2>
      <p style={{ margin: '0 0 4px 0', color: 'var(--text-secondary)' }}>
        📷 Camera: <strong>{alert.camera_name || alert.camera_id}</strong>
      </p>
      <p style={{ margin: '0 0 16px 0', color: 'var(--text-secondary)' }}>
        📍 Zone: <strong>{alert.location_label || alert.camera_location || 'General Zone'}</strong>
      </p>

      {/* Feedback buttons */}
      <div style={{ background: 'var(--bg-elevated)', padding: 14, borderRadius: 8, border: '1px solid var(--border)' }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>
          Ground Truth Classification:
        </div>
        <FeedbackButtons
          alertId={alertId}
          initialVerdict={alert.feedback?.verdict ?? null}
          onFeedback={handleFeedback}
        />
      </div>

      {/* Inline zone summary after feedback */}
      {zoneSummary?.threshold_flag && (
        <div
          className="inline-flag-notice"
          style={{
            marginTop: 14,
            padding: '10px 14px',
            background: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid #ef4444',
            borderRadius: 8,
            color: '#f87171',
            fontSize: 13,
          }}
        >
          ⚠ {zoneSummary.flag_message}
        </div>
      )}
    </div>
  );
}

export default AlertDetailPage;
