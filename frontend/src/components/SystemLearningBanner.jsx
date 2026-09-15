// frontend/src/components/SystemLearningBanner.jsx
import React, { useEffect, useCallback, useState } from 'react';
import { useWebSocketEvent } from '../utils/useWebSocketEvent';

const API = import.meta.env.VITE_API_URL || '/api/v1';

const SYSTEM_LEARNING_DISCLAIMER =
  "⚠ Cosmetic signal only — detection behavior unchanged";

export function SystemLearningBanner() {
  const [flags, setFlags] = useState([]);

  const fetchSummary = useCallback(async () => {
    try {
      const res = await fetch(`${API}/feedback/summary?since=today`);
      if (!res.ok) return;
      const data = await res.json();
      setFlags((data.summary || []).filter(z => z.threshold_flag));
    } catch (e) {
      console.error('[SystemLearningBanner] Failed to fetch summary:', e);
    }
  }, []);

  useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  const handleFeedbackEvent = useCallback((event) => {
    if (event.threshold_flag) fetchSummary();
  }, [fetchSummary]);

  useWebSocketEvent('feedback.logged', handleFeedbackEvent);

  if (flags.length === 0) return null;

  return (
    <div
      className="system-learning-banner"
      role="alert"
      style={{
        background: 'rgba(239, 68, 68, 0.15)',
        border: '1px solid #ef4444',
        borderRadius: 8,
        padding: '10px 14px',
        marginBottom: 12,
        color: '#f87171',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700 }}>
        <span className="icon" aria-hidden="true">🧠</span>
        <span>System Learning Flagged</span>
      </div>
      {flags.map(flag => (
        <div key={flag.zone_name} className="flag-item" style={{ fontSize: 13, color: '#fca5a5' }}>
          • {flag.flag_message}
        </div>
      ))}
      <span className="disclaimer" style={{ fontSize: 11, opacity: 0.8, marginTop: 2 }}>
        {SYSTEM_LEARNING_DISCLAIMER}
      </span>
    </div>
  );
}

export default SystemLearningBanner;
