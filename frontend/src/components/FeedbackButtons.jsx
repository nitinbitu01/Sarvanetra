// frontend/src/components/FeedbackButtons.jsx
import React, { useState, useEffect } from 'react';
import { getToken } from '../utils/auth';

const API = import.meta.env.VITE_API_URL || '/api/v1';

const VERDICTS = [
  { key: 'GENUINE',       label: '✅ Genuine',      color: '#22c55e' },
  { key: 'FALSE_ALARM',   label: '❌ False Alarm',   color: '#ef4444' },
  { key: 'INVESTIGATING', label: '🔍 Investigating', color: '#f59e0b' },
];

export function FeedbackButtons({ alertId, initialVerdict = null, onFeedback }) {
  const [selected, setSelected] = useState(initialVerdict);
  const [loading,  setLoading]  = useState(false);

  useEffect(() => {
    if (initialVerdict) {
      setSelected(initialVerdict);
    }
  }, [initialVerdict]);

  async function handleVerdict(verdict) {
    if (loading)              return;
    if (selected === verdict) return;

    setLoading(true);

    try {
      const token = getToken();
      const res = await fetch(`${API}/alerts/${alertId}/feedback`, {
        method:  'POST',
        headers: {
          'Content-Type':  'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ verdict }),
      });

      if (res.ok) {
        const data = await res.json();
        setSelected(verdict);
        onFeedback?.(data);
      } else {
        const err = await res.json().catch(() => ({}));
        console.error('[FeedbackButtons] Server error:', res.status, err);
      }
    } catch (e) {
      console.error('[FeedbackButtons] Network error:', e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="feedback-buttons" role="group" aria-label="Alert verdict" style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
      {VERDICTS.map(({ key, label, color }) => (
        <button
          key={key}
          onClick={() => handleVerdict(key)}
          disabled={loading}
          aria-pressed={selected === key}
          style={{
            backgroundColor: selected === key ? color        : 'transparent',
            border:           `2px solid ${color}`,
            color:            selected === key ? '#ffffff'   : color,
            borderRadius:     6,
            padding:          '6px 12px',
            fontSize:         12,
            fontWeight:       600,
            opacity:          loading         ? 0.6          : 1,
            cursor:           loading         ? 'wait'       : 'pointer',
            transition:       'all 0.15s ease',
          }}
        >
          {loading && selected === key ? 'Saving…' : label}
        </button>
      ))}
    </div>
  );
}

export default FeedbackButtons;
