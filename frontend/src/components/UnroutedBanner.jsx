// frontend/src/components/UnroutedBanner.jsx
//
// Sticky banner for alerts that reached nobody.
//
// DISMISSAL IS PER-ALERT, NOT PER-SESSION
//   localStorage "sentineliq.dismissed_unrouted" holds an ARRAY of
//   routed_alert_id values, not a boolean. A boolean flag would mean
//   dismissing one unrouted alert also hides the next one — which is the
//   failure mode that matters here: an operator silences a banner about
//   alert #7 and never learns that #12 also reached nobody.
//
//   On mount the stored array is pruned to IDs that are still unrouted, so
//   it cannot grow without bound across a long-running shift.
import { useState, useEffect, useCallback } from 'react';
import { useAuth, API } from '../context/AuthContext';

const STORAGE_KEY = 'sentineliq.dismissed_unrouted';

function readDismissed() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    return Array.isArray(raw) ? raw.filter(n => Number.isInteger(n)) : [];
  } catch {
    return [];
  }
}

function writeDismissed(ids) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
  } catch {
    /* quota or private mode — dismissal simply won't persist */
  }
}

import Icon from './Icon';

export default function UnroutedBanner({ refreshSignal }) {
  const { authFetch } = useAuth();
  const [items, setItems] = useState([]);
  const [dismissed, setDismissed] = useState(readDismissed);
  const [expanded, setExpanded] = useState(false);

  const load = useCallback(async () => {
    try {
      const res = await authFetch(`${API}/routing/unrouted`);
      if (!res.ok) return;
      const data = await res.json();
      setItems(data);

      const live = new Set(data.map(d => d.routed_alert_id));
      setDismissed(prev => {
        const pruned = prev.filter(id => live.has(id));
        if (pruned.length !== prev.length) writeDismissed(pruned);
        return pruned;
      });
    } catch {
      // advisory
    }
  }, [authFetch]);

  useEffect(() => { load(); }, [load, refreshSignal]);

  const visible = items.filter(i => !dismissed.includes(i.routed_alert_id));

  const dismissOne = (id) => {
    setDismissed(prev => {
      const next = [...new Set([...prev, id])];
      writeDismissed(next);
      return next;
    });
  };

  const dismissAll = () => {
    const allIds = items.map(i => i.routed_alert_id);
    setDismissed(allIds);
    writeDismissed(allIds);
  };

  if (visible.length === 0) {
    return (
      <div
        className="c2-alert-bar"
        style={{
          borderLeft: '3px solid var(--status-success, #10B981)',
          background: 'rgba(16, 185, 129, 0.06)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11.5, color: 'var(--text-secondary, #9CA3AF)' }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--status-success, #10B981)' }} />
          <span style={{ fontWeight: 600, color: 'var(--text-primary, #F9FAFB)', letterSpacing: '0.4px' }}>
            ALL DISPATCH CHANNELS ACTIVE
          </span>
          <span style={{ color: 'var(--text-muted, #6B7280)' }}>— 0 unrouted alerts pending</span>
        </div>
        <div style={{
          fontFamily: 'var(--font-mono, monospace)',
          fontSize: 10,
          color: 'var(--status-success, #10B981)',
          background: 'rgba(16, 185, 129, 0.12)',
          border: '1px solid rgba(16, 185, 129, 0.25)',
          padding: '2px 7px',
          borderRadius: 2,
        }}>
          OPERATIONAL
        </div>
      </div>
    );
  }

  return (
    <div style={{ position: 'sticky', top: 0, zIndex: 40, marginBottom: 6, flexShrink: 0 }}>
      <div className="c2-alert-bar">
        <div className="c2-alert-bar-left">
          <Icon name="triangleAlert" size={15} style={{ color: 'var(--status-critical, #DC2626)' }} />
          <span>
            {visible.length} alert{visible.length === 1 ? '' : 's'} unrouted — no officer available
          </span>
        </div>

        <div className="c2-alert-bar-actions">
          <button
            type="button"
            className="c2-alert-btn"
            onClick={() => setExpanded(e => !e)}
          >
            {expanded ? 'Hide Details' : 'Show Details'}
          </button>
          <button
            type="button"
            className="c2-alert-btn"
            onClick={dismissAll}
          >
            Dismiss All
          </button>
        </div>
      </div>

      {expanded && (
        <div style={{
          background: 'var(--bg-elevated, #111827)',
          border: '1px solid var(--border-default, #374151)',
          borderTop: 'none',
          borderRadius: '0 0 var(--radius-sm, 2px) var(--radius-sm, 2px)',
          padding: '8px 12px',
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
          marginBottom: 6,
        }}>
          {visible.map(i => (
            <div key={i.routed_alert_id} style={{
              display: 'flex', alignItems: 'center', gap: 10, fontSize: 11.5,
              color: 'var(--text-secondary, #9CA3AF)',
              fontFamily: 'var(--font-mono, monospace)',
            }}>
              <span style={{ fontWeight: 600, color: 'var(--text-primary, #F9FAFB)' }}>ALERT #{i.alert_id}</span>
              <span style={{ color: 'var(--text-muted, #6B7280)' }}>
                {i.created_at ? new Date(i.created_at).toLocaleTimeString() : ''}
              </span>
              <span style={{ color: 'var(--status-critical, #DC2626)', fontSize: 10, fontWeight: 700 }}>
                {i.status || 'UNROUTED'}
              </span>
              <button
                type="button"
                className="c2-alert-btn"
                onClick={() => dismissOne(i.routed_alert_id)}
                style={{ marginLeft: 'auto', padding: '1px 6px', fontSize: 10 }}
              >
                Dismiss
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
