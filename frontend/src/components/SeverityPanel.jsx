// frontend/src/components/SeverityPanel.jsx
//
// Severity counts for today, seeded from the server and incremented live.
import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth, API } from '../context/AuthContext';
import { useWebSocketEvent } from '../context/WebSocketContext';
import { PanelSkeleton } from './PanelErrorBoundary';

const CONFIG = {
  CRITICAL: { color: 'var(--status-critical, #DC2626)', label: 'CRITICAL' },
  HIGH:     { color: 'var(--status-warning, #F59E0B)', label: 'HIGH' },
  MEDIUM:   { color: 'var(--status-info, #3B82F6)', label: 'MEDIUM' },
  LOW:      { color: 'var(--status-success, #10B981)', label: 'LOW' },
};

export default function SeverityPanel() {
  const { authFetch } = useAuth();
  const [counts, setCounts] = useState(null);
  const [unscored, setUnscored] = useState(0);
  const [failed, setFailed] = useState(false);
  const fetchedAt = useRef(null);

  const load = useCallback(async () => {
    try {
      const res = await authFetch(`${API}/alerts/summary?since=today`);
      if (!res || !res.ok) throw new Error('summary');
      const d = await res.json();
      fetchedAt.current = Date.now();
      setCounts({
        CRITICAL: d.CRITICAL || 0, HIGH: d.HIGH || 0,
        MEDIUM: d.MEDIUM || 0, LOW: d.LOW || 0,
      });
      setUnscored(d.UNSCORED || 0);
      setFailed(false);
    } catch {
      setFailed(true);
    }
  }, [authFetch]);

  useEffect(() => {
    load();
    const interval = setInterval(load, 12000);
    return () => clearInterval(interval);
  }, [load]);

  const bump = useCallback((evt) => {
    if (!fetchedAt.current) return;
    const sev = String(evt?.severity || 'CRITICAL').toUpperCase();
    if (!CONFIG[sev]) return;
    setCounts((prev) => (prev ? { ...prev, [sev]: (prev[sev] || 0) + 1 } : prev));
  }, []);

  useWebSocketEvent('alert.routed', bump);
  useWebSocketEvent('alert.unrouted', bump);
  useWebSocketEvent('new_alert', useCallback((a) => {
    const sev = String(a?.severity || '').toUpperCase();
    if (sev && sev !== 'CRITICAL') bump(a);
  }, [bump]));

  if (failed) throw new Error('Fetch failed');
  if (!counts) return <PanelSkeleton lines={4} />;

  return (
    <div
      className="panel-content"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        padding: '10px 12px',
        height: '100%',
      }}
    >
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 8,
          flex: 1,
        }}
      >
        {Object.entries(CONFIG).map(([sev, cfg]) => (
          <div
            key={sev}
            style={{
              background: 'var(--bg-elevated, #111827)',
              border: '1px solid var(--border, #1F2937)',
              borderRadius: 'var(--radius-md, 4px)',
              padding: '12px 14px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
            }}
          >
            <div
              style={{
                fontFamily: 'var(--font-mono, monospace)',
                fontSize: 26,
                fontWeight: 600,
                color: cfg.color,
                lineHeight: 1.1,
              }}
            >
              {counts[sev] ?? 0}
            </div>
            <div
              style={{
                fontFamily: 'var(--font-sans, sans-serif)',
                fontSize: 10.5,
                color: cfg.color,
                marginTop: 4,
                fontWeight: 600,
                letterSpacing: '0.8px',
              }}
            >
              {cfg.label}
            </div>
          </div>
        ))}
      </div>
      {unscored > 0 && (
        <div
          style={{
            fontSize: 10,
            color: 'var(--text-muted, #6B7280)',
            fontFamily: 'var(--font-mono, monospace)',
          }}
        >
          +{unscored} unscored telemetry events
        </div>
      )}
    </div>
  );
}
