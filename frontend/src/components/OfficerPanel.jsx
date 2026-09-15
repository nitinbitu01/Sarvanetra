// frontend/src/components/OfficerPanel.jsx
//
// Collapsible roster showing who is available, busy, or offline.
//
// The list is rendered in the order the server returns it (ORDER BY name
// ASC). Do not re-sort by status here: officers change status constantly
// during a demo, and sorting on a mutating field makes rows jump around the
// panel on every WebSocket event, which reads as a UI glitch.
import { useState, useEffect, useCallback } from 'react';
import { useAuth, API } from '../context/AuthContext';

const DOT = {
  AVAILABLE: 'var(--accent-green)',
  BUSY: 'var(--accent-red)',
  OFFLINE: 'var(--text-muted)',
};

export default function OfficerPanel({ refreshSignal, referenceCamera }) {
  const { authFetch } = useAuth();
  const [officers, setOfficers] = useState([]);
  const [open, setOpen] = useState(true);
  const [busyId, setBusyId] = useState(null);

  const load = useCallback(async () => {
    try {
      const res = await authFetch(`${API}/officers`);
      if (res.ok) setOfficers(await res.json());
    } catch {
      /* advisory panel — a failed poll must not break the dashboard */
    }
  }, [authFetch]);

  useEffect(() => { load(); }, [load, refreshSignal]);

  const markAvailable = async (id) => {
    setBusyId(id);
    try {
      const res = await authFetch(`${API}/officers/${id}/available`, { method: 'POST' });
      if (res.ok) {
        // Update inline rather than re-fetching: the officer.status event
        // this triggers will reconcile every other panel anyway.
        setOfficers(prev => prev.map(o =>
          o.id === id ? { ...o, status: 'AVAILABLE', current_alert_id: null } : o
        ));
      }
    } catch {
      /* ignore — next refresh reconciles */
    } finally {
      setBusyId(null);
    }
  };

  // Distance is display-only. Computed client-side against the last alerted
  // camera purely so an operator can sanity-check the dispatch choice; the
  // actual routing decision is made server-side with haversine_km.
  const distanceLabel = (o) => {
    if (!referenceCamera || referenceCamera.lat == null || referenceCamera.lng == null) {
      return '—';
    }
    const R = 6371;
    const dLat = (o.lat - referenceCamera.lat) * Math.PI / 180;
    const dLng = (o.lng - referenceCamera.lng) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2
      + Math.cos(referenceCamera.lat * Math.PI / 180) * Math.cos(o.lat * Math.PI / 180)
      * Math.sin(dLng / 2) ** 2;
    const d = R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return d < 0.05 ? '< 50 m' : `${d.toFixed(2)} km`;
  };

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          width: '100%', background: 'none', border: 'none', color: 'inherit',
          cursor: 'pointer', padding: 0,
        }}
      >
        <span className="card-title">
          👮 Officers
          <span style={{ color: 'var(--text-muted)', fontWeight: 400, fontSize: 12, marginLeft: 6 }}>
            ({officers.filter(o => o.status === 'AVAILABLE').length} available / {officers.length})
          </span>
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{open ? '▾' : '▸'}</span>
      </button>

      {open && (
        <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {officers.length === 0 && (
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No officers registered.</div>
          )}
          {officers.map(o => (
            <div key={o.id} style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '7px 10px', borderRadius: 'var(--radius-md)',
              background: 'var(--bg-elevated)', border: '1px solid var(--border)',
            }}>
              <span style={{
                width: 9, height: 9, borderRadius: '50%',
                background: DOT[o.status] || 'var(--text-muted)', flexShrink: 0,
              }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 13, fontWeight: 600 }}>{o.name}</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {o.status}
                  {o.status === 'BUSY' && o.current_alert_id
                    ? ` · Assigned to Alert #${o.current_alert_id}`
                    : ''}
                  {' · '}{distanceLabel(o)}
                </div>
              </div>
              {(o.status === 'BUSY' || o.status === 'OFFLINE') && (
                <button
                  onClick={() => markAvailable(o.id)}
                  disabled={busyId === o.id}
                  style={{
                    padding: '3px 9px', borderRadius: 6,
                    border: '1px solid var(--border)', background: 'var(--bg-elevated)',
                    color: 'var(--text-primary)', fontSize: 11,
                    cursor: busyId === o.id ? 'not-allowed' : 'pointer',
                  }}
                >
                  {busyId === o.id ? '…' : 'Mark Available'}
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
