// frontend/src/components/SentinelIQCard.jsx
//
// Live SENTINEL IQ score card — one entry per camera with an active alert.
//
// Every number rendered here comes from the server, which summed it from
// Alert.iq_contribution values stored at fire time. This component does NOT
// recompute anything: it does not multiply base scores by multipliers, it
// does not sum contributions client-side, and it does not render a
// placeholder number while loading. If the server hasn't answered yet, the
// card says so — a number that appears before the data arrives is a number
// someone might act on.
//
// Live updates reuse the Day 8 alert_status_update broadcast plus the Day 5
// new_alert broadcast: both bump a refetch, so dismissing an alert in one tab
// drops it out of the aggregate in every other tab without a refresh.
import { useState, useEffect, useCallback } from 'react';
import { useAuth, API } from '../context/AuthContext';

const WINDOW_OPTIONS = [15, 60, 240, 1440];

function fmt(n) {
  // Trim trailing zeroes: 3.9 not 3.90, 4.68 not 4.6800.
  return Number(n).toFixed(2).replace(/\.?0+$/, '');
}

function scoreClass(score) {
  if (score >= 10) return 'badge-high';
  if (score >= 5) return 'badge-medium';
  return 'badge-low';
}

// One "BASE × Mult × Mult = TOTAL" line, rendered from the stored breakdown.
// The '=' total is the server's stored figure, NOT the product of the parts
// re-multiplied here — if those two ever disagreed, this would show the
// disagreement rather than paper over it.
function BreakdownLine({ contribution }) {
  const b = contribution.breakdown;
  if (!b) {
    return (
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
        {contribution.alert_type.replace(/_/g, ' ')} · {fmt(contribution.iq_contribution)}
        <span style={{ marginLeft: 6, fontStyle: 'italic' }}>
          (no stored breakdown — alert predates the scoring engine)
        </span>
      </div>
    );
  }

  const applied = (b.multipliers || []).filter(m => m.applied);
  const skipped = (b.multipliers || []).filter(m => !m.applied);

  return (
    <div style={{ padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
      <div style={{ fontSize: 12, fontFamily: 'monospace' }}>
        <strong>{b.alert_type.replace(/_/g, ' ')}</strong> {fmt(b.base_score)}
        {applied.map(m => (
          <span key={m.name}> × {m.name} {fmt(m.factor)}</span>
        ))}
        {' = '}
        <strong>{fmt(b.total)}</strong>
      </div>

      {applied.map(m => (
        <div key={m.name} style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 10 }}>
          ↳ {m.name}: {m.reason}
          {m.detail?.night_source && (
            <span style={{ marginLeft: 6 }}>
              · night_source: <code>{m.detail.night_source}</code>
            </span>
          )}
          {m.detail?.zones && <span style={{ marginLeft: 6 }}>· zones: {m.detail.zones.join(', ')}</span>}
        </div>
      ))}

      {/* Skipped multipliers are shown, not hidden. "Why is this 3.0 and not
          3.9" is exactly the question an officer will ask about a score. */}
      {skipped.map(m => (
        <div key={m.name} style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 10, opacity: 0.7 }}>
          ↳ {m.name}: not applied (<code>{m.reason}</code>) — 1.0×
        </div>
      ))}

      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 10, marginTop: 2 }}>
        alert #{contribution.alert_id} · {contribution.lifecycle_status}
        {contribution.subject_label ? ` · ${contribution.subject_label}` : ''}
      </div>
    </div>
  );
}

function CameraScoreRow({ camera }) {
  const [open, setOpen] = useState(false);

  return (
    <div style={{
      border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
      background: 'var(--bg-elevated)', marginBottom: 8, overflow: 'hidden',
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 12, width: '100%',
          padding: '12px 14px', background: 'none', border: 'none',
          color: 'inherit', cursor: 'pointer', textAlign: 'left',
        }}
      >
        <span className={`badge ${scoreClass(camera.total_score)}`} style={{ fontSize: 15, padding: '5px 12px' }}>
          🧠 {fmt(camera.total_score)}
        </span>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 600 }}>{camera.camera_name || `Camera #${camera.camera_id}`}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {camera.camera_zone || 'No zone'} · {camera.alert_count} scored alert{camera.alert_count === 1 ? '' : 's'}
            {camera.unscored_count > 0 && ` · ${camera.unscored_count} unscored`}
          </div>
        </div>
        <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{open ? '▾ Hide' : '▸ Breakdown'}</span>
      </button>

      {open && (
        <div style={{ padding: '4px 14px 12px' }}>
          {camera.contributions.length === 0 ? (
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              No scored alerts contributing in this window.
            </div>
          ) : (
            camera.contributions.map(c => <BreakdownLine key={c.alert_id} contribution={c} />)
          )}
          {camera.unscored_count > 0 && (
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 8, fontStyle: 'italic' }}>
              {camera.unscored_count} active alert{camera.unscored_count === 1 ? '' : 's'} in this window
              carr{camera.unscored_count === 1 ? 'ies' : 'y'} no IQ score and contribute nothing to the
              total — they are not counted as zero.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function NightModeControl({ nightMode, onChanged }) {
  const { authFetch, isAdmin } = useAuth();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const label = nightMode?.override === null || nightMode?.override === undefined
    ? 'Automatic (clock)'
    : nightMode.override ? 'Forced ON' : 'Forced OFF';

  const set = async (enabled) => {
    setBusy(true);
    setError(null);
    try {
      const res = await authFetch(`${API}/admin/night-mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        setError(d.detail || 'Failed to change night mode.');
        return;
      }
      onChanged();
    } catch {
      setError('Network error.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>🌙 Night multiplier (1.3×):</span>
      <span className={`badge ${nightMode?.mode === 'manual_override' ? 'badge-medium' : 'badge-low'}`}>
        {label}
      </span>
      {isAdmin && (
        <>
          <button className="btn btn-secondary" style={{ fontSize: 11, padding: '4px 9px' }}
            disabled={busy} onClick={() => set(true)}>Force ON</button>
          <button className="btn btn-secondary" style={{ fontSize: 11, padding: '4px 9px' }}
            disabled={busy} onClick={() => set(false)}>Force OFF</button>
          <button className="btn btn-secondary" style={{ fontSize: 11, padding: '4px 9px' }}
            disabled={busy} onClick={() => set(null)}>Auto</button>
        </>
      )}
      {error && <span style={{ color: 'var(--accent-red)', fontSize: 11 }}>{error}</span>}
    </div>
  );
}

export default function SentinelIQCard({ liveAlert, statusEvent }) {
  const { authFetch } = useAuth();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [windowMinutes, setWindowMinutes] = useState(60);

  const fetchScores = useCallback(async () => {
    try {
      const res = await authFetch(`${API}/iq/cameras?window_minutes=${windowMinutes}`);
      if (!res.ok) {
        setError('Failed to load Sarvanetra IQ scores.');
        return;
      }
      setData(await res.json());
      setError(null);
    } catch {
      setError('Network error loading Sarvanetra IQ scores.');
    } finally {
      setLoading(false);
    }
  }, [authFetch, windowMinutes]);

  useEffect(() => { fetchScores(); }, [fetchScores]);

  // A new alert changes the aggregate; so does any lifecycle transition from
  // any operator (a dismissal removes that alert's contribution). Both are
  // already broadcast on the dashboard socket, so no polling is needed.
  useEffect(() => { if (liveAlert) fetchScores(); }, [liveAlert, fetchScores]);
  useEffect(() => { if (statusEvent) fetchScores(); }, [statusEvent, fetchScores]);

  if (loading) {
    return (
      <div className="empty-state">
        <span className="empty-state-icon">⏳</span>
        Loading Sarvanetra IQ scores…
      </div>
    );
  }

  if (error) {
    return <div className="empty-state"><span className="empty-state-icon">⚠️</span>{error}</div>;
  }

  const cameras = data?.cameras || [];

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, gap: 12, flexWrap: 'wrap' }}>
        <div className="card-title">
          Sarvanetra IQ
          <span style={{ color: 'var(--text-muted)', fontWeight: 400, fontSize: 12, marginLeft: 6 }}>
            ({cameras.length} camera{cameras.length === 1 ? '' : 's'} with active alerts)
          </span>
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Window:</span>
          {WINDOW_OPTIONS.map(w => (
            <button key={w}
              className={`btn ${windowMinutes === w ? 'btn-danger' : 'btn-secondary'}`}
              style={{ fontSize: 11, padding: '4px 9px' }}
              onClick={() => setWindowMinutes(w)}>
              {w >= 60 ? `${w / 60}h` : `${w}m`}
            </button>
          ))}
          <button className="btn btn-secondary" style={{ fontSize: 11, padding: '4px 9px' }}
            onClick={fetchScores}>↻</button>
        </div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <NightModeControl nightMode={data?.night_mode} onChanged={fetchScores} />
      </div>

      {cameras.length === 0 ? (
        <div className="empty-state">
          <span className="empty-state-icon">🧠</span>
          No camera has an active scored alert in the last {windowMinutes >= 60 ? `${windowMinutes / 60}h` : `${windowMinutes} min`}.
          <div style={{ fontSize: 11, marginTop: 6, color: 'var(--text-muted)' }}>
            Cameras with nothing to score are omitted rather than shown at 0 —
            a zero here would read as "assessed and clear", which is not what it means.
          </div>
        </div>
      ) : (
        cameras.map(c => <CameraScoreRow key={c.camera_id} camera={c} />)
      )}

      {/* Roadmap triggers: label only, never a number — not even 0. */}
      {data?.roadmap_triggers?.length > 0 && (
        <div style={{
          marginTop: 16, padding: '10px 14px', borderRadius: 'var(--radius-md)',
          border: '1px dashed var(--border)', background: 'transparent',
        }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, fontWeight: 600 }}>
            Not scored — no detector exists yet
          </div>
          {data.roadmap_triggers.map(t => (
            <div key={t.alert_type} style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              <span className="badge" style={{ marginRight: 6, opacity: 0.6 }}>
                {t.alert_type.replace(/_/g, ' ')}
              </span>
              <em>{t.status}</em>
            </div>
          ))}
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6 }}>
            These carry no score of any kind, including zero. Nothing in this system
            fires them, so "0" would claim a check that never ran.
          </div>
        </div>
      )}

      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 14, lineHeight: 1.5 }}>
        Every figure above is summed from per-alert scores stored at fire time —
        nothing is recomputed in the browser. Multipliers not built (high-crime
        zone, school/hospital proximity, prior-flag history) are absent rather
        than estimated: there is no GIS or incident-history source in this system
        to compute them from.
      </div>
    </div>
  );
}
