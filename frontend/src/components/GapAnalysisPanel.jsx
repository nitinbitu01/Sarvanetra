// frontend/src/components/GapAnalysisPanel.jsx
//
// Frontend surface for GET /cameras/gap-analysis — the Model 1 deliverable
// "sample gap-analysis report". The endpoint existed and was tested since
// earlier work this project, but nothing in the UI ever called it; it was
// reachable only by hitting the API directly.
import { useState, useCallback, useEffect } from 'react';
import { useAuth, API } from '../context/AuthContext';

function GapSection({ title, count, pct, note, cameras, tone = 'warn', extra }) {
  const [open, setOpen] = useState(false);
  if (count === 0) {
    return (
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '8px 12px', borderRadius: 6, background: 'rgba(16,185,129,0.08)',
        border: '1px solid rgba(16,185,129,0.25)', fontSize: 12,
      }}>
        <span>{title}</span>
        <span style={{ color: '#10b981', fontWeight: 600 }}>✓ none</span>
      </div>
    );
  }
  const color = tone === 'info' ? '#38bdf8' : '#f59e0b';
  return (
    <div style={{
      borderRadius: 6, border: `1px solid ${color}55`, background: `${color}14`,
      overflow: 'hidden',
    }}>
      <button type="button" onClick={() => setOpen((o) => !o)} style={{
        width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '8px 12px', background: 'none', border: 'none', cursor: 'pointer',
        color: 'inherit', fontSize: 12, textAlign: 'left',
      }}>
        <span>{open ? '▾' : '▸'} {title}</span>
        <span style={{ color, fontWeight: 700 }}>
          {count}{pct != null ? ` (${pct}%)` : ''}
        </span>
      </button>
      {open && (
        <div style={{ padding: '0 12px 10px', fontSize: 11.5 }}>
          {note && <div style={{ color: 'var(--text-muted)', marginBottom: 6 }}>{note}</div>}
          {extra}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 180, overflowY: 'auto' }}>
            {cameras.map((c) => (
              <div key={c.id} style={{
                display: 'flex', justifyContent: 'space-between', gap: 8,
                padding: '3px 0', borderTop: '1px solid rgba(255,255,255,0.06)',
              }}>
                <span>{c.name}</span>
                <span style={{ color: 'var(--text-muted)' }}>
                  {c.department || 'no dept'} · {c.zone || 'no zone'}
                  {c.installed_at ? ` · installed ${c.installed_at.slice(0, 10)}` : ''}
                  {c.last_checked_minutes_ago != null ? ` · ${c.last_checked_minutes_ago}m ago` : ''}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function GapAnalysisPanel() {
  const { authFetch } = useAuth();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await authFetch(`${API}/cameras/gap-analysis`);
      if (!res.ok) throw new Error(`${res.status}`);
      setReport(await res.json());
    } catch {
      setError('Could not load gap-analysis report.');
    } finally {
      setLoading(false);
    }
  }, [authFetch]);

  useEffect(() => { load(); }, [load]);

  const exportCsv = async () => {
    const res = await authFetch(`${API}/cameras/export`);
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const disposition = res.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="?([^"]+)"?/);
    a.download = match ? match[1] : 'camera_registry.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="card" style={{ maxWidth: 700, marginTop: 20 }}>
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="card-title">🕳️ Registry Gap Analysis</div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button type="button" className="btn btn-secondary" onClick={load} disabled={loading}>
            {loading ? '⏳' : '↻'} Refresh
          </button>
          <button type="button" className="btn btn-secondary" onClick={exportCsv}>
            ⬇ Export CSV
          </button>
        </div>
      </div>

      {error && <div className="test-result error">{error}</div>}

      {report && !report.gaps && (
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{report.note}</div>
      )}

      {report && report.gaps && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {report.total_cameras} camera(s) in scope
            {report.scope ? ` (department: ${report.scope})` : ' (state-wide)'}
            {' · generated '}{new Date(report.generated_at).toLocaleString()}
          </div>

          <GapSection title="Missing GPS" tone="warn"
            count={report.gaps.missing_gps.count} pct={report.gaps.missing_gps.pct}
            cameras={report.gaps.missing_gps.cameras} />
          <GapSection title="Missing department" tone="warn"
            count={report.gaps.missing_department.count} pct={report.gaps.missing_department.pct}
            note={report.gaps.missing_department.note}
            cameras={report.gaps.missing_department.cameras} />
          <GapSection title="Offline" tone="warn"
            count={report.gaps.offline.count} pct={report.gaps.offline.pct}
            note={report.gaps.offline.note}
            cameras={report.gaps.offline.cameras} />
          <GapSection title="Never health-checked" tone="warn"
            count={report.gaps.never_health_checked.count}
            cameras={report.gaps.never_health_checked.cameras} />
          <GapSection title="Stale health data" tone="warn"
            count={report.gaps.stale_health.count}
            note={`Not re-checked in over ${report.gaps.stale_health.threshold_minutes} minutes.`}
            cameras={report.gaps.stale_health.cameras} />
          <GapSection title="Ageing infrastructure" tone="warn"
            count={report.gaps.ageing_infrastructure.count} pct={report.gaps.ageing_infrastructure.pct}
            note={`Installed more than ${report.gaps.ageing_infrastructure.threshold_years} years ago.`}
            cameras={report.gaps.ageing_infrastructure.cameras}
            extra={report.gaps.ageing_infrastructure.unknown_install_date.count > 0 && (
              <div style={{
                marginBottom: 8, padding: '6px 8px', borderRadius: 4,
                background: 'rgba(148,163,184,0.1)', color: 'var(--text-muted)',
              }}>
                Plus {report.gaps.ageing_infrastructure.unknown_install_date.count} camera(s) with no
                recorded install date — age cannot be assessed for these.
              </div>
            )} />
          <GapSection title="Under maintenance" tone="info"
            count={report.under_maintenance.count}
            note="Deliberately flagged, not a gap — shown for visibility only."
            cameras={report.under_maintenance.cameras} />

          {Object.keys(report.zones.camera_count_by_zone).length > 0 && (
            <div style={{ marginTop: 6 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                Cameras by zone
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {Object.entries(report.zones.camera_count_by_zone).map(([zone, n]) => (
                  <span key={zone} style={{
                    fontSize: 11, padding: '2px 8px', borderRadius: 10,
                    background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  }}>
                    {zone}: {n}
                  </span>
                ))}
              </div>
              {report.zones.zones_with_gps_gaps.length > 0 && (
                <div style={{ fontSize: 11, color: '#f59e0b', marginTop: 6 }}>
                  Zones with unplaced cameras: {report.zones.zones_with_gps_gaps.join(', ')}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
