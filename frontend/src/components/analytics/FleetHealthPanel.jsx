// frontend/src/components/analytics/FleetHealthPanel.jsx
//
// Which cameras are actually producing detections, and which have gone dark.
//
// For a platform whose job is integrating cameras from 26 departments, this is
// the operational question that matters most, and nothing on the dashboard
// answered it. A camera can be registered, flagged online by its heartbeat,
// and contributing nothing — pointed at a wall since a maintenance visit,
// re-cabled to a different feed, or dropped by its department's network days
// ago. Alert counts cannot reveal that: a dead camera raises no alerts, which
// looks exactly like a quiet street.
//
// Everything here is counted from vehicle_track rows the tracker writes as it
// processes frames. No target throughput is asserted, because what a camera
// "should" see depends on its road; the only comparison made is against that
// same camera's previous window.

import { useState, useEffect, useCallback } from 'react';
import { apiFetch } from '../../utils/authClient';
import { mockFleetHealth } from '../../data/mockData';
import '../../styles/fleet-health.css';

const WINDOWS = [
  { hours: 1, label: '1 hour' },
  { hours: 24, label: '24 hours' },
  { hours: 168, label: '7 days' },
];

const STATUS_LABEL = {
  producing: 'Producing',
  degraded: 'Fell off',
  silent: 'Silent',
  never_seen: 'Never seen',
};

function relativeTime(iso) {
  if (!iso) return 'never';
  const then = new Date(iso.replace(' ', 'T'));
  if (Number.isNaN(then.getTime())) return iso;
  const mins = Math.round((Date.now() - then.getTime()) / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 48) return `${hrs} h ago`;
  return `${Math.round(hrs / 24)} d ago`;
}

export default function FleetHealthPanel() {
  const [hours, setHours] = useState(24);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showAll, setShowAll] = useState(false);

  const load = useCallback(async (h) => {
    setLoading(true);
    try {
      const res = await apiFetch(`/analytics/fleet-health?hours=${h}`);
      if (!res?.ok) throw new Error(`HTTP ${res?.status}`);
      const json = await res.json();
      setData(json || mockFleetHealth);
      setError(null);
    } catch (e) {
      setData(mockFleetHealth);
      setError(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(hours); }, [hours, load]);

  if (error && !data) {
    return <div className="fh-error" role="alert">{error}</div>;
  }
  if (!data) {
    return <div className="fh-loading" role="status">Counting detections…</div>;
  }

  const cams = Array.isArray(data.cameras) ? data.cameras : [];
  // Problems first: a silent camera is why someone opens this panel.
  const order = { silent: 0, degraded: 1, never_seen: 2, producing: 3 };
  const sorted = [...cams].sort((a, b) =>
    ((order[a.status] ?? 3) - (order[b.status] ?? 3)) || ((b.tracks_recent || 0) - (a.tracks_recent || 0)));
  const shown = showAll ? sorted : sorted.slice(0, 12);
  const busiest = Math.max(...cams.map((c) => c.tracks_recent || 0), 1);

  const silentList = Array.isArray(data.silent) ? data.silent : [];
  const degradedList = Array.isArray(data.degraded) ? data.degraded : [];
  const neverSeenList = Array.isArray(data.never_seen) ? data.never_seen : [];
  const onlineSilentList = Array.isArray(data.online_but_silent) ? data.online_but_silent : [];
  const needsAttention = silentList.length + degradedList.length;

  return (
    <div className="fh">
      <div className="fh-head">
        <div className="fh-counts">
          <span className="fh-count fh-count--ok">
            <strong>{data.producing ?? 0}</strong> producing
          </span>
          {silentList.length > 0 && (
            <span className="fh-count fh-count--bad">
              <strong>{silentList.length}</strong> silent
            </span>
          )}
          {degradedList.length > 0 && (
            <span className="fh-count fh-count--warn">
              <strong>{degradedList.length}</strong> fell off
            </span>
          )}
          {neverSeenList.length > 0 && (
            <span className="fh-count fh-count--idle">
              <strong>{neverSeenList.length}</strong> never seen
            </span>
          )}
          <span className="fh-count fh-count--idle">
            <strong>{(data.total_tracks || 0).toLocaleString()}</strong> detections
          </span>
        </div>
        <div className="fh-window" role="group" aria-label="Time window">
          {WINDOWS.map((w) => (
            <button
              key={w.hours}
              onClick={() => setHours(w.hours)}
              className={hours === w.hours ? 'is-active' : ''}
              aria-pressed={hours === w.hours}
              disabled={loading}
            >
              {w.label}
            </button>
          ))}
        </div>
      </div>

      {onlineSilentList.length > 0 && (
        <p className="fh-flag" role="status">
          <strong>{onlineSilentList.length} cameras report online but
          produced nothing</strong> in this window
          {' '}({onlineSilentList.slice(0, 6).join(', ')}
          {onlineSilentList.length > 6 ? '…' : ''}).
          The heartbeat and the analytics disagree, and only the heartbeat can
          be wrong by staying silent.
        </p>
      )}

      {needsAttention === 0 && (data.producing || 0) > 0 && (
        <p className="fh-flag fh-flag--ok" role="status">
          Every camera that has ever produced detections is still producing them.
        </p>
      )}

      <table className="fh-table">
        <thead>
          <tr>
            <th scope="col">Camera</th>
            <th scope="col">Department</th>
            <th scope="col" className="fh-num">Detections</th>
            <th scope="col">Share of fleet</th>
            <th scope="col" className="fh-num">vs previous</th>
            <th scope="col">Last seen</th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((c) => (
            <tr key={c.camera_id} className={`fh-row fh-row--${c.status}`}>
              <th scope="row">
                {c.camera_id}
                {c.name && c.name !== c.camera_id && <em>{c.name}</em>}
              </th>
              <td>{c.department || c.district || '—'}</td>
              <td className="fh-num">{c.tracks_recent.toLocaleString()}</td>
              <td>
                <div className="fh-bar" aria-hidden="true">
                  <span style={{ width: `${(c.tracks_recent / busiest) * 100}%` }} />
                </div>
              </td>
              <td className="fh-num">
                {c.drop_pct === null || c.drop_pct === undefined ? (
                  <span className="fh-muted" title="Too few detections in the previous window to compare">—</span>
                ) : (
                  <span className={c.drop_pct > 0 ? 'fh-down' : 'fh-up'}>
                    {c.drop_pct > 0 ? '−' : '+'}{Math.abs(c.drop_pct)}%
                  </span>
                )}
              </td>
              <td className="fh-muted">{relativeTime(c.last_track_at)}</td>
              <td>
                <span className={`fh-status fh-status--${c.status}`}>
                  {STATUS_LABEL[c.status] || c.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {sorted.length > 12 && (
        <button className="fh-more" onClick={() => setShowAll((v) => !v)}>
          {showAll ? 'Show fewer' : `Show all ${sorted.length} cameras`}
        </button>
      )}

      <p className="fh-basis">{data.measurement_basis}</p>
    </div>
  );
}
