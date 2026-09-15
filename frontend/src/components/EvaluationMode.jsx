// frontend/src/components/EvaluationMode.jsx
//
// One screen for the technical evaluation's central task: an evaluator hands
// over a designated vehicle registration number, and this answers "where has
// it been" completely — route on GIS, timestamped location-wise movement
// history, watchlist cross-referencing, and the alerts raised — without
// navigating anywhere else.
//
// Every number shown comes from an endpoint that measured it. Where the
// backend ships an explicit caveat about how a figure was derived
// (confidence_basis, distance_basis, straight-line vs road distance) it is
// rendered next to the figure rather than dropped — a route summary that
// hides its own basis is the thing an evaluator is entitled to distrust.
//
// Data sources, all real:
//   GET  /journeys?mode=vehicle&search=<reg>   route, stops, legs, GIS points
//   GET  /plate-search/watchlist               is this vehicle watchlisted
//   POST /plate-search/watchlist               put it on the watchlist live
//   GET  /alerts                               alerts already raised for it
import { useState, useCallback, useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useAuth, API } from '../context/AuthContext';

// divIcon, not an image marker: bundled Leaflet marker PNGs need a separate
// copy-to-public build step and vanish silently if it is missed (the same
// reason LiveMap.jsx documents for its own icons).
function numberedIcon(n, isFirst, isLast) {
  const bg = isFirst ? '#10b981' : isLast ? '#ef4444' : '#3b82f6';
  return L.divIcon({
    className: '',
    html: `<div style="width:26px;height:26px;border-radius:50%;background:${bg};
           color:#fff;font:700 12px/26px system-ui,sans-serif;text-align:center;
           border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.5)">${n}</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
  });
}

/** Zoom to the whole route. Leaflet measures its container at init, which in a
 *  grid cell can still be 0px, so this also nudges a resize. */
function FitRoute({ points }) {
  const map = useMap();
  useEffect(() => {
    const t = setTimeout(() => {
      map.invalidateSize();
      if (points && points.length) {
        map.fitBounds(L.latLngBounds(points), { padding: [40, 40], maxZoom: 15 });
      }
    }, 120);
    return () => clearTimeout(t);
  }, [map, points]);
  return null;
}

const fmtTime = (iso) => {
  if (!iso) return '—';
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
};
const num = (v, suffix = '') =>
  (v === null || v === undefined || Number.isNaN(v)) ? '—' : `${v}${suffix}`;

function Stat({ label, value, tone }) {
  return (
    <div style={{
      padding: '8px 14px', borderRadius: 8, background: 'var(--bg-elevated)',
      border: '1px solid var(--border)', minWidth: 108,
    }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase',
                    letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: 17, fontWeight: 700, color: tone || 'var(--text-primary)' }}>
        {value}
      </div>
    </div>
  );
}

export default function EvaluationMode() {
  const { authFetch } = useAuth();
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searched, setSearched] = useState(false);
  const [journeys, setJourneys] = useState([]);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [watchlist, setWatchlist] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [addingWatch, setAddingWatch] = useState(false);
  const [watchMsg, setWatchMsg] = useState('');

  const journey = journeys[selectedIdx] || null;

  const normalise = (s) => (s || '').toUpperCase().replace(/[^A-Z0-9]/g, '');

  const loadWatchlist = useCallback(async () => {
    try {
      const r = await authFetch(`${API}/plate-search/watchlist`);
      if (r.ok) setWatchlist((await r.json()).entries || []);
    } catch { /* watchlist status is supplementary to the trace */ }
  }, [authFetch]);

  useEffect(() => { loadWatchlist(); }, [loadWatchlist]);

  // `plateOverride` exists so a suggestion chip can trace immediately. Reading
  // `query` after setQuery() would use the previous render's value and trace
  // the wrong plate.
  const runTrace = useCallback(async (e, plateOverride) => {
    e?.preventDefault?.();
    const q = (plateOverride ?? query).trim();
    if (!q) return;
    if (plateOverride) setQuery(plateOverride);
    setLoading(true);
    setError('');
    setWatchMsg('');
    setSelectedIdx(0);
    try {
      const r = await authFetch(
        `${API}/journeys?mode=vehicle&search=${encodeURIComponent(q)}`);
      if (!r.ok) throw new Error(`Search failed (${r.status})`);
      const data = await r.json();
      setJourneys(data.journeys || []);
      setSearched(true);

      // Alerts already raised for this vehicle — the "alerts & history" half
      // of the evaluation. Filtered client-side because /alerts has no plate
      // filter; the feed is capped and this keeps to one request.
      try {
        const ar = await authFetch(`${API}/alerts?limit=100`);
        if (ar.ok) {
          const all = await ar.json();
          const rows = Array.isArray(all) ? all : (all.alerts || []);
          const qn = normalise(q);
          setAlerts(rows.filter((a) =>
            normalise(a.subject_label).includes(qn)
            || normalise(a.plate_text).includes(qn)));
        }
      } catch { setAlerts([]); }
    } catch (err) {
      setError(err.message || 'Search failed');
      setJourneys([]);
      setSearched(true);
    } finally {
      setLoading(false);
    }
  }, [authFetch, query]);

  const watchEntry = useMemo(() => {
    if (!journey) return null;
    const target = normalise(journey.reid_id);
    return watchlist.find((w) => normalise(w.plate) === target) || null;
  }, [watchlist, journey]);

  const addToWatchlist = async () => {
    if (!journey) return;
    setAddingWatch(true);
    setWatchMsg('');
    try {
      const r = await authFetch(`${API}/plate-search/watchlist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plate: journey.reid_id,
          reason: 'Designated vehicle — technical evaluation',
          category: 'suspect',
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `Failed (${r.status})`);
      setWatchMsg(`✅ ${data.plate} is on the watchlist — the live pipeline starts cross-referencing it within a few seconds.`);
      loadWatchlist();
    } catch (err) {
      setWatchMsg(`❌ ${err.message}`);
    } finally {
      setAddingWatch(false);
    }
  };

  const stops = journey?.stops || [];
  const routePoints = (journey?.route_points || []).filter(
    (p) => Array.isArray(p) && p.length === 2 && p[0] != null && p[1] != null);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* ── Search ─────────────────────────────────────────────────────── */}
      <form onSubmit={runTrace} className="card" style={{ padding: 16 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ flex: '1 1 280px' }}>
            <label className="form-label">Designated vehicle registration number</label>
            <input
              className="form-input"
              placeholder="e.g. GJ03O0301"
              value={query}
              onChange={(e) => setQuery(e.target.value.toUpperCase())}
              style={{ fontFamily: 'var(--font-mono)', fontSize: 18, letterSpacing: 1.5 }}
              autoFocus
            />
          </div>
          <button type="submit" className="btn btn-primary" disabled={loading}
                  style={{ height: 42, padding: '0 20px' }}>
            {loading ? 'Tracing…' : 'Trace Vehicle'}
          </button>
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
          Searches every sighting indexed across the integrated camera network and
          reconstructs the route in time order. Partial plate reads are matched too —
          a vehicle is often read differently by different cameras.
        </div>
      </form>

      {error && <div className="test-result error">{error}</div>}

      {/* ── Idle state ───────────────────────────────────────────────────
          Before a search this page was a single input on an empty canvas,
          which tells a first-time operator nothing about what to type. The
          chips below are the plates genuinely on this deployment's watchlist
          — real rows from /plate-search/watchlist, not invented examples —
          so one click gives a trace that is certain to return something. */}
      {!searched && !loading && !error && (
        <div className="card" style={{ padding: 18 }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>
            Start with a registration number
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.65,
                        maxWidth: '72ch' }}>
            The trace returns every camera that read this vehicle, in time order,
            plotted as a route on the map with a timestamped sighting list, the
            leg-by-leg gaps between cameras, its watchlist status, and any alerts
            it has already raised.
          </div>

          {watchlist.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div className="form-label" style={{ marginBottom: 8 }}>
                On the watchlist right now
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {watchlist.slice(0, 8).map((w) => (
                  <button
                    key={w.id ?? w.plate}
                    type="button"
                    className="btn btn-secondary"
                    style={{ fontFamily: 'var(--font-mono)', letterSpacing: 0.5,
                             fontSize: 12.5, padding: '6px 12px' }}
                    title={w.reason || 'Watchlisted vehicle'}
                    onClick={() => runTrace(null, w.plate)}
                  >
                    {w.plate}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {searched && !loading && journeys.length === 0 && !error && (
        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>
            No sighting of “{query}” in the indexed network.
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            This means the registration was never read on any camera in the index —
            not that the vehicle was absent. Plate legibility depends on camera
            optics; run <code>plate_size_probe</code> against the feed to see whether
            plates on it are large enough to read at all.
          </div>
        </div>
      )}

      {journey && (
        <>
          {/* ── Summary ─────────────────────────────────────────────────── */}
          <div className="card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center',
                          flexWrap: 'wrap', marginBottom: 12 }}>
              <span style={{ fontFamily: 'monospace', fontSize: 24, fontWeight: 800 }}>
                {journey.reid_id}
              </span>
              {watchEntry ? (
                <span className="badge badge-critical">🚨 ON WATCHLIST — {watchEntry.category}</span>
              ) : (
                <button type="button" className="btn btn-secondary" onClick={addToWatchlist}
                        disabled={addingWatch}>
                  {addingWatch ? '⏳ Adding…' : '+ Add to Watchlist'}
                </button>
              )}
              {journey.route_flag && journey.route_flag !== 'NORMAL' && (
                <span className="badge badge-high">{journey.route_flag}</span>
              )}
            </div>

            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <Stat label="Cameras" value={num(journey.stops_count)} />
              <Stat label="First seen" value={fmtTime(journey.first_time)} />
              <Stat label="Last seen" value={fmtTime(journey.last_time)} />
              <Stat label="Time span" value={num(journey.time_span_minutes, ' min')} />
              <Stat label="Distance" value={num(journey.total_distance_km, ' km')} />
              <Stat label="Avg speed" value={num(journey.avg_speed_kmh, ' km/h')} />
              <Stat label="Confidence"
                    value={journey.avg_confidence != null
                      ? `${(journey.avg_confidence * 100).toFixed(1)}%` : '—'} />
            </div>

            {watchMsg && (
              <div className={`test-result ${watchMsg.startsWith('✅') ? 'success' : 'error'}`}
                   style={{ marginTop: 12 }}>{watchMsg}</div>
            )}

            {/* The backend ships these caveats with the numbers; showing the
              * figure without them would overstate what was measured. */}
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', marginTop: 10,
                          lineHeight: 1.5 }}>
              {journey.confidence_basis && <div>Confidence: {journey.confidence_basis}</div>}
              {journey.distance_basis && <div>Distance: {journey.distance_basis}</div>}
            </div>
          </div>

          {/* ── Map + timeline ──────────────────────────────────────────── */}
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.6fr) minmax(0,1fr)',
                        gap: 16, alignItems: 'stretch' }}>
            <div className="card" style={{ padding: 0, overflow: 'hidden', minHeight: 460 }}>
              {routePoints.length ? (
                <MapContainer center={routePoints[0]} zoom={12}
                              style={{ height: '100%', minHeight: 460, width: '100%' }}>
                  <FitRoute points={routePoints} />
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution="© OpenStreetMap"
                  />
                  {routePoints.length > 1 && (
                    <Polyline positions={routePoints}
                              pathOptions={{ color: '#3b82f6', weight: 4, opacity: 0.85 }} />
                  )}
                  {stops.map((s, i) => (
                    s.lat == null || s.lon == null ? null : (
                      <Marker key={`${s.camera_id}-${i}`} position={[s.lat, s.lon]}
                              icon={numberedIcon(i + 1, i === 0, i === stops.length - 1)}>
                        <Popup>
                          <strong>{i + 1}. {s.camera_location || s.camera_id}</strong><br />
                          {fmtTime(s.timestamp_utc)}<br />
                          Camera: {s.camera_id}
                          {s.department ? <> · {s.department}</> : null}<br />
                          Read as: <code>{s.plate_text || '—'}</code><br />
                          Match: {s.match_type || '—'}
                          {s.final_score != null
                            ? ` (${(s.final_score * 100).toFixed(1)}%)` : ''}
                        </Popup>
                      </Marker>
                    )
                  ))}
                </MapContainer>
              ) : (
                <div style={{ padding: 24, color: 'var(--text-muted)', fontSize: 13 }}>
                  This vehicle has sightings but no camera on the route has GPS
                  coordinates recorded, so the movement cannot be placed on the map.
                  The timeline is still complete.
                </div>
              )}
            </div>

            <div className="card" style={{ padding: 14, overflowY: 'auto', maxHeight: 520 }}>
              <div className="card-title" style={{ marginBottom: 10 }}>
                Movement history ({stops.length})
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {stops.map((s, i) => (
                  <div key={`${s.camera_id}-t-${i}`} style={{
                    display: 'flex', gap: 10, padding: '8px 10px', borderRadius: 6,
                    background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  }}>
                    <div style={{
                      width: 24, height: 24, borderRadius: '50%', flexShrink: 0,
                      background: i === 0 ? '#10b981' : i === stops.length - 1 ? '#ef4444' : '#3b82f6',
                      color: '#fff', fontSize: 12, fontWeight: 700,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>{i + 1}</div>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontWeight: 600, fontSize: 13 }}>
                        {s.camera_location || s.camera_id}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        {fmtTime(s.timestamp_utc)}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        {s.camera_id}
                        {s.lat != null && s.lon != null
                          ? ` · ${s.lat.toFixed(4)}, ${s.lon.toFixed(4)}` : ' · no GPS'}
                      </div>
                      <div style={{ fontSize: 11, marginTop: 2 }}>
                        read <code>{s.plate_text || '—'}</code>
                        {s.final_score != null && (
                          <span style={{ color: 'var(--text-muted)' }}>
                            {' '}· {(s.final_score * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── Legs ────────────────────────────────────────────────────── */}
          {(journey.legs || []).length > 0 && (
            <div className="card" style={{ padding: 14 }}>
              <div className="card-title" style={{ marginBottom: 10 }}>
                Segments between sightings
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                      <th style={{ padding: '6px 8px' }}>From → To</th>
                      <th style={{ padding: '6px 8px' }}>Straight-line</th>
                      <th style={{ padding: '6px 8px' }}>Duration</th>
                      <th style={{ padding: '6px 8px' }}>Implied speed</th>
                      <th style={{ padding: '6px 8px' }}>Heading</th>
                      <th style={{ padding: '6px 8px' }}>Flag</th>
                    </tr>
                  </thead>
                  <tbody>
                    {journey.legs.map((l, i) => (
                      <tr key={i} style={{ borderTop: '1px solid var(--border)' }}>
                        <td style={{ padding: '6px 8px' }}>
                          {l.from_camera || l.from_cam} → {l.to_camera || l.to_cam}
                        </td>
                        <td style={{ padding: '6px 8px' }}>{num(l.straight_line_km, ' km')}</td>
                        <td style={{ padding: '6px 8px' }}>{num(l.time_minutes, ' min')}</td>
                        <td style={{ padding: '6px 8px' }}>{num(l.speed_kmh, ' km/h')}</td>
                        <td style={{ padding: '6px 8px' }}>
                          {l.cardinal_heading || l.cardinal || '—'}
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          {l.flag && l.flag !== 'NORMAL'
                            ? <span className="badge badge-high" title={l.flag_reason}>{l.flag}</span>
                            : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ fontSize: 10.5, color: 'var(--text-muted)', marginTop: 8 }}>
                Straight-line geodesic between consecutive GPS fixes — the road distance
                is longer, so the implied speed is a lower bound.
              </div>
            </div>
          )}

          {/* ── Alerts raised ───────────────────────────────────────────── */}
          <div className="card" style={{ padding: 14 }}>
            <div className="card-title" style={{ marginBottom: 10 }}>
              Alerts raised for this vehicle ({alerts.length})
            </div>
            {alerts.length === 0 ? (
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                No alert has fired for this vehicle yet. Adding it to the watchlist above
                makes the live pipeline raise one automatically on the next sighting.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {alerts.map((a) => (
                  <div key={a.id} style={{
                    display: 'flex', gap: 10, alignItems: 'center', fontSize: 12,
                    padding: '6px 10px', borderRadius: 6,
                    background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  }}>
                    <span className="badge badge-critical">
                      {(a.alert_type || '').replace(/_/g, ' ')}
                    </span>
                    <span style={{ flex: 1 }}>{a.subject_label || '—'}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{fmtTime(a.created_at)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ── Other candidates ────────────────────────────────────────── */}
          {journeys.length > 1 && (
            <div className="card" style={{ padding: 14 }}>
              <div className="card-title" style={{ marginBottom: 8 }}>
                Other vehicles matching “{query}” ({journeys.length - 1})
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                Ranked below the result above. Plates are read per camera under different
                angles and lighting, so a near-miss can be the same vehicle read badly —
                or a genuinely different one. Select to inspect.
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {journeys.map((j, i) => (
                  i === selectedIdx ? null : (
                    <button key={j.reid_id + i} type="button" className="btn btn-secondary"
                            style={{ fontSize: 12 }}
                            onClick={() => setSelectedIdx(i)}>
                      <span style={{ fontFamily: 'monospace' }}>{j.reid_id}</span>
                      {' · '}{j.stops_count} sighting{j.stops_count === 1 ? '' : 's'}
                    </button>
                  )
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
