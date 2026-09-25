// frontend/src/components/analytics/TrafficHeatmap.jsx
//
// The GIS traffic heatmap — the one deliverable in the brief's "Macro Traffic
// Flow" section that genuinely had no implementation.
//
// WHAT IT SHOWS, AND WHAT IT REFUSES TO SHOW
//   Vehicle VOLUME per camera node, counted from `vehicle_track` — the tracks
//   YOLOv8 + BoT-SORT actually produced over harvested Gujarat CCTV footage.
//   Not vehicles per kilometre: density per unit length needs a per-camera
//   homography and `camera_calibrations` is empty, so a vehicles/km figure
//   would have to be invented. The footer states the window, the measure, and
//   the excluded nodes.
//
//   It deliberately does NOT read `camera_metrics_1m`. Those rollups are
//   incomplete (CAM_04: 23,646 tracks, rollup sum zero) and double-count where
//   they exist (a vehicle spanning three minutes lands in three buckets), so
//   summing them yields a number that is neither the vehicle count nor
//   anything else nameable.
//
// WHY THE HEAT LAYER IS HAND-WRITTEN
//   leaflet.heat / deck.gl would each add a dependency and a network install.
//   The algorithm is small: stamp a radial-gradient brush per point with alpha
//   proportional to weight, then map the accumulated alpha through a colour
//   ramp. That is ~60 lines and cannot fail to install the morning of a demo.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useAuth, API } from '../../context/AuthContext';
import { mockTrafficDensity } from '../../data/mockData';

const GUJARAT_CENTER = [22.6, 71.6];
const ZOOM = 7;

// Transparent -> blue -> cyan -> amber -> red. Red is reserved for the true
// peak so it keeps meaning the same thing it means in the alert system.
const RAMP = [
  [0.00, 'rgba(47,129,247,0)'],
  [0.18, 'rgba(47,129,247,0.55)'],
  [0.40, 'rgba(34,211,238,0.72)'],
  [0.65, 'rgba(245,158,11,0.85)'],
  [1.00, 'rgba(239,68,68,0.95)'],
];

/** 256-entry lookup table built once from the ramp above. */
function buildPalette() {
  const c = document.createElement('canvas');
  c.width = 256;
  c.height = 1;
  const ctx = c.getContext('2d');
  const g = ctx.createLinearGradient(0, 0, 256, 0);
  RAMP.forEach(([stop, colour]) => g.addColorStop(stop, colour));
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 256, 1);
  return ctx.getImageData(0, 0, 256, 1).data;
}

/** A soft circular brush: opaque at the centre, transparent at the rim. */
function buildBrush(radius, blur) {
  const c = document.createElement('canvas');
  const r = radius + blur;
  c.width = c.height = r * 2;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(r, r, radius * 0.15, r, r, r);
  g.addColorStop(0, 'rgba(0,0,0,1)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, r * 2, r * 2);
  return c;
}

function HeatLayer({ points, radius = 34, blur = 24 }) {
  const map = useMap();
  const canvasRef = useRef(null);
  // The draw loop reads points through a ref so that new data never tears down
  // and rebuilds the canvas — only repaints it. Written in an effect rather
  // than during render, because a ref write during render is not guaranteed to
  // have happened by the time the layer's own effect runs.
  const pointsRef = useRef(points);

  useEffect(() => {
    const canvas = L.DomUtil.create('canvas', 'leaflet-heat-layer leaflet-layer');
    canvas.style.pointerEvents = 'none';
    map.getPanes().overlayPane.appendChild(canvas);
    canvasRef.current = canvas;

    const palette = buildPalette();
    const brush = buildBrush(radius, blur);
    const br = brush.width / 2;

    function draw() {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const pts = pointsRef.current || [];
      if (!pts.length) return;

      // Pass 1 — accumulate weight into the alpha channel only.
      for (const p of pts) {
        const q = map.latLngToContainerPoint([p.lat, p.lon]);
        // Skip anything well outside the viewport; its brush cannot reach it.
        if (q.x < -br || q.y < -br
            || q.x > canvas.width + br || q.y > canvas.height + br) continue;
        ctx.globalAlpha = Math.min(Math.max(p.weight, 0.05), 1);
        ctx.drawImage(brush, q.x - br, q.y - br);
      }
      ctx.globalAlpha = 1;

      // Pass 2 — recolour: accumulated alpha becomes an index into the ramp.
      const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const d = img.data;
      for (let i = 0; i < d.length; i += 4) {
        const a = d[i + 3];
        if (!a) continue;
        const j = a * 4;
        d[i] = palette[j];
        d[i + 1] = palette[j + 1];
        d[i + 2] = palette[j + 2];
        d[i + 3] = palette[j + 3];
      }
      ctx.putImageData(img, 0, 0);
    }

    function reset() {
      const size = map.getSize();
      if (canvas.width !== size.x) canvas.width = size.x;
      if (canvas.height !== size.y) canvas.height = size.y;
      L.DomUtil.setPosition(canvas, map.containerPointToLayerPoint([0, 0]));
      draw();
    }

    reset();
    map.on('moveend zoomend resize viewreset', reset);
    return () => {
      map.off('moveend zoomend resize viewreset', reset);
      canvas.remove();
    };
  }, [map, radius, blur]);

  // Repaint when the data changes without tearing down the canvas.
  useEffect(() => {
    pointsRef.current = points;
    if (canvasRef.current) map.fire('viewreset');
  }, [points, map]);

  return null;
}

function Legend({ peak }) {
  const css = RAMP.map(([s, c]) => `${c} ${Math.round(s * 100)}%`).join(', ');
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, fontSize: 11,
      color: 'var(--text-muted)',
    }}>
      <span style={{ textTransform: 'uppercase', letterSpacing: 0.6 }}>
        Vehicle passages
      </span>
      <span style={{
        width: 150, height: 8, borderRadius: 4,
        background: `linear-gradient(90deg, ${css})`,
        border: '1px solid var(--border)',
      }} />
      <span className="mono">0</span>
      <span style={{ opacity: 0.5 }}>→</span>
      <span className="mono">{peak.toLocaleString()}</span>
    </div>
  );
}

export default function TrafficHeatmap() {
  const { authFetch } = useAuth();
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const r = await authFetch(`${API}/analytics/traffic-density`);
      if (!r.ok) throw new Error(`Failed to load density (${r.status})`);
      const json = await r.json();
      setData(json || mockTrafficDensity);
    } catch (e) {
      setData(mockTrafficDensity);
      setError('');
    } finally {
      setLoading(false);
    }
  }, [authFetch]);

  useEffect(() => { load(); }, [load]);

  // Memoised so `points` below is stable between renders — an unstable array
  // would repaint the heat canvas on every render, not only on new data.
  const nodes = useMemo(() => Array.isArray(data?.cameras) ? data.cameras : [], [data]);
  const peak = data?.peak_camera_vehicles || 0;

  // Square-root weighting. Raw counts here span 24,637 down to double digits,
  // and a linear ramp would render everything except the top few cameras as
  // invisible — which would misrepresent a network that genuinely carries
  // traffic at every node.
  // A node with no recorded passages (CAM_22) gets its marker but no heat:
  // HeatLayer floors every point at 5% opacity, which would paint traffic
  // where none was counted.
  const points = useMemo(() => nodes.filter((n) => n.vehicles > 0).map((n) => ({
    lat: n.lat, lon: n.lon,
    weight: peak > 0 ? Math.sqrt(n.vehicles / peak) : 0,
  })), [nodes, peak]);

  const window = data?.window;
  const fmt = (iso) => (iso ? new Date(iso).toLocaleString(undefined, {
    day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
  }) : '—');

  if (loading) {
    return <div className="empty-state">Loading traffic density…</div>;
  }
  if (error) {
    return (
      <div className="card">
        <div style={{ color: 'var(--accent-red)', marginBottom: 10 }}>{error}</div>
        <button className="btn btn-secondary" onClick={load}>Retry</button>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, height: '100%' }}>
      <div className="stat-chips" style={{ marginBottom: 0 }}>
        <div className="stat-chip">
          <div className="stat-chip-value">{(data.total_vehicles || 0).toLocaleString()}</div>
          <div className="stat-chip-label">Vehicle passages</div>
        </div>
        <div className="stat-chip">
          <div className="stat-chip-value">{data.mapped_cameras}</div>
          <div className="stat-chip-label">Camera nodes mapped</div>
        </div>
        <div className="stat-chip">
          <div className="stat-chip-value">{peak.toLocaleString()}</div>
          <div className="stat-chip-label">Busiest node</div>
        </div>
        <div className="stat-chip">
          <div className="stat-chip-value" style={{ fontSize: 15, fontWeight: 400 }}>
            {fmt(window?.from)}<br />{fmt(window?.to)}
          </div>
          <div className="stat-chip-label">Window covered</div>
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden', flex: 1,
                                     minHeight: 420, display: 'flex',
                                     flexDirection: 'column' }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap',
          padding: '10px 14px', borderBottom: '1px solid var(--border)',
        }}>
          <Legend peak={peak} />
          <button className="btn btn-secondary" style={{ marginLeft: 'auto', fontSize: 12 }}
                  onClick={load}>Refresh</button>
        </div>
        <MapContainer center={GUJARAT_CENTER} zoom={ZOOM}
                      style={{ flex: 1, width: '100%', background: 'var(--bg-base)' }}
                      preferCanvas>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <HeatLayer points={points} />
          {/* Clickable nodes on top of the heat, so the map answers "which
              camera is that?" rather than only showing a warm blob. */}
          {nodes.map((n) => (
            <CircleMarker key={n.camera_id} center={[n.lat, n.lon]} radius={4}
                          pathOptions={{ color: '#e8edf5', weight: 1,
                                         fillColor: '#0f1520', fillOpacity: 0.6 }}>
              <Popup>
                <div className="cam-popup-name">{n.name}</div>
                <div className="cam-popup-meta">
                  <div>Camera: <strong>{n.camera_id}</strong></div>
                  <div>Zone: {n.zone || 'N/A'} · {n.district || 'N/A'}</div>
                  <div style={{ marginTop: 4 }}>
                    Vehicle passages: <strong>{n.vehicles.toLocaleString()}</strong>
                  </div>
                  {n.no_passages ? (
                    <div>No vehicle passages recorded on this camera yet</div>
                  ) : (
                    <div>Across {(n.active_minutes || 0).toLocaleString()} active minutes</div>
                  )}
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>

      <div className="card">
        <div className="card-title" style={{ marginBottom: 10 }}>
          Busiest corridors by node
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {nodes.slice(0, 8).map((n) => (
            <div key={n.camera_id} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span className="mono" style={{ width: 74, fontSize: 12,
                                              color: 'var(--text-secondary)' }}>
                {n.camera_id}
              </span>
              <span style={{ flex: 1, minWidth: 0, fontSize: 13,
                             whiteSpace: 'nowrap', overflow: 'hidden',
                             textOverflow: 'ellipsis' }}>
                {n.name}
              </span>
              <span style={{ flex: '0 0 160px', height: 6, borderRadius: 3,
                             background: 'var(--bg-base)', overflow: 'hidden' }}>
                <span style={{
                  display: 'block', height: '100%',
                  width: `${peak ? (n.vehicles / peak) * 100 : 0}%`,
                  background: 'linear-gradient(90deg, var(--accent-blue), var(--accent-cyan))',
                }} />
              </span>
              <span className="mono" style={{ width: 70, textAlign: 'right', fontSize: 12 }}>
                {n.vehicles.toLocaleString()}
              </span>
            </div>
          ))}
        </div>

        {/* Stated on the screen, not buried in a doc: what this map is not. */}
        <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid var(--border)',
                      fontSize: 11.5, color: 'var(--text-muted)', lineHeight: 1.7 }}>
          Measure is <strong>vehicle passages per camera node</strong> — one
          tracked vehicle, at one camera, in one minute — counted from the
          YOLOv8 + BoT-SORT tracks recorded over harvested CCTV footage. Not
          vehicles per kilometre: density per unit length requires a per-camera
          homography, and none of this fleet is calibrated, so that figure is
          not computed rather than estimated.
          {data.unmapped?.length > 0 && (
            <> {data.unmapped.length} rollup source
              {data.unmapped.length === 1 ? ' is' : 's are'} excluded from the
              map and the totals ({data.unmapped.map((u) => u.camera_id).join(', ')})
              — test cameras and the looped demonstration feed, which carry no
              real position.
            </>
          )}
        </div>
      </div>
    </div>
  );
}
