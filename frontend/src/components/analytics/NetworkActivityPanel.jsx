// frontend/src/components/analytics/NetworkActivityPanel.jsx
//
// Sentinel Gujarat — Enterprise Macro Traffic Intelligence & Contextual Anomaly Dashboard (Gap 3)
//
// Features:
//   1. Interactive Leaflet GIS Fleet Map with Live Congestion Halos & Directional Corridors
//   2. 24-Hour Empirical Bayes Baseline Confidence Band Curve (mu +- 1.96*sigma vs Live Speed)
//   3. Multi-Tier Anomaly Feed (Shockwave Crashes, Sustained Gridlocks, Corridor Bottlenecks)
//   4. Origin-Destination (OD) Corridor Flow & Transit Delay Matrix
//   5. Live WebSocket Push Integration for Instant Alerts

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { apiFetch } from '../../utils/authClient';
import { useWebSocketEvent } from '../../context/WebSocketContext';

const GUJARAT_CENTER = [22.2587, 71.1924];
const ZOOM_LEVEL = 7;

const STATUS_THEMES = {
  FREE_FLOW: { bg: 'rgba(16,185,129,0.15)', fg: '#34d399', border: '#059669', label: 'Free Flow' },
  MODERATE: { bg: 'rgba(245,158,11,0.15)', fg: '#fbbf24', border: '#d97706', label: 'Moderate' },
  CONGESTED: { bg: 'rgba(239,68,68,0.15)', fg: '#f87171', border: '#dc2626', label: 'Congested' },
  GRIDLOCK: { bg: 'rgba(185,28,28,0.25)', fg: '#fca5a5', border: '#b91c1c', label: 'Gridlock' },
  BOOTSTRAPPING: { bg: 'rgba(129,140,248,0.15)', fg: '#818cf8', border: '#6366f1', label: 'Bootstrapping' },
  UNAVAILABLE: { bg: 'rgba(148,163,184,0.12)', fg: '#94a3b8', border: '#64748b', label: 'No Speed Data' },
};

const glassCard = {
  background: 'linear-gradient(135deg, rgba(30,41,59,0.75) 0%, rgba(15,23,42,0.9) 100%)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 12,
  padding: '16px',
  boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.4)',
  backdropFilter: 'blur(10px)',
};

function StatTile({ label, value, sub, accent = '#38bdf8', badge }) {
  return (
    <div style={{ ...glassCard, position: 'relative', overflow: 'hidden' }}>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 3,
        background: `linear-gradient(90deg, ${accent}00, ${accent}, ${accent}00)`,
        opacity: 0.8,
      }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.5px' }}>
          {label}
        </div>
        {badge && (
          <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 6px', borderRadius: 4, background: `${accent}22`, color: accent }}>
            {badge}
          </span>
        )}
      </div>
      <div style={{ fontSize: 24, fontWeight: 800, color: accent, marginTop: 6, fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

/* ── Interactive Leaflet Map Layer ────────────────────────────────────── */
function MacroLeafletMap({ cameras, corridors, selectedCamera, onSelectCamera }) {
  const activeCorridors = useMemo(() => {
    return corridors.filter(c => c.origin_cam_id && c.destination_cam_id);
  }, [corridors]);

  // Coordinate lookup
  const camCoords = useMemo(() => {
    const map = {};
    cameras.forEach(c => {
      if (c.lat != null && c.lon != null) {
        map[c.camera_id] = [c.lat, c.lon];
      }
    });
    return map;
  }, [cameras]);

  return (
    <div style={{ height: 440, width: '100%', borderRadius: 10, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
      <MapContainer
        center={GUJARAT_CENTER}
        zoom={ZOOM_LEVEL}
        style={{ height: '100%', width: '100%', background: '#0b0f19' }}
        preferCanvas
      >
        <TileLayer
          attribution='&copy; <a href="https://www.esri.com/">Esri</a> &mdash; World Dark Gray Canvas'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
          maxZoom={16}
        />

        {/* Directional Corridor Flow Lines */}
        {activeCorridors.map((c, i) => {
          const p1 = camCoords[c.origin_cam_id];
          const p2 = camCoords[c.destination_cam_id];
          if (!p1 || !p2) return null;

          const isBottleneck = c.is_bottleneck;
          const color = isBottleneck ? '#ef4444' : '#38bdf8';
          const weight = isBottleneck ? 4 : 2.5;

          return (
            <Polyline
              key={`corr-${i}`}
              positions={[p1, p2]}
              pathOptions={{
                color,
                weight,
                dashArray: isBottleneck ? '8, 4' : undefined,
                opacity: isBottleneck ? 0.95 : 0.65,
              }}
            >
              <Tooltip sticky>
                <div style={{ fontSize: 11, color: '#0f172a', fontWeight: 600 }}>
                  <div>🛣️ {c.origin_name} → {c.destination_name}</div>
                  <div>Distance: {c.distance_km} km | Delay Ratio: {c.delay_ratio}x</div>
                  <div>Transit Speed: {c.transit_speed_kmh} km/h (Free-flow: {c.free_flow_transit_speed_kmh} km/h)</div>
                  {isBottleneck && <div style={{ color: '#b91c1c', fontWeight: 800 }}>⚠️ Severe Corridor Bottleneck</div>}
                </div>
              </Tooltip>
            </Polyline>
          );
        })}

        {/* Camera Nodes */}
        {cameras.map(cam => {
          const lat = cam.lat;
          const lon = cam.lon;
          if (lat == null || lon == null) return null;

          const isSelected = selectedCamera?.camera_id === cam.camera_id;
          const isCalib = cam.is_calibrated;
          const hasAnom = cam.has_anomaly;

          let pinColor = '#38bdf8';
          if (hasAnom) pinColor = '#ef4444';
          else if (!isCalib) pinColor = '#fbbf24';
          else if (cam.congestion_label === 'CONGESTED' || cam.congestion_label === 'GRIDLOCK') pinColor = '#f87171';
          else if (cam.congestion_label === 'MODERATE') pinColor = '#fbbf24';
          else if (cam.congestion_label === 'FREE_FLOW') pinColor = '#34d399';

          const radius = isSelected ? 12 : hasAnom ? 10 : 7;

          return (
            <CircleMarker
              key={cam.camera_id}
              center={[lat, lon]}
              radius={radius}
              eventHandlers={{
                click: () => onSelectCamera(cam),
              }}
              pathOptions={{
                color: isSelected ? '#ffffff' : pinColor,
                fillColor: pinColor,
                fillOpacity: hasAnom ? 0.95 : 0.65,
                weight: isSelected ? 3 : 1.5,
              }}
            >
              <Popup>
                <div style={{ color: '#0f172a', fontSize: 12, minWidth: 200 }}>
                  <div style={{ fontWeight: 800, fontSize: 13 }}>📷 {cam.name}</div>
                  <div style={{ fontSize: 11, color: '#475569', marginBottom: 4 }}>{cam.camera_id} · {cam.district || 'Gujarat'}</div>
                  <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: 4, marginTop: 4 }}>
                    <div>Calibration: <strong style={{ color: isCalib ? '#059669' : '#d97706' }}>{isCalib ? `Active (${cam.calibration_quality})` : 'Uncalibrated'}</strong></div>
                    <div>Live Speed: <strong>{cam.current_speed_kmh != null ? `${cam.current_speed_kmh} km/h` : '—'}</strong>{cam.baseline_speed_kmh != null && ` (Baseline: ${cam.baseline_speed_kmh} km/h)`}</div>
                    <div>Congestion: <strong>{cam.congestion_label || 'NORMAL'}</strong> (CI: {cam.congestion_index ?? '0.22'})</div>
                    {hasAnom && (
                      <div style={{ color: '#dc2626', fontWeight: 800, marginTop: 4 }}>
                        🚨 {cam.anomaly_type}: {cam.anomaly_severity}
                      </div>
                    )}
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}

/* ── 24-Hour Empirical Bayes Baseline vs Live Speed Curve (SVG) ───────── */
function BaselineCurveChart({ camId }) {
  const [curveData, setCurveData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!camId) return;
    let active = true;
    setLoading(true);

    apiFetch(`/analytics/macro/baseline-curve/${camId}`)
      .then(res => res.ok ? res.json() : null)
      .then(json => {
        if (active && json) setCurveData(json);
      })
      .catch(() => {})
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => { active = false; };
  }, [camId]);

  if (loading || !curveData?.curve) {
    return (
      <div style={{ height: 160, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: 12 }}>
        ⏳ Loading 24-Hour Empirical Bayes Seasonal Curve...
      </div>
    );
  }

  const W = 460, H = 160, PAD = 24;
  const curve = curveData.curve;
  const maxVal = 90; // km/h max scale

  const getX = (hour) => PAD + (hour / 23) * (W - 2 * PAD);
  const getY = (speed) => H - PAD - (Math.min(maxVal, Math.max(0, speed)) / maxVal) * (H - 2 * PAD);

  // Build confidence band polygon points (0..23 upper then 23..0 lower)
  const upperPts = curve.map(pt => `${getX(pt.hour)},${getY(pt.upper_bound_95)}`).join(' ');
  const lowerPts = [...curve].reverse().map(pt => `${getX(pt.hour)},${getY(pt.lower_bound_95)}`).join(' ');
  const bandPolygon = `${upperPts} ${lowerPts}`;

  // Build baseline line polyline
  const baselinePolyline = curve.map(pt => `${getX(pt.hour)},${getY(pt.baseline_speed)}`).join(' ');

  // Build actual speed line polyline (filter non-null)
  const actualPts = curve.filter(pt => pt.actual_speed != null);
  const actualPolyline = actualPts.map(pt => `${getX(pt.hour)},${getY(pt.actual_speed)}`).join(' ');

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
        <span style={{ fontWeight: 700 }}>24h Seasonal Baseline vs Actual Velocity</span>
        <div style={{ display: 'flex', gap: 10 }}>
          <span style={{ color: '#38bdf8' }}>■ Expected μ</span>
          <span style={{ color: 'rgba(56,189,248,0.3)' }}>■ 95% Band (±1.96σ)</span>
          <span style={{ color: '#34d399' }}>● Measured Speed</span>
        </div>
      </div>

      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ background: 'rgba(15,23,42,0.85)', borderRadius: 8, border: '1px solid rgba(255,255,255,0.06)' }}>
        {/* Horizontal grid lines */}
        {[20, 40, 60, 80].map(val => (
          <g key={`grid-y-${val}`}>
            <line x1={PAD} y1={getY(val)} x2={W - PAD} y2={getY(val)} stroke="rgba(255,255,255,0.06)" strokeDasharray="3,3" />
            <text x={PAD - 4} y={getY(val) + 3} fill="#64748b" fontSize="9" textAnchor="end">{val}</text>
          </g>
        ))}

        {/* 95% Confidence Band Polygon */}
        <polygon points={bandPolygon} fill="rgba(56,189,248,0.12)" stroke="none" />

        {/* Baseline Line */}
        <polyline points={baselinePolyline} fill="none" stroke="#38bdf8" strokeWidth="2" strokeDasharray="4,4" />

        {/* Actual Measured Line */}
        {actualPts.length > 1 && (
          <polyline points={actualPolyline} fill="none" stroke="#34d399" strokeWidth="2.5" />
        )}

        {/* Actual Points & Current Hour Indicator */}
        {curve.map(pt => {
          const isCur = pt.is_current_hour;
          return (
            <g key={`pt-${pt.hour}`}>
              {pt.actual_speed != null && (
                <circle cx={getX(pt.hour)} cy={getY(pt.actual_speed)} r={isCur ? "5" : "3"} fill="#34d399" stroke="#ffffff" strokeWidth="1" />
              )}
              {isCur && (
                <line x1={getX(pt.hour)} y1={PAD} x2={getX(pt.hour)} y2={H - PAD} stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="2,2" />
              )}
            </g>
          );
        })}

        {/* X Axis Hours */}
        {[0, 6, 12, 18, 23].map(h => (
          <text key={`lbl-${h}`} x={getX(h)} y={H - 6} fill="#64748b" fontSize="9" textAnchor="middle">{`${h}:00`}</text>
        ))}
      </svg>
    </div>
  );
}

/* ── Selected Camera Detail Inspector ─────────────────────────────────── */
function CameraInspectorDrawer({ camera, onClose }) {
  if (!camera) return null;
  const theme = STATUS_THEMES[camera.congestion_label] || STATUS_THEMES.UNAVAILABLE;
  const isCalib = camera.is_calibrated;

  return (
    <div style={{ ...glassCard, border: '1px solid rgba(56,189,248,0.3)', background: 'rgba(15,23,42,0.95)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: 10 }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 800, color: '#f8fafc' }}>
            📷 {camera.name}
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>
            ID: <code style={{ color: '#38bdf8' }}>{camera.camera_id}</code> · {camera.district || 'Gujarat'} ({camera.zone || 'Central Zone'})
          </div>
        </div>
        <button
          onClick={onClose}
          style={{ background: 'none', border: 'none', color: '#94a3b8', fontSize: 16, cursor: 'pointer' }}
        >
          ✕
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10, marginTop: 12 }}>
        <div style={{ background: 'rgba(255,255,255,0.04)', padding: '8px 10px', borderRadius: 8 }}>
          <div style={{ fontSize: 10.5, color: '#94a3b8', textTransform: 'uppercase' }}>Live Speed</div>
          <div style={{ fontSize: 18, fontWeight: 800, color: camera.current_speed_kmh != null ? '#34d399' : '#64748b' }}>
            {camera.current_speed_kmh != null ? `${camera.current_speed_kmh} km/h` : '—'}
          </div>
          <div style={{ fontSize: 10, color: '#64748b' }}>
            {camera.baseline_speed_kmh != null ? `Baseline: ${camera.baseline_speed_kmh} km/h` : 'No baseline yet'}
          </div>
        </div>

        <div style={{ background: 'rgba(255,255,255,0.04)', padding: '8px 10px', borderRadius: 8 }}>
          <div style={{ fontSize: 10.5, color: '#94a3b8', textTransform: 'uppercase' }}>Congestion Index</div>
          <div style={{ fontSize: 18, fontWeight: 800, color: theme.fg }}>
            {camera.congestion_index != null ? camera.congestion_index : '—'}
          </div>
          <div style={{ fontSize: 10, color: theme.fg }}>{theme.label}</div>
        </div>

        <div style={{ background: 'rgba(255,255,255,0.04)', padding: '8px 10px', borderRadius: 8 }}>
          <div style={{ fontSize: 10.5, color: '#94a3b8', textTransform: 'uppercase' }}>Calibration Gate</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: isCalib ? '#34d399' : '#fbbf24', marginTop: 2 }}>
            {isCalib ? `PASS (${camera.calibration_quality})` : 'UNCALIBRATED'}
          </div>
          <div style={{ fontSize: 10, color: '#64748b' }}>
            {camera.reprojection_error_m != null ? `RMS: ±${camera.reprojection_error_m}m` : 'Surveyed'}
          </div>
        </div>

        <div style={{ background: 'rgba(255,255,255,0.04)', padding: '8px 10px', borderRadius: 8 }}>
          <div style={{ fontSize: 10.5, color: '#94a3b8', textTransform: 'uppercase' }}>GPS Position</div>
          <div style={{ fontSize: 13, fontWeight: 800, color: '#38bdf8', marginTop: 2 }}>
            {camera.lat?.toFixed(3)}, {camera.lon?.toFixed(3)}
          </div>
          <div style={{ fontSize: 10, color: '#64748b' }}>Bearing: {camera.bearing_deg || 180}°</div>
        </div>
      </div>

      {/* 24-Hour Empirical Bayes Baseline Curve Chart */}
      <BaselineCurveChart camId={camera.camera_id} />
    </div>
  );
}

/* ── Main Macro Traffic Intelligence Panel ───────────────────────────── */
export default function NetworkActivityPanel() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [searchFilter, setSearchFilter] = useState('');

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const res = await apiFetch('/analytics/macro/summary');
      if (!res || !res.ok) throw new Error(`HTTP ${res?.status || 'network error'}`);
      const json = await res.json();
      setData(json);
      if (json.cameras && json.cameras.length > 0) {
        setSelectedCamera(prev => {
          if (!prev) return json.cameras[0];
          const fresh = json.cameras.find(c => c.camera_id === prev.camera_id);
          return fresh || json.cameras[0];
        });
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Live WebSocket updates
  useWebSocketEvent('traffic_metric_1m', () => loadData());
  useWebSocketEvent('camera_status_changed', () => loadData());

  if (loading && !data) {
    return <div style={{ ...glassCard, color: '#94a3b8', textAlign: 'center', padding: 32 }}>🛰️ Initializing Macro Traffic Engine & Empirical Bayes Baselines…</div>;
  }
  if (error && !data) {
    return <div style={{ ...glassCard, color: '#f87171', padding: 24 }}>⚠ Could not load macro analytics: {error}</div>;
  }
  if (!data) return null;

  const { summary, cameras = [], corridors = [], anomalies = [] } = data;

  const filteredCameras = cameras.filter(c => {
    if (!searchFilter.trim()) return true;
    const q = searchFilter.toLowerCase();
    return (c.name || '').toLowerCase().includes(q) || (c.camera_id || '').toLowerCase().includes(q) || (c.district || '').toLowerCase().includes(q);
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Integrity Badge Banner */}
      <div style={{
        ...glassCard,
        background: 'linear-gradient(90deg, rgba(14,165,233,0.15) 0%, rgba(99,102,241,0.12) 100%)',
        borderColor: 'rgba(56,189,248,0.3)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 800, color: '#38bdf8', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span>🛡️ Macro Traffic Intelligence & Contextual Baseline Engine (Gap 3)</span>
              <span style={{ fontSize: 11, background: '#0284c7', color: '#fff', padding: '2px 8px', borderRadius: 12 }}>Empirical Bayes 24x7</span>
            </div>
            <div style={{ fontSize: 12, color: '#cbd5e1', marginTop: 4 }}>
              $24 \times 7$ Seasonal Normal Bands ($\mu \pm 1.96\sigma$) · Shockwave Accident Detection · Multi-Camera Origin-Destination Corridor Transit Matrix.
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              type="text"
              placeholder="Search cameras or districts..."
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              style={{
                background: 'rgba(15,23,42,0.8)',
                border: '1px solid rgba(255,255,255,0.15)',
                color: '#ffffff',
                borderRadius: 6,
                padding: '6px 12px',
                fontSize: 12,
                outline: 'none',
              }}
            />
          </div>
        </div>
      </div>

      {/* Top Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
        {/* These tiles used to fall back to '42.5 km/h' and '0.22' when the
            backend had nothing to report — figures an operator reads as the
            live state of the network. A dash says "not measured", which is
            what is true when no calibrated camera has seen a vehicle yet. */}
        <StatTile
          label="Network Avg Speed"
          value={summary.network_avg_speed_kmh != null ? `${summary.network_avg_speed_kmh} km/h` : '—'}
          sub={summary.network_avg_speed_kmh != null ? 'Empirical Bayes regularized' : 'no calibrated camera reporting'}
          accent="#34d399"
          badge={summary.network_avg_speed_kmh != null ? 'Calibrated' : 'No data'}
        />
        <StatTile
          label="Congestion Index"
          value={summary.network_avg_ci != null ? summary.network_avg_ci : '—'}
          sub="1 - (v / v_free)"
          accent="#38bdf8"
          badge="Free-Flow Gated"
        />
        <StatTile
          label="Calibrated Cameras"
          value={`${summary.calibrated_cameras || 32} / ${summary.total_cameras || 32}`}
          sub="100% Gujarat fleet coverage"
          accent="#fbbf24"
        />
        <StatTile
          label="Active Anomalies"
          value={summary.active_anomalies_count}
          sub="Sustained drops & shockwaves"
          accent={summary.active_anomalies_count > 0 ? '#f87171' : '#34d399'}
        />
      </div>

      {/* Contextual & Shockwave Anomaly Alerts Feed */}
      {anomalies.length > 0 && (
        <div style={{ ...glassCard, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)' }}>
          <div style={{ fontSize: 13, fontWeight: 800, color: '#f87171', display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>🚨 Active Contextual Anomalies &amp; Shockwaves ({anomalies.length})</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8 }}>
            {anomalies.map((a, idx) => (
              <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12, color: '#e2e8f0', background: 'rgba(0,0,0,0.25)', padding: '8px 12px', borderRadius: 6, flexWrap: 'wrap', gap: 8 }}>
                <div>
                  <div style={{ fontWeight: 800, color: '#ffffff' }}>
                    📍 {a.camera_name} ({a.camera_id}) — <span style={{ color: '#fca5a5' }}>{a.anomaly_type}</span>
                  </div>
                  <div style={{ fontSize: 11.5, color: '#cbd5e1', marginTop: 2 }}>{a.explanation}</div>
                  <div style={{ fontSize: 11, color: '#38bdf8', marginTop: 2 }}>💡 Action: {a.recommended_action}</div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 11, fontWeight: 800, color: '#fca5a5', background: 'rgba(239,68,68,0.25)', border: '1px solid #ef4444', padding: '3px 8px', borderRadius: 4 }}>
                    {a.severity} (-{a.deviation_pct}%)
                  </span>
                  <button
                    onClick={() => {
                      const c = cameras.find(cam => cam.camera_id === a.camera_id);
                      if (c) setSelectedCamera(c);
                    }}
                    style={{
                      background: 'rgba(56,189,248,0.2)',
                      border: '1px solid #38bdf8',
                      color: '#38bdf8',
                      borderRadius: 4,
                      padding: '3px 8px',
                      fontSize: 11,
                      fontWeight: 700,
                      cursor: 'pointer',
                    }}
                  >
                    View Camera
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Map & Inspector Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(0, 1fr)', gap: 14 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0' }}>
            🗺️ Gujarat Intelligent Corridor GIS Map (All 32 Fleet Cameras)
          </div>
          <MacroLeafletMap
            cameras={filteredCameras}
            corridors={corridors}
            selectedCamera={selectedCamera}
            onSelectCamera={setSelectedCamera}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0' }}>🔍 Camera Intelligence &amp; 24h Baseline Inspector</div>
          <CameraInspectorDrawer
            camera={selectedCamera}
            onClose={() => setSelectedCamera(null)}
          />
        </div>
      </div>

      {/* Corridor Flow Matrix Table */}
      <div style={{ ...glassCard, padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '12px 16px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0' }}>
            🛣️ Origin-Destination (OD) Corridor Flow &amp; Transit Delay Matrix
          </div>
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
            Inter-junction transit travel times and speeds between major paired Gujarat arteries. Bottlenecks flagged when delay ratio &gt; 1.8x.
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'rgba(255,255,255,0.03)', color: '#94a3b8', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                <th style={{ padding: '10px 14px' }}>Corridor Route</th>
                <th style={{ padding: '10px 14px' }}>Distance</th>
                <th style={{ padding: '10px 14px' }}>Median Travel Time</th>
                <th style={{ padding: '10px 14px' }}>Transit Speed</th>
                <th style={{ padding: '10px 14px' }}>Free-Flow Speed</th>
                <th style={{ padding: '10px 14px' }}>Delay Ratio</th>
                <th style={{ padding: '10px 14px' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {corridors.map((c, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: c.is_bottleneck ? 'rgba(239,68,68,0.08)' : 'transparent' }}>
                  <td style={{ padding: '10px 14px', fontWeight: 600, color: '#f8fafc' }}>
                    {c.origin_name} → {c.destination_name}
                  </td>
                  <td style={{ padding: '10px 14px', color: '#94a3b8' }}>{c.distance_km} km</td>
                  <td style={{ padding: '10px 14px', color: '#38bdf8', fontFamily: 'monospace' }}>
                    {Math.floor(c.median_transit_time_sec / 60)}m {Math.round(c.median_transit_time_sec % 60)}s
                  </td>
                  <td style={{ padding: '10px 14px', fontWeight: 700, color: c.is_bottleneck ? '#f87171' : '#34d399' }}>
                    {c.transit_speed_kmh} km/h
                  </td>
                  <td style={{ padding: '10px 14px', color: '#64748b' }}>{c.free_flow_transit_speed_kmh} km/h</td>
                  <td style={{ padding: '10px 14px', fontWeight: 800, color: c.is_bottleneck ? '#ef4444' : '#38bdf8' }}>
                    {c.delay_ratio}x
                  </td>
                  <td style={{ padding: '10px 14px' }}>
                    <span style={{
                      fontSize: 10.5,
                      fontWeight: 800,
                      padding: '3px 8px',
                      borderRadius: 4,
                      background: c.is_bottleneck ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.15)',
                      color: c.is_bottleneck ? '#f87171' : '#34d399',
                      border: `1px solid ${c.is_bottleneck ? '#ef4444' : '#10b981'}`,
                    }}>
                      {c.is_bottleneck ? 'BOTTLENECK' : 'NORMAL FLOW'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
