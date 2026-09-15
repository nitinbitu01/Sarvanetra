// frontend/src/components/LiveMap.jsx
//
// Cameras, officers, and active CRITICAL alerts on one map.
// Defense-grade C2 operational map with:
// - Crystal-clear basemaps (Street High-Vis OSM, Tactical Dark Esri, Satellite)
// - Zero watermark / no API key requirements
// - Auto-fit fleet bounds & sector quick-jump (Ahmedabad, Rajkot, Surat, etc.)
// - Rich hover tooltips & click inspection popups
// - Real-time websocket telemetry and layered GIS filters
import { Fragment, useState, useEffect, useCallback, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Circle, Popup, Tooltip, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useAuth, API } from '../context/AuthContext';
import { useWebSocketEvent } from '../context/WebSocketContext';

const GUJARAT_CENTER = [22.4, 71.5];
const GUJARAT_DEFAULT_ZOOM = 8;
const ASSUMED_COVERAGE_RADIUS_M = 120;
const ALL = '__all__';

// High-clarity basemap providers with 0 watermarks and 100% readability
const BASEMAPS = {
  street: {
    id: 'street',
    label: 'Street',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    subdomains: 'abc',
    maxZoom: 19,
  },
  tactical: {
    id: 'tactical',
    label: 'Tactical Dark',
    base: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    reference: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
    maxZoom: 16,
  },
  satellite: {
    id: 'satellite',
    label: 'Satellite',
    base: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    reference: 'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar',
    maxZoom: 18,
  },
};

const SECTORS = [
  { id: 'fleet', label: 'All Gujarat' },
  { id: 'ahmedabad', label: 'Ahmedabad', center: [23.0225, 72.5714], zoom: 13 },
  { id: 'gandhinagar', label: 'Gandhinagar', center: [23.2156, 72.6369], zoom: 13 },
  { id: 'rajkot', label: 'Rajkot', center: [22.3039, 70.8022], zoom: 13 },
  { id: 'surat', label: 'Surat', center: [21.1702, 72.8311], zoom: 13 },
  { id: 'junagadh', label: 'Junagadh', center: [21.5222, 70.4579], zoom: 13 },
];

function dot(color, size = 11, ring = '#0A0E1A', glow = '') {
  return L.divIcon({
    className: 'c2-map-pin',
    html: `<div style="width:${size}px;height:${size}px;border-radius:50%;
           background:${color};border:2px solid ${ring};
           box-shadow:${glow || `0 0 6px ${color}`};transition:transform 0.15s ease;"></div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

const ICON = {
  cameraOnline: dot('#10B981', 11, '#047857', '0 0 8px rgba(16,185,129,0.7)'),
  cameraOffline: dot('#EF4444', 11, '#991B1B', '0 0 6px rgba(239,68,68,0.6)'),
  cameraUnreachable: dot('#F59E0B', 11, '#B45309', '0 0 6px rgba(245,158,11,0.6)'),
  cameraMaintenance: dot('#8B5CF6', 11, '#5B21B6', '0 0 6px rgba(139,92,246,0.6)'),
  officer: dot('#3B82F6', 12, '#FFFFFF', '0 0 8px rgba(59,130,246,0.8)'),
  alert: dot('#DC2626', 15, '#FFFFFF', '0 0 12px #DC2626'),
};

function distinctSorted(cameras, field) {
  return Array.from(new Set(
    cameras.map((c) => c[field]).filter((v) => v != null && v !== ''),
  )).sort();
}

/** Controller component inside MapContainer for automatic size invalidation and fitBounds */
function MapController({ placedCameras, triggerSector, onSectorComplete }) {
  const map = useMap();
  const hasAutoFitted = useRef(false);

  useEffect(() => {
    const t = setTimeout(() => map.invalidateSize(), 150);
    return () => clearTimeout(t);
  }, [map]);

  // Initial auto-fit when cameras first arrive
  useEffect(() => {
    if (hasAutoFitted.current || !placedCameras || placedCameras.length === 0) return;
    const valid = placedCameras
      .filter((c) => c.gps_lat != null && c.gps_lon != null)
      .map((c) => [c.gps_lat, c.gps_lon]);
    if (valid.length > 0) {
      const bounds = L.latLngBounds(valid);
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [40, 40], maxZoom: 14 });
        hasAutoFitted.current = true;
      }
    }
  }, [map, placedCameras]);

  // Handle Sector jumps
  useEffect(() => {
    if (!triggerSector) return;
    if (triggerSector === 'fleet') {
      const valid = placedCameras
        .filter((c) => c.gps_lat != null && c.gps_lon != null)
        .map((c) => [c.gps_lat, c.gps_lon]);
      if (valid.length > 0) {
        map.fitBounds(L.latLngBounds(valid), { padding: [40, 40], maxZoom: 14 });
      } else {
        map.setView(GUJARAT_CENTER, GUJARAT_DEFAULT_ZOOM);
      }
    } else {
      const sec = SECTORS.find((s) => s.id === triggerSector);
      if (sec && sec.center) {
        map.flyTo(sec.center, sec.zoom, { duration: 0.8 });
      }
    }
    onSectorComplete?.();
  }, [map, triggerSector, placedCameras, onSectorComplete]);

  return null;
}

export default function LiveMap({ cameras: externalCameras } = {}) {
  const { authFetch } = useAuth();
  const [cameras, setCameras] = useState(() => (Array.isArray(externalCameras) && externalCameras.length > 0 ? externalCameras : []));
  const [officers, setOfficers] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [failed, setFailed] = useState(false);
  const [activeBasemap, setActiveBasemap] = useState(() => {
    return localStorage.getItem('c2_basemap_mode') || 'street';
  });
  const [targetSector, setTargetSector] = useState(null);
  const camerasRef = useRef([]);
  camerasRef.current = cameras;

  useEffect(() => {
    if (Array.isArray(externalCameras) && externalCameras.length > 0) {
      setCameras(externalCameras);
    }
  }, [externalCameras]);

  const [deptFilter, setDeptFilter] = useState(ALL);
  const [zoneFilter, setZoneFilter] = useState(ALL);
  const [statusFilter, setStatusFilter] = useState(ALL);
  const [riskFilter, setRiskFilter] = useState(ALL);
  const [typeFilter, setTypeFilter] = useState(ALL);
  const [showOfficers, setShowOfficers] = useState(true);
  const [showAlerts, setShowAlerts] = useState(true);
  const [showCoverage, setShowCoverage] = useState(false);

  const loadMapData = useCallback(async () => {
    try {
      if (!externalCameras || externalCameras.length === 0) {
        const cRes = await authFetch(`${API}/cameras`).catch(() => null);
        if (cRes && cRes.ok) {
          const cData = await cRes.json();
          setCameras(Array.isArray(cData) ? cData : (cData.cameras || []));
        }
      }
      const oRes = await authFetch(`${API}/officers`).catch(() => null);
      if (oRes && oRes.ok) {
        const oData = await oRes.json();
        setOfficers(Array.isArray(oData) ? oData : (oData.officers || []));
      }
    } catch {
      // transient network error, retry automatically
    }
  }, [authFetch, externalCameras]);

  useEffect(() => {
    loadMapData();
    const interval = setInterval(loadMapData, 12000);
    return () => clearInterval(interval);
  }, [loadMapData]);

  useWebSocketEvent('camera_status_changed', useCallback((e) => {
    setCameras((prev) => prev.map((c) =>
      c.id === e.camera_id ? { ...c, status: e.status } : c));
  }, []));

  useWebSocketEvent('officer.status', useCallback((e) => {
    setOfficers((prev) => prev.map((o) =>
      o.id === e.officer_id ? { ...o, status: e.status } : o));
  }, []));

  useWebSocketEvent('alert.routed', useCallback((e) => {
    const cam = camerasRef.current.find((c) => c.id === e.camera_id);
    if (!cam || cam.gps_lat == null || cam.gps_lon == null) return;
    setAlerts((prev) => [
      ...prev.filter((a) => a.alert_id !== e.alert_id),
      { alert_id: e.alert_id, lat: cam.gps_lat, lng: cam.gps_lon, camera_name: cam.name },
    ].slice(-20));
  }, []));

  useWebSocketEvent('alert.ack', useCallback((e) => {
    setAlerts((prev) => prev.filter((a) => a.alert_id !== e.alert_id));
  }, []));

  if (failed) throw new Error('Map data failed to load');

  const withGps = cameras.filter((c) => c.gps_lat != null && c.gps_lon != null);
  const departments = distinctSorted(withGps, 'department');
  const zones = distinctSorted(withGps, 'zone');
  const risks = distinctSorted(withGps, 'risk_level');
  const types = distinctSorted(withGps, 'camera_type');

  const placed = withGps.filter((c) =>
    (deptFilter === ALL || c.department === deptFilter)
    && (zoneFilter === ALL || c.zone === zoneFilter)
    && (statusFilter === ALL || c.status === statusFilter)
    && (riskFilter === ALL || c.risk_level === riskFilter)
    && (typeFilter === ALL || (c.camera_type || 'Fixed') === typeFilter));

  const onlineCount = placed.filter((c) => c.status === 'ONLINE').length;
  const offlineCount = placed.filter((c) => c.status !== 'ONLINE').length;

  const handleBasemapChange = (mode) => {
    setActiveBasemap(mode);
    try { localStorage.setItem('c2_basemap_mode', mode); } catch {}
  };

  const selectStyle = {
    background: 'var(--bg-elevated, #111827)',
    color: 'var(--text-primary, #F9FAFB)',
    border: '1px solid var(--border-default, #374151)',
    borderRadius: 'var(--radius-sm, 2px)',
    padding: '3px 8px',
    fontSize: 11,
    fontFamily: 'var(--font-sans, sans-serif)',
    height: 28,
  };

  const filterActive = deptFilter !== ALL || zoneFilter !== ALL
    || statusFilter !== ALL || riskFilter !== ALL || typeFilter !== ALL;

  const currentBasemap = BASEMAPS[activeBasemap] || BASEMAPS.street;

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', position: 'relative' }}>
      {/* ── Top GIS Filter Toolbar ─────────────────────────────────────── */}
      <div style={{
        display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap',
        padding: '6px 10px', background: 'var(--bg-surface, #0F172A)', borderBottom: '1px solid var(--border, #1F2937)',
        zIndex: 10,
      }}>
        <select value={deptFilter} onChange={(e) => setDeptFilter(e.target.value)}
                style={selectStyle} aria-label="Filter by department">
          <option value={ALL}>All departments ({withGps.length})</option>
          {departments.map((d) => (
            <option key={d} value={d}>{d}
              {` (${withGps.filter((c) => c.department === d).length})`}</option>
          ))}
        </select>
        <select value={zoneFilter} onChange={(e) => setZoneFilter(e.target.value)}
                style={selectStyle} aria-label="Filter by zone">
          <option value={ALL}>All zones</option>
          {zones.map((z) => <option key={z} value={z}>{z}</option>)}
        </select>
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}
                style={selectStyle} aria-label="Filter by status">
          <option value={ALL}>All statuses</option>
          <option value="ONLINE">Online</option>
          <option value="OFFLINE">Offline</option>
          <option value="UNREACHABLE">Unreachable</option>
        </select>
        <select value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)}
                style={selectStyle} aria-label="Filter by risk level">
          <option value={ALL}>All risk levels</option>
          {risks.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
        <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}
                style={selectStyle} aria-label="Filter by camera type">
          <option value={ALL}>All types</option>
          {types.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>

        {/* Layer toggles */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginLeft: 4 }}>
          <label style={{ fontSize: 11, color: 'var(--text-secondary, #9CA3AF)', display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
            <input type="checkbox" checked={showOfficers} onChange={(e) => setShowOfficers(e.target.checked)} />
            Officers
          </label>
          <label style={{ fontSize: 11, color: 'var(--text-secondary, #9CA3AF)', display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
            <input type="checkbox" checked={showAlerts} onChange={(e) => setShowAlerts(e.target.checked)} />
            Alerts
          </label>
          <label style={{ fontSize: 11, color: 'var(--text-secondary, #9CA3AF)', display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}
                 title={`Illustrative ${ASSUMED_COVERAGE_RADIUS_M}m radius`}>
            <input type="checkbox" checked={showCoverage} onChange={(e) => setShowCoverage(e.target.checked)} />
            Coverage
          </label>
        </div>

        {filterActive && (
          <button
            onClick={() => {
              setDeptFilter(ALL); setZoneFilter(ALL);
              setStatusFilter(ALL); setRiskFilter(ALL); setTypeFilter(ALL);
            }}
            style={{ ...selectStyle, cursor: 'pointer', color: 'var(--accent-primary, #2563EB)', fontWeight: 600 }}
          >
            Reset Filters
          </button>
        )}

        {/* Basemap Switcher */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 2, background: 'var(--bg-base, #0A0E1A)', padding: 2, borderRadius: 'var(--radius-sm, 2px)', border: '1px solid var(--border-default, #374151)' }}>
          {Object.entries(BASEMAPS).map(([key, bm]) => (
            <button
              key={key}
              onClick={() => handleBasemapChange(key)}
              style={{
                padding: '2px 8px',
                fontSize: 10.5,
                fontWeight: activeBasemap === key ? 600 : 400,
                color: activeBasemap === key ? '#FFFFFF' : 'var(--text-muted, #6B7280)',
                background: activeBasemap === key ? 'var(--accent-primary, #2563EB)' : 'transparent',
                border: 'none',
                borderRadius: 'var(--radius-sm, 2px)',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
              title={`Switch to ${bm.label} basemap`}
            >
              {bm.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Sub-bar: Sector Quick Jump & Fleet Status ───────────────────── */}
      <div style={{
        display: 'flex', gap: 4, alignItems: 'center', padding: '4px 10px',
        background: '#0B1120', borderBottom: '1px solid var(--border, #1F2937)',
        fontSize: 10.5, color: 'var(--text-secondary, #9CA3AF)', zIndex: 9,
      }}>
        <span style={{ fontWeight: 600, color: 'var(--text-muted, #6B7280)', marginRight: 4, textTransform: 'uppercase', letterSpacing: '0.6px' }}>
          Sector Jump:
        </span>
        {SECTORS.map((s) => (
          <button
            key={s.id}
            onClick={() => setTargetSector(s.id)}
            style={{
              background: 'var(--bg-elevated, #1F2937)',
              color: 'var(--text-primary, #F9FAFB)',
              border: '1px solid var(--border, #374151)',
              borderRadius: 'var(--radius-sm, 2px)',
              padding: '2px 7px',
              fontSize: 10,
              cursor: 'pointer',
              fontWeight: 500,
            }}
          >
            {s.label}
          </button>
        ))}

        <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono, monospace)', color: 'var(--text-secondary, #9CA3AF)' }}>
          <span style={{ color: '#10B981', fontWeight: 600 }}>{onlineCount}</span> online ·{' '}
          <span style={{ color: '#EF4444', fontWeight: 600 }}>{offlineCount}</span> offline ·{' '}
          {placed.length} of {withGps.length} cameras
        </span>
      </div>

      {/* ── Leaflet Map Container ──────────────────────────────────────── */}
      <MapContainer
        center={GUJARAT_CENTER}
        zoom={GUJARAT_DEFAULT_ZOOM}
        style={{ flex: 1, width: '100%', background: activeBasemap === 'street' ? '#E5E7EB' : '#0A0E1A' }}
        preferCanvas
      >
        <MapController
          placedCameras={placed}
          triggerSector={targetSector}
          onSectorComplete={() => setTargetSector(null)}
        />

        {/* Render Active Basemap without any watermarks */}
        {currentBasemap.url ? (
          <TileLayer
            key={currentBasemap.id}
            url={currentBasemap.url}
            attribution={currentBasemap.attribution}
            subdomains={currentBasemap.subdomains || 'abc'}
            maxZoom={currentBasemap.maxZoom || 19}
          />
        ) : (
          <>
            <TileLayer
              key={`${currentBasemap.id}-base`}
              url={currentBasemap.base}
              attribution={currentBasemap.attribution}
              maxZoom={currentBasemap.maxZoom || 16}
            />
            {currentBasemap.reference && (
              <TileLayer
                key={`${currentBasemap.id}-ref`}
                url={currentBasemap.reference}
                maxZoom={currentBasemap.maxZoom || 16}
                opacity={0.9}
              />
            )}
          </>
        )}

        {/* ── Camera Markers ────────────────────────────────────────────── */}
        {placed.map((c) => (
          <Fragment key={`cam-frag-${c.id}`}>
            {showCoverage && (
              <Circle
                center={[c.gps_lat, c.gps_lon]}
                radius={ASSUMED_COVERAGE_RADIUS_M}
                pathOptions={{
                  color: c.status === 'ONLINE' ? '#10B981' : '#EF4444',
                  fillColor: c.status === 'ONLINE' ? '#10B981' : '#EF4444',
                  fillOpacity: 0.12,
                  weight: 1,
                  opacity: 0.4,
                }}
              />
            )}
            <Marker
              position={[c.gps_lat, c.gps_lon]}
              icon={
                c.status === 'MAINTENANCE' ? ICON.cameraMaintenance
                  : c.status === 'ONLINE' ? ICON.cameraOnline
                  : c.status === 'UNREACHABLE' ? ICON.cameraUnreachable
                  : ICON.cameraOffline
              }
            >
              {/* Hover Tooltip for instant, effortless reading without clicking */}
              <Tooltip direction="top" offset={[0, -8]} opacity={0.95} className="c2-map-tooltip">
                <div style={{ fontWeight: 600, fontSize: 11.5 }}>{c.name}</div>
                <div style={{ fontSize: 10, color: '#9CA3AF', marginTop: 1 }}>
                  {c.zone || 'Central'} Zone · {c.department || 'Traffic'} ·{' '}
                  <span style={{ color: c.status === 'ONLINE' ? '#10B981' : '#EF4444', fontWeight: 600 }}>
                    {c.status}
                  </span>
                </div>
              </Tooltip>

              {/* Click Popup with rich metadata & real snapshot preview */}
              <Popup maxWidth={280}>
                <div style={{ fontFamily: 'var(--font-sans, sans-serif)', fontSize: 12, padding: 2 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div style={{ fontWeight: 600, color: 'var(--text-primary, #F9FAFB)', fontSize: 13 }}>
                      {c.name}
                    </div>
                    <span style={{
                      fontFamily: 'var(--font-mono, monospace)',
                      fontSize: 10,
                      fontWeight: 600,
                      padding: '2px 6px',
                      borderRadius: 2,
                      background: c.status === 'ONLINE' ? '#064E3B' : '#7F1D1D',
                      color: c.status === 'ONLINE' ? '#34D399' : '#FCA5A5',
                    }}>
                      {c.status}
                    </span>
                  </div>

                  <div style={{ color: 'var(--text-secondary, #9CA3AF)', fontSize: 11, marginTop: 4 }}>
                    {c.zone || 'Central'} Zone · {c.camera_type || 'Fixed Dome'}
                  </div>

                  {/* Monospace GPS Coordinates */}
                  <div style={{
                    marginTop: 6,
                    padding: '4px 6px',
                    background: 'var(--bg-elevated, #1F2937)',
                    border: '1px solid var(--border-default, #374151)',
                    borderRadius: 2,
                    fontFamily: 'var(--font-mono, monospace)',
                    fontSize: 10.5,
                    color: 'var(--text-secondary, #9CA3AF)',
                  }}>
                    GPS: {Number(c.gps_lat).toFixed(4)}° N, {Number(c.gps_lon).toFixed(4)}° E
                  </div>

                  {c.maintenance_mode && c.maintenance_note && (
                    <div style={{ fontSize: 11, color: '#A78BFA', marginTop: 6 }}>
                      Maintenance: {c.maintenance_note}
                    </div>
                  )}

                  <div style={{ marginTop: 8, display: 'flex', gap: 6 }}>
                    <button
                      onClick={() => alert(`Camera ${c.name} (${c.id}) focused.`)}
                      style={{
                        flex: 1,
                        padding: '4px 8px',
                        background: 'var(--accent-primary, #2563EB)',
                        color: '#FFFFFF',
                        border: 'none',
                        borderRadius: 2,
                        fontSize: 10.5,
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      Focus Stream
                    </button>
                  </div>
                </div>
              </Popup>
            </Marker>
          </Fragment>
        ))}

        {/* ── Officer Units ────────────────────────────────────────────── */}
        {showOfficers && officers
          .filter((o) => o.status !== 'OFFLINE' && o.lat != null && o.lng != null)
          .map((o) => (
            <Marker key={`off-${o.id}`} position={[o.lat, o.lng]} icon={ICON.officer}>
              <Tooltip direction="top" offset={[0, -8]} opacity={0.95} className="c2-map-tooltip">
                <div style={{ fontWeight: 600, fontSize: 11.5 }}>{o.name} (Officer)</div>
                <div style={{ fontSize: 10, color: '#60A5FA' }}>Status: {o.status}</div>
              </Tooltip>
              <Popup>
                <div style={{ fontFamily: 'var(--font-sans, sans-serif)', fontSize: 12 }}>
                  <strong>{o.name}</strong>
                  <div style={{ color: 'var(--text-secondary)', fontSize: 11, marginTop: 2 }}>
                    Callsign: {o.callsign || 'PATROL-UNIT'} · Status: {o.status}
                  </div>
                </div>
              </Popup>
            </Marker>
          ))}

        {/* ── Live Critical Alerts ──────────────────────────────────────── */}
        {showAlerts && alerts.map((a) => (
          <Marker key={`alert-${a.alert_id}`} position={[a.lat, a.lng]} icon={ICON.alert}>
            <Tooltip direction="top" offset={[0, -10]} opacity={0.95} className="c2-map-tooltip">
              <div style={{ fontWeight: 700, fontSize: 12, color: '#EF4444' }}>CRITICAL ALERT #{a.alert_id}</div>
              <div style={{ fontSize: 10, color: '#F9FAFB' }}>{a.camera_name}</div>
            </Tooltip>
            <Popup>
              <div style={{ fontFamily: 'var(--font-sans, sans-serif)', fontSize: 12, color: 'var(--status-critical)' }}>
                <strong>CRITICAL ALERT #{a.alert_id}</strong>
                <div style={{ color: 'var(--text-secondary)', fontSize: 11, marginTop: 3 }}>
                  Location: {a.camera_name}
                </div>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}
