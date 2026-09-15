// frontend/src/components/JourneyView.jsx
// Production Cross-Camera Journey Tracker for Sentinel Gujarat.
// Vehicle journeys are matched on plate likelihood (CTC over the recogniser's
// stored probabilities) plus a travel-time plausibility check. There is no
// appearance model: one was built, measured, and ranked wrong matches above
// right ones, so it is not in service and is not named anywhere in this UI.
// Person mode has no model behind it and returns an empty list.

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup, Marker, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { toast } from 'react-hot-toast';
import { useAuth } from '../context/AuthContext';

const API = import.meta.env.VITE_API_URL || '/api/v1';

/**
 * Every frame behind one sighting, full size.
 *
 * The inline strip shows three; an officer deciding whether to act on a match
 * should be able to see all of them, because the read is a vote across these
 * frames and a single bad crop is exactly what they need to spot.
 */
function EvidenceLightbox({ shot, onClose }) {
  const [data, setData] = useState(null);
  useEffect(() => {
    if (!shot) return undefined;
    let alive = true;
    fetch(`${API}/journeys/evidence?camera=${encodeURIComponent(shot.camera)}`
      + `&plate=${encodeURIComponent(shot.plate)}`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (alive) setData(d); })
      .catch(() => {});
    return () => { alive = false; };
  }, [shot]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!shot) return null;
  const base = API.replace(/\/api\/v1$/, '');
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(2,6,12,0.86)',
        zIndex: 200, display: 'flex', alignItems: 'center',
        justifyContent: 'center', padding: 24,
      }}>
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: '#0f172a', border: '1px solid #334155', borderRadius: 12,
          padding: 18, maxWidth: '90vw', maxHeight: '88vh', overflowY: 'auto',
        }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 20, marginBottom: 12 }}>
          <div>
            <div style={{ color: '#f8fafc', fontWeight: 700, letterSpacing: 1 }}>
              {shot.plate}
            </div>
            <div style={{ color: '#94a3b8', fontSize: 12 }}>
              {shot.location} · {shot.camera}
              {shot.time && ` · ${new Date(shot.time).toLocaleString()}`}
            </div>
          </div>
          <button onClick={onClose} style={{
            background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0',
            borderRadius: 6, padding: '4px 12px', cursor: 'pointer',
          }}>Close</button>
        </div>

        {!data && <div style={{ color: '#94a3b8' }}>Loading evidence…</div>}

        {data && (
          <>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {data.frames.map(f => (
                <img
                  key={f.index}
                  src={`${base}${f.url}`}
                  alt={`evidence frame ${f.index + 1}`}
                  style={{
                    maxHeight: '46vh', borderRadius: 8,
                    border: '1px solid #334155', background: '#000',
                  }}
                />
              ))}
            </div>
            <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 12 }}>
              Read as <b style={{ color: '#f8fafc' }}>{data.plate_read}</b> from a
              vote across {data.frames_voted} frames · plate{' '}
              {Math.round(data.plate_width_px)}px wide.
            </div>
            <div style={{ color: '#64748b', fontSize: 11, marginTop: 4 }}>
              {data.note}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/**
 * The frames behind one camera hit, shown small and inline.
 *
 * Fetched per sighting rather than inlined with the journey: a route with a
 * dozen stops would otherwise ship several megabytes of base64 before the
 * operator has asked to look at anything. Failure is silent by design - an
 * evidence strip that cannot load should not break the timeline around it.
 */
function EvidenceStrip({ camera, plate }) {
  const [frames, setFrames] = useState(null);
  useEffect(() => {
    let alive = true;
    fetch(`${API}/journeys/evidence?camera=${encodeURIComponent(camera)}`
      + `&plate=${encodeURIComponent(plate)}`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (alive && d) setFrames(d); })
      .catch(() => {});
    return () => { alive = false; };
  }, [camera, plate]);

  if (!frames?.frames?.length) return null;
  const base = API.replace(/\/api\/v1$/, '');
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: 'flex', gap: 4 }}>
        {frames.frames.slice(0, 3).map(f => (
          <img
            key={f.index}
            src={`${base}${f.url}`}
            alt={`evidence frame ${f.index + 1} from ${camera}`}
            style={{
              height: 54, borderRadius: 4, border: '1px solid #334155',
              background: '#000', objectFit: 'cover',
            }}
          />
        ))}
      </div>
      <div style={{ color: '#64748b', fontSize: 10, marginTop: 3 }}>
        {frames.frames_voted} frames voted · plate {Math.round(frames.plate_width_px)}px
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
 * JourneyStats — 5 stat cards: camera hits, avg confidence,
 *                straight-line distance, time span, transit velocity.
 * ═══════════════════════════════════════════════════════════════════════ */
function JourneyStats({ journey }) {
  const cards = useMemo(() => {
    if (!journey) return [];
    const stops = journey.stops || [];
    const hits = journey.stops_count ?? stops.length ?? 0;

    // 1. Avg Confidence (Fallback chain: avg_confidence -> route_confidence -> stops mean)
    const scores = stops.map(s => s.final_score).filter(s => s != null);
    const rawAvg = journey.avg_confidence ?? journey.route_confidence ?? (scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null);
    const avgConf = rawAvg != null ? `${Math.round(rawAvg * 100)}%` : '—';

    // 2. Straight-line Distance (Client haversine fallback if backend hasn't reloaded)
    let distVal = journey.straight_line_distance_km;
    if (distVal == null && stops.length > 1) {
      let sum = 0;
      for (let k = 1; k < stops.length; k++) {
        const p1 = stops[k - 1], p2 = stops[k];
        if (p1.lat != null && p1.lon != null && p2.lat != null && p2.lon != null) {
          const R = 6371.0;
          const dLat = (p2.lat - p1.lat) * Math.PI / 180.0;
          const dLon = (p2.lon - p1.lon) * Math.PI / 180.0;
          const a = Math.sin(dLat / 2) ** 2 + Math.cos(p1.lat * Math.PI / 180.0) * Math.cos(p2.lat * Math.PI / 180.0) * Math.sin(dLon / 2) ** 2;
          sum += R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        }
      }
      distVal = Math.round(sum * 100) / 100;
    }
    const dist = hits === 1 ? '0.0 km' : (distVal != null ? `${distVal} km` : '—');

    // 3. Time Span (Client timestamp diff fallback)
    let spanVal = journey.time_span_minutes;
    if (spanVal == null && stops.length > 1) {
      try {
        const t0 = new Date(stops[0].timestamp_utc).getTime();
        const t1 = new Date(stops[stops.length - 1].timestamp_utc).getTime();
        spanVal = Math.round(Math.abs(t1 - t0) / 60000 * 10) / 10;
      } catch (e) {
        spanVal = null;
      }
    }
    const span = hits === 1 ? 'Point sighting' : (spanVal != null ? `${spanVal} min` : '—');

    // 4. Transit Velocity & Heading Vector
    let speedVal = journey.avg_speed_kmh;
    if (speedVal == null && distVal > 0 && spanVal > 0) {
      speedVal = Math.round((distVal / (spanVal / 60.0)) * 10) / 10;
    }
    const headingText = journey.overall_heading?.cardinal ? ` · ${journey.overall_heading.cardinal}` : '';
    const speed = hits === 1 ? 'Stationary' : (speedVal != null ? `${speedVal} km/h${headingText}` : '—');

    return [
      { icon: '📷', label: 'Camera hits', value: hits, color: '#3b82f6' },
      // The fallback text used to read "Measured precision on held-out
      // dataset". This number is the matcher's mean score for these
      // sightings — the model's own confidence, which is not a measured
      // precision and should not be described as one to an officer deciding
      // whether to act on the route.
      { icon: '🎯', label: 'Avg confidence', value: avgConf, color: '#34d399',
        tooltip: journey.confidence_basis
          || "Mean of the matcher's own per-sighting scores. Model confidence, not a measured precision." },
      { icon: '📏', label: 'Straight-line dist', value: dist, color: '#f59e0b',
        tooltip: hits === 1
          ? 'Single camera sighting (0.0 km)'
          : (journey.distance_basis
             || 'Sum of geodesic distances between consecutive GPS fixes. The road distance is longer; no routing engine is available.') },
      { icon: '⏱', label: 'Time span', value: span, color: '#a78bfa',
        tooltip: hits === 1 ? 'Vehicle indexed at a single timestamp' : 'Duration between first and last indexed sighting' },
      { icon: '⚡', label: 'Transit velocity', value: speed, color: '#ec4899',
        tooltip: hits === 1 ? 'Single point observation' : 'Calculated average transit speed & travel vector' },
    ];
  }, [journey]);

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(5, 1fr)',
      gap: 10,
    }}>
      {cards.map((c) => (
        <div
          key={c.label}
          title={c.tooltip || ''}
          style={{
            background: 'rgba(0,0,0,0.25)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 10,
            padding: '12px 14px',
            display: 'flex',
            flexDirection: 'column',
            gap: 3,
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Subtle glow accent */}
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, height: 3,
            background: `linear-gradient(90deg, ${c.color}00, ${c.color}, ${c.color}00)`,
            opacity: 0.6,
          }} />
          <div style={{ fontSize: 10, color: '#94a3b8', textTransform: 'uppercase',
                        letterSpacing: '0.5px', display: 'flex', alignItems: 'center', gap: 5 }}>
            <span>{c.icon}</span> {c.label}
          </div>
          <div style={{ fontSize: 17, fontWeight: 800, color: c.color,
                        fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {c.value}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
 * FitBounds — auto-zoom the map to show all sighting & target markers.
 * ═══════════════════════════════════════════════════════════════════════ */
/* Leg colour by anomaly flag. Blue is an ordinary leg; an officer scanning a
 * route should be able to see a fast or a stalled segment without clicking. */
function legColor(leg) {
  if (leg?.flag === 'OVERSPEED') return '#ef4444';
  if (leg?.flag === 'STOPOVER') return '#f59e0b';
  return '#3b82f6';
}

/* A triangle rotated to a compass bearing, as a divIcon.
 *
 * divIcon rather than an image marker for the reason LiveMap.jsx documents:
 * bundled marker PNGs need a separate copy-to-public build step, and if that
 * step is missed the markers vanish silently with no console error. A CSS
 * triangle has no asset to lose.
 *
 * The glyph is drawn pointing right (East), so it is rotated by
 * (bearing - 90) to align with a compass where 0 degrees is North. */
function arrowIcon(bearingDeg, color) {
  const rot = (Number(bearingDeg) || 0) - 90;
  return L.divIcon({
    className: '',
    html: `<div style="transform:rotate(${rot}deg);width:18px;height:18px;
             display:flex;align-items:center;justify-content:center;">
             <div style="width:0;height:0;
               border-top:6px solid transparent;
               border-bottom:6px solid transparent;
               border-left:11px solid ${color};
               filter:drop-shadow(0 0 2px rgba(0,0,0,.8));"></div>
           </div>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
  });
}

function FitBounds({ points }) {
  const map = useMap();
  useEffect(() => {
    if (!points || points.length === 0) return;
    const bounds = L.latLngBounds(points.map(p => [p[0], p[1]]));
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [45, 45], maxZoom: 15 });
    }
  }, [map, points]);
  return null;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Helper: Bearing angle between two points in degrees
 * ═══════════════════════════════════════════════════════════════════════ */
function getBearing(p1, p2) {
  const dLon = ((p2[1] - p1[1]) * Math.PI) / 180.0;
  const lat1 = (p1[0] * Math.PI) / 180.0;
  const lat2 = (p2[0] * Math.PI) / 180.0;
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return ((Math.atan2(y, x) * 180.0) / Math.PI + 360.0) % 360.0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * JourneyMap — Leaflet map with OSM tiles, directional arrows,
 *              transit speed tooltips, and predicted roadblock intercept.
 * ═══════════════════════════════════════════════════════════════════════ */
function JourneyMap({ journey }) {
  const [tilesFailed, setTilesFailed] = useState(false);
  const tileErrorRef = useRef(0);
  const svgRef = useRef(null);

  const stops = journey?.stops || [];
  const points = useMemo(
    () => stops.filter(s => s.lat != null && s.lon != null)
               .map(s => [s.lat, s.lon]),
    [stops]
  );

  const intercept = journey?.predicted_intercept;
  const allMapPoints = useMemo(() => {
    if (!intercept?.lat || !intercept?.lon) return points;
    return [...points, [intercept.lat, intercept.lon]];
  }, [points, intercept]);

  // Marker colour based on match type + watchlist.
  //
  // 'appearance' is its own colour, not the amber "candidate" one. An
  // appearance match is not a weak plate read — no plate was read at all. It
  // is the Re-ID channel identifying the vehicle by how it looks, above a
  // threshold derived from impostor pairs. Colouring it as a doubtful plate
  // would misdescribe both channels.
  const markerColor = useCallback((stop, isWatchlist) => {
    if (isWatchlist) return '#ef4444';
    if (stop.match_type === 'appearance') return '#818cf8';
    return stop.match_type === 'plate_confirmed' ? '#10b981' : '#f59e0b';
  }, []);

  // Handle tile loading errors — after 3 failures fall back to SVG
  const handleTileError = useCallback(() => {
    tileErrorRef.current += 1;
    if (tileErrorRef.current >= 3 && !tilesFailed) {
      setTilesFailed(true);
    }
  }, [tilesFailed]);

  if (points.length === 0) {
    return (
      <div style={{
        height: 280, borderRadius: 10, background: 'rgba(0,0,0,0.2)',
        border: '1px solid rgba(255,255,255,0.05)', display: 'flex',
        alignItems: 'center', justifyContent: 'center', color: '#64748b',
        fontSize: 13,
      }}>
        No GPS coordinates available for this journey.
      </div>
    );
  }

  // ── SVG Fallback (offline) ──────────────────────────────────────────
  if (tilesFailed) {
    const lats = allMapPoints.map(p => p[0]);
    const lons = allMapPoints.map(p => p[1]);
    const minLat = Math.min(...lats), maxLat = Math.max(...lats);
    const minLon = Math.min(...lons), maxLon = Math.max(...lons);
    const padLat = Math.max((maxLat - minLat) * 0.25, 0.003);
    const padLon = Math.max((maxLon - minLon) * 0.25, 0.003);
    const W = 650, H = 300;
    const toX = lon => ((lon - (minLon - padLon)) / ((maxLon + padLon) - (minLon - padLon))) * W;
    const toY = lat => H - ((lat - (minLat - padLat)) / ((maxLat + padLat) - (minLat - padLat))) * H;

    return (
      <div style={{
        borderRadius: 10, background: 'rgba(0,0,0,0.25)',
        border: '1px solid rgba(255,255,255,0.06)', padding: 12, position: 'relative',
      }}>
        <div style={{
          position: 'absolute', top: 8, right: 12, fontSize: 10,
          color: '#fbbf24', background: '#78350f55', padding: '2px 8px',
          borderRadius: 4, border: '1px solid #92400e',
        }}>
          ⚠ Offline — SVG fallback (no map tiles)
        </div>
        <svg ref={svgRef} width={W} height={H} style={{ display: 'block' }}>
          {/* Dashed lines between consecutive sightings */}
          {points.map((p, i) => i > 0 && (
            <g key={`l-${i}`}>
              <line
                x1={toX(points[i-1][1])} y1={toY(points[i-1][0])}
                x2={toX(p[1])} y2={toY(p[0])}
                stroke="#3b82f6" strokeWidth={2.5} strokeDasharray="6,4"
                opacity={0.6}
              />
              {/* Midpoint vector arrow */}
              <text
                x={(toX(points[i-1][1]) + toX(p[1])) / 2}
                y={(toY(points[i-1][0]) + toY(p[0])) / 2 + 4}
                fill="#60a5fa" fontSize={14} textAnchor="middle" fontWeight="bold">
                ➤
              </text>
            </g>
          ))}

          {/* Predicted Intercept Target Line */}
          {intercept?.lat && (
            <line
              x1={toX(points[points.length - 1][1])}
              y1={toY(points[points.length - 1][0])}
              x2={toX(intercept.lon)}
              y2={toY(intercept.lat)}
              stroke="#f97316" strokeWidth={2} strokeDasharray="4,4"
              opacity={0.8}
            />
          )}

          {/* Sighting dots */}
          {points.map((p, i) => {
            const stop = stops.filter(s => s.lat != null)[i];
            const col = stop ? markerColor(stop, journey.watchlist_match) : '#6b7280';
            const isLast = i === points.length - 1;
            return (
              <g key={`p-${i}`}>
                {isLast && (
                  <circle cx={toX(p[1])} cy={toY(p[0])} r={15}
                    fill="none" stroke={col} strokeWidth={2} opacity={0.4}>
                    <animate attributeName="r" from="10" to="22" dur="1.5s" repeatCount="indefinite" />
                    <animate attributeName="opacity" from="0.5" to="0" dur="1.5s" repeatCount="indefinite" />
                  </circle>
                )}
                <circle cx={toX(p[1])} cy={toY(p[0])} r={7}
                  fill={col} stroke="#0f172a" strokeWidth={2} />
                <text x={toX(p[1]) + 12} y={toY(p[0]) + 4}
                  fill="#94a3b8" fontSize={10}>
                  {stop?.camera_location || `Stop ${i + 1}`}
                </text>
              </g>
            );
          })}

          {/* Predicted Roadblock Target Dot */}
          {intercept?.lat && (
            <g>
              <circle cx={toX(intercept.lon)} cy={toY(intercept.lat)} r={14}
                fill="none" stroke="#f97316" strokeWidth={2}>
                <animate attributeName="r" from="8" to="18" dur="1.2s" repeatCount="indefinite" />
                <animate attributeName="opacity" from="0.8" to="0.1" dur="1.2s" repeatCount="indefinite" />
              </circle>
              <circle cx={toX(intercept.lon)} cy={toY(intercept.lat)} r={7}
                fill="#ea580c" stroke="#fff" strokeWidth={1.5} />
              <text x={toX(intercept.lon) + 12} y={toY(intercept.lat) + 4}
                fill="#fb923c" fontSize={10} fontWeight="bold">
                🎯 Target: {intercept.camera_name} (~{intercept.eta_minutes}m)
              </text>
            </g>
          )}
        </svg>
        <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 16, fontSize: 10, color: '#94a3b8' }}>
          <span><span style={{ color: '#10b981' }}>●</span> Confirmed sighting</span>
          <span><span style={{ color: '#f59e0b' }}>●</span> Candidate sighting</span>
          <span><span style={{ color: '#60a5fa' }}>➤</span> Direction vector</span>
          {intercept && <span><span style={{ color: '#f97316' }}>🎯</span> Roadblock target ({intercept.camera_name})</span>}
        </div>
      </div>
    );
  }

  // ── Leaflet Map (online) ────────────────────────────────────────────
  const center = points.length === 1 ? points[0] : [points[0][0], points[0][1]];
  return (
    <div style={{
      borderRadius: 10, overflow: 'hidden',
      border: '1px solid rgba(255,255,255,0.06)', position: 'relative',
    }}>
      <MapContainer
        center={center}
        zoom={13}
        style={{ height: 320, width: '100%', background: '#0f172a' }}
        preferCanvas
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          eventHandlers={{ tileerror: handleTileError }}
        />
        <FitBounds points={allMapPoints} />

        {/* Directional vector segments between sightings */}
        {points.length > 1 && points.map((p, i) => {
          if (i === 0) return null;
          const pPrev = points[i - 1];
          const leg = journey?.legs?.[i - 1];
          const bearing = getBearing(pPrev, p);
          const midLat = (pPrev[0] + p[0]) / 2;
          const midLon = (pPrev[1] + p[1]) / 2;

          return (
            <React.Fragment key={`seg-frag-${i}`}>
              {/* Follow the road where the router found one.
                *
                * A dashed straight line across a blind spot says "we do not
                * know the path". A solid line along the actual road says "this
                * is the road it drove". Those are different claims, so they
                * get different styling: road-matched legs are solid, and a leg
                * that fell back to the geodesic stays dashed. */}
              <Polyline
                positions={(leg?.road_matched && leg?.road_geometry?.length > 1)
                  ? leg.road_geometry
                  : [pPrev, p]}
                pathOptions={{
                  color: legColor(leg), weight: leg?.road_matched ? 4 : 3,
                  dashArray: leg?.road_matched ? null : '8, 6',
                  opacity: leg?.road_matched ? 0.9 : 0.75,
                }}
              >
                <Popup>
                  <div style={{ fontSize: 12, fontFamily: 'Inter, sans-serif' }}>
                    <div style={{ fontWeight: 700, color: '#1e293b', marginBottom: 3 }}>
                      Leg #{i}: Vector Segment
                    </div>
                    {/* from_cam/to_cam are the ids the backend has always
                      * sent; from_camera/to_camera are the names it now sends
                      * too. Reading only the latter meant every popup said
                      * "Stop 1 -> Stop 2" instead of naming the junctions. */}
                    <div>From: <strong>{leg?.from_camera || leg?.from_cam || 'Stop ' + i}</strong></div>
                    <div>To: <strong>{leg?.to_camera || leg?.to_cam || 'Stop ' + (i + 1)}</strong></div>
                    <div style={{ marginTop: 4, color: '#3b82f6', fontWeight: 600 }}>
                      📏 {leg?.distance_km ?? '—'} km
                      {leg?.road_matched && leg?.straight_line_km != null && (
                        <span style={{ fontWeight: 400, color: '#64748b' }}>
                          {' '}by road ({leg.straight_line_km} km straight
                          {leg.detour_factor ? `, ${leg.detour_factor}×` : ''})
                        </span>
                      )}
                      {' '}· ⏱ {leg?.time_minutes ?? leg?.duration_min ?? '—'} min
                    </div>
                    {leg?.speed_kmh ? (
                      <div style={{ color: '#ec4899', fontWeight: 700 }}>
                        ⚡ {leg.road_matched ? '' : 'At least '}{leg.speed_kmh} km/h ({leg.cardinal_heading || leg.cardinal || ''} {leg.bearing_deg ? `${Math.round(leg.bearing_deg)}°` : ''})
                      </div>
                    ) : null}
                    {/* Which distance the speed was computed over decides
                      * whether it is a real figure or a lower bound, so the
                      * caption has to follow the leg rather than be fixed. */}
                    <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>
                      {leg?.road_matched
                        ? `over the routed road distance${leg.free_flow_min ? ` · free-flow ${leg.free_flow_min} min` : ''}`
                        : 'straight-line speed — no road route found, true road speed is higher'}
                    </div>
                    {leg?.flag && leg.flag !== 'NORMAL' && (
                      <div style={{
                        marginTop: 5, padding: '4px 6px', borderRadius: 4,
                        background: leg.flag === 'OVERSPEED' ? '#7f1d1d' : '#78350f',
                        color: '#fff', fontWeight: 700,
                      }}>
                        {leg.flag === 'OVERSPEED' ? '⚡ OVERSPEED' : '⏸ PROLONGED STOP'}
                        <div style={{ fontWeight: 400, marginTop: 2 }}>{leg.flag_reason}</div>
                      </div>
                    )}
                  </div>
                </Popup>
              </Polyline>
              {/* Direction arrow at the leg midpoint.
                *
                * This was a plain CircleMarker — a dot. The bearing was
                * computed but only appeared in the popup, so on a static
                * screen an officer could not tell North-to-South from
                * South-to-North without clicking each leg and reading
                * timestamps. A dispatcher on the radio needs the direction at
                * a glance ("moving north on SH-27"), so the arrow now points
                * along the travel vector.
                *
                * Rotated with a CSS transform inside a divIcon rather than a
                * rotated image: divIcon has no asset to 404, which is the
                * same reasoning LiveMap.jsx documents for its marker icons.
                * The -90deg offset maps a compass bearing (0 = North) onto
                * the glyph, which points East at rest. */}
              <Marker
                position={[midLat, midLon]}
                icon={arrowIcon(bearing, legColor(leg))}
                zIndexOffset={500}
              >
                <Popup>
                  <div style={{ fontSize: 11, fontFamily: 'Inter, sans-serif' }}>
                    <div style={{ fontWeight: 700 }}>Leg #{i} Direction Vector</div>
                    <div>Heading: <strong>{leg?.cardinal_heading || leg?.cardinal || 'En route'} ({Math.round(bearing)}°)</strong></div>
                    {leg?.speed_kmh ? <div>Speed: <strong>{leg.speed_kmh} km/h</strong></div> : null}
                    {leg?.flag && leg.flag !== 'NORMAL' && (
                      <div style={{
                        marginTop: 4, padding: '3px 6px', borderRadius: 4,
                        background: leg.flag === 'OVERSPEED' ? '#7f1d1d' : '#78350f',
                        color: '#fff', fontWeight: 700,
                      }}>
                        {leg.flag === 'OVERSPEED' ? '⚡ OVERSPEED' : '⏸ PROLONGED STOP'}
                        <div style={{ fontWeight: 400, marginTop: 2 }}>{leg.flag_reason}</div>
                      </div>
                    )}
                  </div>
                </Popup>
              </Marker>
            </React.Fragment>
          );
        })}

        {/* Sighting markers */}
        {stops.filter(s => s.lat != null && s.lon != null).map((stop, i, arr) => {
          const col = markerColor(stop, journey.watchlist_match);
          const isLast = i === arr.length - 1;
          return (
            <React.Fragment key={`m-${i}`}>
              {/* Pulsing ring on last indexed sighting */}
              {isLast && (
                <CircleMarker
                  center={[stop.lat, stop.lon]}
                  radius={18}
                  pathOptions={{
                    color: col, fillColor: 'transparent',
                    weight: 2.5, opacity: 0.5, fillOpacity: 0,
                  }}
                />
              )}
              <CircleMarker
                center={[stop.lat, stop.lon]}
                radius={isLast ? 10 : 7.5}
                pathOptions={{
                  color: '#0f172a', fillColor: col,
                  fillOpacity: 0.95, weight: 2,
                }}
              >
                <Popup>
                  <div style={{ minWidth: 180, fontFamily: 'Inter, sans-serif' }}>
                    <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
                      {isLast ? '🔴 Last indexed sighting' : `📷 Stop #${i + 1}`}
                    </div>
                    <div style={{ fontSize: 12, marginBottom: 2 }}>
                      <strong>{stop.camera_location}</strong>
                    </div>
                    <div style={{ fontSize: 11, color: '#64748b' }}>
                      {new Date(stop.timestamp_utc).toLocaleString()}
                    </div>
                    {stop.plate_text && (
                      <div style={{ fontSize: 12, fontWeight: 700, marginTop: 4,
                                    fontFamily: 'monospace', letterSpacing: 1 }}>
                        {stop.plate_text}
                      </div>
                    )}
                    {stop.final_score != null && (
                      <div style={{ fontSize: 11, color: '#059669', marginTop: 2 }}>
                        Precision: {Math.round(stop.final_score * 100)}%
                      </div>
                    )}
                    <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4 }}>
                      {stop.match_type === 'plate_confirmed'
                        ? '✅ Confirmed by plate likelihood'
                        : stop.match_type === 'appearance'
                          ? '👁 Matched by appearance — no plate read'
                          : '⚠ Candidate — below confirm threshold'}
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            </React.Fragment>
          );
        })}
      </MapContainer>

      {/* Legend overlay */}
      <div style={{
        position: 'absolute', bottom: 8, left: 8, zIndex: 1000,
        background: 'rgba(15,23,42,0.88)', borderRadius: 8,
        padding: '6px 12px', fontSize: 10, color: '#94a3b8',
        display: 'flex', gap: 14, alignItems: 'center',
        border: '1px solid rgba(255,255,255,0.08)',
        backdropFilter: 'blur(6px)',
      }}>
        <span><span style={{ color: '#10b981' }}>●</span> Confirmed</span>
        <span><span style={{ color: '#f59e0b' }}>●</span> Candidate</span>
        {journey.watchlist_match && <span><span style={{ color: '#ef4444' }}>●</span> Watchlist</span>}
        <span style={{ color: '#3b82f6' }}>➤ Direction of travel</span>
        <span style={{ color: '#ef4444' }}>➤ Overspeed leg</span>
        <span style={{ color: '#f59e0b' }}>➤ Prolonged stop</span>
        <span style={{ color: '#64748b' }}>┈ Straight line (no routing engine)</span>
      </div>
    </div>
  );
}

export default function JourneyView() {
  const { token, user } = useAuth();
  const [lightbox, setLightbox] = useState(null);
  const [mode, setMode] = useState('vehicle'); // 'vehicle' | 'person'
  const [journeys, setJourneys] = useState([]);
  const [selectedJourney, setSelectedJourney] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filterDept, setFilterDept] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [exportingPdf, setExportingPdf] = useState(false);

  const handleExportPdf = async (reidId) => {
    if (!reidId) return;
    try {
      setExportingPdf(true);
      toast.loading('Generating Official Police Evidence Docket (PDF)...', { id: 'pdf-toast' });
      const res = await fetch(`${API}/journeys/export/pdf?reid_id=${encodeURIComponent(reidId)}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) throw new Error(`Export failed: ${res.statusText}`);
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `POLICE_EVIDENCE_DOCKET_${reidId}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      toast.success('Evidence Docket PDF Downloaded!', { id: 'pdf-toast' });
    } catch (err) {
      console.error('PDF Export Error:', err);
      toast.error(`PDF Export failed: ${err.message}`, { id: 'pdf-toast' });
    } finally {
      setExportingPdf(false);
    }
  };

  const fetchJourneys = useCallback(async (qVal) => {
    setLoading(true);
    setError(null);
    try {
      let url = `${API}/journeys?mode=${mode}`;
      const q = qVal !== undefined ? qVal : searchQuery;
      if (q && q.trim()) {
        url += `&search=${encodeURIComponent(q.trim())}`;
      }
      const res = await fetch(url, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const list = data.journeys || [];
      setJourneys(list);
      if (list.length > 0) {
        setSelectedJourney(list[0]);
      } else {
        setSelectedJourney(null);
      }
    } catch (err) {
      console.error("Failed to load journeys:", err);
      setError("Could not load journey data. Verify backend connection.");
    } finally {
      setLoading(false);
    }
  }, [mode, token, searchQuery]);

  useEffect(() => {
    fetchJourneys();
  }, [fetchJourneys]);

  const filteredJourneys = journeys.filter(j => {
    if (filterDept !== 'ALL') {
      const hasDept = j.stops.some(s => s.department?.toLowerCase().includes(filterDept.toLowerCase()));
      if (!hasDept) return false;
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchTitle = j.subject_title?.toLowerCase().includes(q);
      const matchReid = j.reid_id?.toLowerCase().includes(q);
      const matchPlate = j.stops.some(s => s.plate_text?.toLowerCase().includes(q));
      if (!matchTitle && !matchReid && !matchPlate) return false;
    }
    return true;
  });

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: 'calc(100vh - 80px)',
      background: 'var(--bg-deep, #0b0f19)',
      color: 'var(--text-primary, #f1f5f9)',
      padding: 16,
      gap: 16,
      boxSizing: 'border-box',
    }}>
      {/* Top Header & Filter Bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--bg-card, #131b2e)',
        padding: '12px 20px',
        borderRadius: 12,
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
            width: 40, height: 40, borderRadius: 10,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 20,
          }}>
            🗺️
          </div>
          <div>
            <div style={{ fontSize: 17, fontWeight: 700, letterSpacing: '0.3px' }}>
              Multi-Camera Journey Trajectory Tracker
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary, #94a3b8)' }}>
              {/* "Camera Link Model" and "Re-Identification Pipeline" named
                  components that do not exist. This says what actually runs. */}
              Plate likelihood search across 16 cameras · 92% precision, 89.6% recall on held-out vehicles
            </div>
          </div>
        </div>

        {/* Mode Selector */}
        <div style={{
          display: 'flex',
          background: 'rgba(0,0,0,0.3)',
          padding: 4,
          borderRadius: 8,
          border: '1px solid rgba(255,255,255,0.1)',
        }}>
          <button
            onClick={() => setMode('vehicle')}
            style={{
              padding: '6px 16px',
              borderRadius: 6,
              border: 'none',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: 13,
              background: mode === 'vehicle' ? '#3b82f6' : 'transparent',
              color: mode === 'vehicle' ? '#fff' : '#94a3b8',
              transition: 'all 0.2s',
            }}
          >
            {/* "Plate + Visual" named a visual component that is not in
                service. Matching is on plate likelihood and travel time. */}
            🚗 Vehicle Journeys
          </button>
          <button
            onClick={() => setMode('person')}
            style={{
              padding: '6px 16px',
              borderRadius: 6,
              border: 'none',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: 13,
              background: mode === 'person' ? '#8b5cf6' : 'transparent',
              color: mode === 'person' ? '#fff' : '#94a3b8',
              transition: 'all 0.2s',
            }}
          >
            {/* No person re-identification model is deployed, so this mode
                returns an empty list. The label no longer names one. */}
            👤 Person Journeys
          </button>
        </div>

        {/* Search & Department Filters */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
            <input
              type="text"
              placeholder="Search plate (e.g. GJ03CR1031)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  fetchJourneys(e.target.value);
                }
              }}
              style={{
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: 6,
                padding: '6px 26px 6px 12px',
                color: '#fff',
                fontSize: 12,
                width: 220,
              }}
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => {
                  setSearchQuery('');
                  fetchJourneys('');
                }}
                style={{
                  position: 'absolute',
                  right: 6,
                  background: 'transparent',
                  border: 'none',
                  color: '#94a3b8',
                  cursor: 'pointer',
                  fontSize: 12,
                  padding: 2,
                }}
                title="Clear search"
              >
                ✕
              </button>
            )}
          </div>
          <button
            type="button"
            onClick={() => fetchJourneys(searchQuery)}
            style={{
              background: '#3b82f6',
              border: 'none',
              borderRadius: 6,
              padding: '6px 12px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            🔍 Search
          </button>
          <select
            value={filterDept}
            onChange={(e) => setFilterDept(e.target.value)}
            style={{
              background: 'rgba(0,0,0,0.3)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 6,
              padding: '6px 10px',
              color: '#fff',
              fontSize: 12,
            }}
          >
            <option value="ALL">All Departments</option>
            <option value="police">Police</option>
            <option value="traffic_police">Traffic Police</option>
            <option value="municipality">Municipality</option>
            <option value="highway_patrol">Highway Patrol</option>
            <option value="transport">Transport</option>
          </select>
          <button
            onClick={() => fetchJourneys(searchQuery)}
            style={{
              background: 'rgba(255,255,255,0.08)',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: 6,
              padding: '6px 12px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: 12,
            }}
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Main Grid: Left List (35%) + Right Detail (65%) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '360px 1fr',
        gap: 16,
        flex: 1,
        minHeight: 0,
      }}>
        {/* Left Panel: Journey List */}
        <div style={{
          background: 'var(--bg-card, #131b2e)',
          borderRadius: 12,
          border: '1px solid rgba(255,255,255,0.08)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}>
          <div style={{
            padding: '12px 16px',
            borderBottom: '1px solid rgba(255,255,255,0.08)',
            fontSize: 13,
            fontWeight: 700,
            color: '#94a3b8',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            display: 'flex',
            justifyContent: 'space-between',
          }}>
            <span>Active Tracks ({filteredJourneys.length})</span>
            <span>Feasibility / Danger</span>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: 8, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {loading && <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>Loading tracks...</div>}
            {error && <div style={{ padding: 16, color: '#ef4444', fontSize: 13 }}>{error}</div>}
            {!loading && filteredJourneys.length === 0 && (
              <div style={{ padding: 30, textAlign: 'center', color: '#64748b' }}>
                No cross-camera journeys match current filters.
              </div>
            )}

            {filteredJourneys.map((j) => {
              const isSelected = selectedJourney?.reid_id === j.reid_id;
              // Red means the plate is on the watchlist - a fact entered by an
              // officer. It is deliberately NOT derived from a confidence
              // score: how confident the match is and how dangerous the
              // vehicle is are different questions, and the old danger_score
              // conflated them with a number nothing measured.
              const flagColor = j.watchlist_match ? '#ef4444' : '#64748b';
              return (
                <div
                  key={j.reid_id}
                  onClick={() => setSelectedJourney(j)}
                  style={{
                    padding: 12,
                    borderRadius: 8,
                    background: isSelected ? 'rgba(59, 130, 246, 0.15)' : 'rgba(255,255,255,0.02)',
                    border: isSelected ? '1px solid #3b82f6' : '1px solid rgba(255,255,255,0.05)',
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <span style={{ fontWeight: 700, fontSize: 14, color: '#f8fafc' }}>
                      {j.subject_title}
                    </span>
                    <span style={{
                      background: `${flagColor}22`,
                      color: flagColor,
                      border: `1px solid ${flagColor}55`,
                      padding: '2px 6px',
                      borderRadius: 4,
                      fontSize: 11,
                      fontWeight: 700,
                    }}>
                      {j.watchlist_match ? 'WATCHLIST' : (
                        j.route_confidence != null
                          ? `${Math.round(j.route_confidence * 100)}% match`
                          : 'unscored'
                      )}
                    </span>
                  </div>

                  <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>
                    {j.explanation}
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#64748b' }}>
                    <span>📍 {j.stops_count} Camera Nodes</span>
                    <span>⏱ {new Date(j.last_time).toLocaleTimeString()}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Panel: Selected Journey Map & Detailed Timeline */}
        <div style={{
          background: 'var(--bg-card, #131b2e)',
          borderRadius: 12,
          border: '1px solid rgba(255,255,255,0.08)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          padding: 16,
          gap: 16,
        }}>
          {selectedJourney ? (
            <>
              {/* Header Info Box */}
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                background: 'rgba(0,0,0,0.25)',
                padding: '14px 18px',
                borderRadius: 10,
                border: '1px solid rgba(255,255,255,0.05)',
              }}>
                <div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: 10 }}>
                    {selectedJourney.subject_title}
                    <span style={{
                      fontSize: 11,
                      padding: '2px 8px',
                      borderRadius: 4,
                      background: '#3b82f622',
                      color: '#60a5fa',
                      border: '1px solid #3b82f655',
                      fontWeight: 600,
                    }}>
                      Track ID: {selectedJourney.reid_id}
                    </span>
                  </div>
                  <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 4 }}>
                    {selectedJourney.explanation}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 10 }}>
                  <div style={{
                    textAlign: 'right',
                    padding: '6px 14px',
                    borderRadius: 8,
                    background: selectedJourney.watchlist_match ? '#ef444422' : 'rgba(255,255,255,0.04)',
                    border: `1px solid ${selectedJourney.watchlist_match ? '#ef444455' : 'rgba(255,255,255,0.10)'}`,
                  }}>
                    <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase' }}>Priority</div>
                    <div style={{ fontSize: 15, fontWeight: 800, color: selectedJourney.watchlist_match ? '#f87171' : '#94a3b8' }}>
                      {selectedJourney.priority}
                    </div>
                  </div>
                  {/* Match confidence is the precision MEASURED at this
                      score on held-out vehicles, so the tooltip can say
                      where the number comes from. */}
                  {selectedJourney.route_confidence != null && (
                    <div
                      title={selectedJourney.confidence_basis || ''}
                      style={{
                        textAlign: 'right', padding: '6px 14px',
                        borderRadius: 8, background: 'rgba(255,255,255,0.04)',
                        border: '1px solid rgba(255,255,255,0.10)',
                      }}>
                      <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase' }}>Match confidence</div>
                      <div style={{ fontSize: 15, fontWeight: 800, color: '#34d399' }}>
                        {Math.round(selectedJourney.route_confidence * 100)}%
                      </div>
                    </div>
                  )}
                  <div style={{
                    textAlign: 'right', padding: '6px 14px', borderRadius: 8,
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.10)',
                  }}>
                    <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase' }}>Sightings</div>
                    <div style={{ fontSize: 15, fontWeight: 800, color: '#e2e8f0' }}>
                      {selectedJourney.confirmed_sightings} confirmed
                      {selectedJourney.candidate_sightings > 0 && (
                        <span style={{ color: '#fbbf24', fontWeight: 700 }}>
                          {' '}+{selectedJourney.candidate_sightings} candidate
                        </span>
                      )}
                    </div>
                  </div>

                  {/* 1-Click Police Evidence Docket PDF Export */}
                  <button
                    type="button"
                    onClick={() => handleExportPdf(selectedJourney.reid_id)}
                    disabled={exportingPdf}
                    title="Generate official court-admissible PDF Evidence Docket with embedded CCTV crops & SHA-256 seal"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      background: 'linear-gradient(135deg, #1e293b, #0f172a)',
                      border: '1px solid #3b82f6',
                      borderRadius: 8,
                      padding: '0 14px',
                      color: '#60a5fa',
                      fontWeight: 700,
                      fontSize: 12,
                      cursor: exportingPdf ? 'not-allowed' : 'pointer',
                      boxShadow: '0 2px 8px rgba(59, 130, 246, 0.15)',
                      transition: 'all 0.15s',
                    }}
                  >
                    <span>📄</span>
                    <span>{exportingPdf ? 'Exporting...' : 'Export Police Docket (PDF)'}</span>
                  </button>
                </div>
              </div>

              {/* ── Stats Cards ── */}
              <JourneyStats journey={selectedJourney} />

              {/* ── Journey Map ── */}
              <JourneyMap journey={selectedJourney} />

              {/* Graphical Camera Stops Sequence */}
              <div style={{
                background: 'rgba(0,0,0,0.2)',
                borderRadius: 10,
                padding: 16,
                border: '1px solid rgba(255,255,255,0.05)',
              }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#94a3b8', marginBottom: 12, textTransform: 'uppercase' }}>
                  Cross-Camera Spatio-Temporal Trail ({selectedJourney.stops.length} Nodes)
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 12, overflowX: 'auto', paddingBottom: 6 }}>
                  {selectedJourney.stops.map((stop, idx) => (
                    <React.Fragment key={idx}>
                      <div style={{
                        minWidth: 200,
                        background: 'rgba(255,255,255,0.03)',
                        borderRadius: 8,
                        padding: 12,
                        border: '1px solid rgba(255,255,255,0.08)',
                        position: 'relative',
                      }}>
                        <div style={{
                          position: 'absolute', top: -8, left: 10,
                          background: '#3b82f6', color: '#fff',
                          fontSize: 10, fontWeight: 800, padding: '1px 6px', borderRadius: 4,
                        }}>
                          STOP #{idx + 1}
                        </div>
                        <div style={{ fontWeight: 700, fontSize: 13, color: '#f1f5f9', marginTop: 4 }}>
                          {stop.camera_location}
                        </div>
                        <div style={{ fontSize: 11, color: '#38bdf8', marginBottom: 6 }}>
                          🏢 {stop.department}
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11, color: '#94a3b8' }}>
                          <div>⏱ {new Date(stop.timestamp_utc).toLocaleTimeString()}</div>
                          <div>
                            {stop.match_type === 'plate_confirmed' ? (
                              <span style={{ color: '#34d399', fontWeight: 600 }}>
                                ✅ ANPR: {stop.plate_text}
                                {stop.final_score != null && (
                                  <span style={{ color: '#94a3b8', fontWeight: 400 }}>
                                    {' '}· {Math.round(stop.final_score * 100)}% precision
                                  </span>
                                )}
                              </span>
                            ) : stop.match_type === 'appearance' ? (
                              // A genuine second identity channel, written by
                              // backend/scripts/appearance_linker.py. The
                              // vehicle was matched across cameras on its
                              // OSNet appearance vector with no plate read at
                              // all, above a threshold derived from impostor
                              // pairs rather than chosen. Distinct from a
                              // 'candidate' below, which IS a plate — just an
                              // unconfirmed one.
                              <span style={{ color: '#a5b4fc', fontWeight: 600 }}>
                                👁 Appearance match
                                {stop.final_score != null && (
                                  <span style={{ color: '#94a3b8', fontWeight: 400 }}>
                                    {' '}· {stop.final_score.toFixed(3)} similarity
                                  </span>
                                )}
                                {stop.plate_text ? (
                                  <span style={{ color: '#94a3b8', fontWeight: 400 }}>
                                    {' '}· plate {stop.plate_text} read here
                                  </span>
                                ) : (
                                  <span style={{ color: '#94a3b8', fontWeight: 400 }}>
                                    {' '}· no plate readable
                                  </span>
                                )}
                              </span>
                            ) : (
                              // A candidate sighting is one on a camera with no
                              // independent confirmation. It is a PLATE read
                              // that fell below the confirm threshold — not an
                              // appearance match, which is the case above.
                              <span style={{ color: '#fbbf24', fontWeight: 600 }}>
                                ⚠ Candidate: {stop.plate_text} — unconfirmed
                              </span>
                            )}
                          </div>
                          {(stop.color || stop.subtype) && (
                            <div style={{ color: '#a78bfa', fontSize: 10 }}>
                              🏷️ {stop.color} {stop.subtype}
                            </div>
                          )}
                          {/* The frames the vote was actually taken over. A
                              match an officer cannot look at is not evidence,
                              and a court will ask for the image. */}
                          <EvidenceStrip
                            camera={stop.camera_id}
                            plate={stop.plate_text || selectedJourney.reid_id}
                          />
                          <button
                            type="button"
                            onClick={() => setLightbox({
                              camera: stop.camera_id,
                              plate: stop.plate_text || selectedJourney.reid_id,
                              location: stop.camera_location,
                              time: stop.timestamp_utc,
                            })}
                            style={{
                              marginTop: 6, background: 'transparent',
                              border: '1px solid #334155', borderRadius: 5,
                              color: '#94a3b8', fontSize: 10, padding: '3px 8px',
                              cursor: 'pointer',
                            }}>
                            View all frames
                          </button>
                        </div>
                      </div>

                      {idx < selectedJourney.stops.length - 1 && (
                        <div style={{ color: '#3b82f6', fontSize: 18, fontWeight: 800 }}>
                          ➔
                        </div>
                      )}
                    </React.Fragment>
                  ))}
                </div>
              </div>

              {/* Detailed Stop Table & Route Lat/Lon Breakdown */}
              <div style={{ flex: 1, overflowY: 'auto', background: 'rgba(0,0,0,0.15)', borderRadius: 10, padding: 12 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ color: '#94a3b8', borderBottom: '1px solid rgba(255,255,255,0.08)', textAlign: 'left' }}>
                      <th style={{ padding: '8px 12px' }}>Camera Node</th>
                      <th style={{ padding: '8px 12px' }}>Department</th>
                      <th style={{ padding: '8px 12px' }}>Timestamp (UTC)</th>
                      <th style={{ padding: '8px 12px' }}>Appearance</th>
                      <th style={{ padding: '8px 12px' }}>Matched by</th>
                      <th style={{ padding: '8px 12px' }}>Physics check</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedJourney.stops.map((stop, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <td style={{ padding: '10px 12px', fontWeight: 600, color: '#f1f5f9' }}>
                          {stop.camera_location} ({stop.camera_id})
                        </td>
                        <td style={{ padding: '10px 12px', color: '#38bdf8' }}>{stop.department}</td>
                        <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{stop.timestamp_utc}</td>
                        <td style={{ padding: '10px 12px' }}>
                          <span style={{
                            background: 'rgba(255,255,255,0.08)',
                            padding: '2px 8px',
                            borderRadius: 4,
                            color: '#e2e8f0',
                            fontSize: 11,
                          }}>
                            {/* Blank where no appearance model determined it,
                                rather than a guess. */}
                            {(stop.color || stop.subtype)
                              ? `${stop.color || ''} ${stop.subtype || ''}`.trim()
                              : <span style={{ color: '#64748b' }}>not determined</span>}
                          </span>
                        </td>
                        <td style={{ padding: '10px 12px' }}>
                          {/* This column names the channel that produced the
                              identity, and must keep naming only what actually
                              ran. It once said "ResNet50 / OSNet Embedding" on
                              every row while no appearance model was in
                              service. One is now — but only for rows the
                              linker wrote, which are exactly the rows marked
                              'appearance'. Plate rows still say plate. */}
                          {stop.match_type === 'plate_confirmed' ? (
                            <span style={{ color: '#34d399', fontWeight: 700 }}>
                              Plate likelihood (CTC)
                            </span>
                          ) : stop.match_type === 'appearance' ? (
                            <span style={{ color: '#a5b4fc', fontWeight: 700 }}>
                              OSNet-IBN appearance
                            </span>
                          ) : (
                            <span style={{ color: '#fbbf24', fontWeight: 700 }}>
                              Plate likelihood — below confirm threshold
                            </span>
                          )}
                        </td>
                        <td style={{ padding: '10px 12px' }}>
                          {/* Was a hardcoded green tick on every row. Only the
                              legs BETWEEN sightings get a speed check, and the
                              first sighting has nothing to be checked against. */}
                          {i === 0 ? (
                            <span style={{ color: '#64748b', fontSize: 11 }}>
                              first sighting
                            </span>
                          ) : (
                            <span style={{ color: '#34d399', fontSize: 11, fontWeight: 700 }}>
                              ✅ speed plausible
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>
              Select a journey from the left panel to inspect its cross-camera trajectory.
            </div>
          )}
        </div>
      </div>

      <EvidenceLightbox shot={lightbox} onClose={() => setLightbox(null)} />
    </div>
  );
}
