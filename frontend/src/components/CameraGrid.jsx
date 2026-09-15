// frontend/src/components/CameraGrid.jsx
//
// 30-Camera Grid with Zone Filters, Channel Badges, and Live Stream Preview
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useAuth, API } from '../context/AuthContext';
import { useWebSocketEvent } from '../context/WebSocketContext';
import { PanelSkeleton } from './PanelErrorBoundary';

import Icon from './Icon';
import MjpegImg from './MjpegImg';

const ZONES = ["All", "Central", "Saurashtra", "South Gujarat", "Capital"];

// Rest of component unchanged until rendering...

// Tiles used to show only a static 📹/🔴 emoji — no actual picture, live or
// otherwise. They now pull the same real JPEG CameraModal's live view uses
// (GET /cameras/{id}/snapshot, backend/routers/v1/cameras.py::camera_snapshot),
// which caches each camera's own corp8 grab for 20s server-side (a real grab
// measures 7-8.5s). Two things here exist specifically because of an earlier
// incident in this project where firing ~30 near-simultaneous corp8 requests
// tripped the portal's login-frequency lockout (see camera_heartbeat's
// circuit breaker): a tile only starts polling once IntersectionObserver
// confirms it is actually scrolled into view, and each tile's first fetch is
// staggered by its camera id rather than firing with the other 29 at once.
// The poll period sits above the server cache TTL so a poll is a real grab,
// not a wasted duplicate of one already cached.
// 25 s was set when a snapshot meant a fresh corp8 grab per tile. It no longer
// does: while the pipeline runs, the endpoint serves the frame already
// published to disk and the server caches it for 0.5 s, so a poll is a local
// file read. At 25 s the wall looked frozen — the pipeline published ~2 fps and
// the grid asked once every twenty-five seconds. 3 s keeps the tiles visibly
// alive while staying far above the cache TTL, and the IntersectionObserver +
// stagger below still stop thirty tiles firing together.
const THUMB_POLL_MS = 3000;
const THUMB_STAGGER_STEP_MS = 700;

// Tiles for these cameras play the live MJPEG stream — the same one the
// expanded view uses — instead of polling a snapshot every 3 s.
//
// A 3 s poll caps a tile at 0.33 fps however fast the pipeline publishes, so a
// camera running at 12.5 fps still looked like a slideshow on the wall. The
// stream pushes every frame the pipeline publishes, over one connection. Only
// cameras actually being ingested belong here; change the list with
// VITE_LIVE_TILE_CAMERAS (comma-separated camera ids). CAM_M1-M4 are recorded
// phone clips replayed as cameras; they stream the same way and their frames
// are stamped RECORDED VIDEO by the pipeline.
const LIVE_TILE_CAMERAS = new Set(
  String(import.meta.env.VITE_LIVE_TILE_CAMERAS || 'CAM_09,CAM_M1,CAM_M2,CAM_M3,CAM_M4')
    .split(',').map((s) => s.trim().toUpperCase()).filter(Boolean),
);
// The server ends an MJPEG response after ~20 s without a new frame, and an
// <img> whose multipart response has ended keeps its last picture without
// firing an error. The periodic re-mint is what brings a tile back after a
// pipeline restart without a page reload; kept long so it is rarely visible.
const LIVE_TILE_RECONNECT_MS = 180000;

function CameraThumbnail({ camId, online, paused }) {
  const { authFetch } = useAuth();
  const [url, setUrl] = useState(null);
  const [streamUrl, setStreamUrl] = useState(null);
  const [reconnectTick, setReconnectTick] = useState(0);
  const [visible, setVisible] = useState(false);
  const elRef = useRef(null);
  const objectUrlRef = useRef(null);
  const isLiveTile = LIVE_TILE_CAMERAS.has(String(camId || '').toUpperCase());

  useEffect(() => {
    const el = elRef.current;
    if (!el || typeof IntersectionObserver === 'undefined') { setVisible(true); return undefined; }
    const obs = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: '150px' });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Live tile: one MJPEG connection. An <img> cannot send an Authorization
  // header, so the URL carries a short-lived stream token, exactly as
  // CameraModal does.
  // Every live tile drops its connection while ANY camera is open in the
  // expanded view (`paused`), and reconnects when it closes.
  //
  // Long-lived MJPEG connections are still ordinary HTTP requests, and the
  // browser caps persistent connections per server (Firefox and Chrome: 6).
  // The live tiles already hold those; the expanded view's own stream then
  // queued behind them and never started — no onerror, just a black box
  // under a "connected" badge (measured 2026-09-11: Firefox holding 12
  // connections to :8000). Pausing only the tile for the same camera was not
  // enough. The grid is hidden behind the modal's overlay anyway, so every
  // tile gives its slot back while the modal is up.
  useEffect(() => {
    if (!isLiveTile || !camId || !visible || paused) {
      if (paused) setStreamUrl(null);
      return undefined;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await authFetch(`${API}/analytics/live/stream-token/${camId}`);
        if (cancelled || !res.ok) return;
        const { token } = await res.json();
        if (cancelled || !token) return;
        setStreamUrl(`${API}/analytics/live/stream/${camId}`
          + `?token=${encodeURIComponent(token)}&r=${reconnectTick}`);
      } catch {
        // Retried on the next reconnect tick.
      }
    })();
    const t = setTimeout(() => setReconnectTick((n) => n + 1), LIVE_TILE_RECONNECT_MS);
    return () => { cancelled = true; clearTimeout(t); };
  }, [isLiveTile, camId, visible, paused, authFetch, reconnectTick]);

  useEffect(() => {
    if (isLiveTile) return undefined;
    if (!camId || !online || !visible) return undefined;
    let cancelled = false;
    let interval;

    const fetchThumb = async () => {
      try {
        const res = await authFetch(`${API}/cameras/${camId}/snapshot`);
        if (cancelled || !res.ok) return;
        const blob = await res.blob();
        if (cancelled) return;
        const objUrl = URL.createObjectURL(blob);
        if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = objUrl;
        setUrl(objUrl);
      } catch {
        // Leave the placeholder showing; the next poll (or the next time
        // this tile scrolls into view) tries again.
      }
    };

    const initialDelay = (Number(camId) || 0) % 30 * THUMB_STAGGER_STEP_MS;
    const startTimer = setTimeout(() => {
      fetchThumb();
      interval = setInterval(fetchThumb, THUMB_POLL_MS);
    }, initialDelay);

    return () => {
      cancelled = true;
      clearTimeout(startTimer);
      if (interval) clearInterval(interval);
    };
  }, [isLiveTile, camId, online, visible, authFetch]);

  useEffect(() => () => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
  }, []);

  const src = isLiveTile ? streamUrl : url;
  // Live tiles show the WHOLE frame ('contain'). CAM_M1-M4 are portrait
  // phone clips (478x850); 'cover' in a 16:9 tile kept only a thin middle
  // band, cutting off the RECORDED VIDEO banner and the GV_ identity label —
  // the proof the tile exists to show.
  const imgStyle = {
    width: '100%', height: '100%', display: 'block',
    objectFit: isLiveTile ? 'contain' : 'cover',
  };
  return (
    <div ref={elRef} style={{ position: 'absolute', inset: 0 }}>
      {src && (isLiveTile ? (
        <MjpegImg
          src={src}
          alt=""
          onError={() => setTimeout(() => setReconnectTick((n) => n + 1), 2000)}
          style={imgStyle}
        />
      ) : (
        <img src={src} alt="" style={imgStyle} />
      ))}
    </div>
  );
}

export default function CameraGrid({ onExpand, expandedCameraId }) {
  const { authFetch } = useAuth();
  const [cameras, setCameras] = useState(null);
  const [failed, setFailed] = useState(false);
  const [selectedZone, setSelectedZone] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");

  const fetchCameras = useCallback(async () => {
    try {
      const res = await authFetch(`${API}/cameras`);
      if (res && res.ok) {
        const data = await res.json();
        const list = Array.isArray(data) ? data : (data.cameras || []);
        if (list.length > 0) {
          setCameras(list);
          setFailed(false);
        }
      }
    } catch {
      // transient failure, retry automatically
    }
  }, [authFetch]);

  useEffect(() => {
    fetchCameras();
    const interval = setInterval(fetchCameras, 10000);
    return () => clearInterval(interval);
  }, [fetchCameras]);

  useWebSocketEvent('camera_status_changed', useCallback((e) => {
    setCameras((prev) => (prev
      ? prev.map((c) => (c.id === e.camera_id ? { ...c, status: e.status } : c))
      : prev));
  }, []));

  // Channel label for a tile.
  //
  // This used to render "CH-01" for EVERY camera. It looked for digits in
  // cam.name (the names here are "Ahmedabad - Chimanbhai Bridge" — no digits)
  // and cam.camera_id (GET /cameras does not return that field at all, only
  // `id`), then fell back to parseInt(cam.id) where id is "CAM_01" — which is
  // NaN. The final expression `(((NaN - 1) % 30) + 1 || 1)` evaluates to 1, so
  // all thirty tiles were labelled channel 1.
  //
  // It also capped at 30, which would have mislabelled camera 31+ on any
  // larger estate. The number is now taken from the identifier itself, and
  // when a camera genuinely has no number in it the real id is shown rather
  // than inventing a channel for it.
  const channelLabel = (cam) => {
    if (!cam) return '—';
    // CAM_M1-M4 (the Re-ID test cameras) would otherwise read CH-01..CH-04,
    // the same labels as the real CAM_01..CAM_04.
    if (/^CAM_M\d+$/i.test(String(cam.id ?? ''))) return String(cam.id).toUpperCase();
    const match = String(cam.id ?? '').match(/(\d{1,4})/)
      || String(cam.camera_id ?? '').match(/(\d{1,4})/)
      || String(cam.name ?? '').match(/(\d{1,4})/);
    if (match) return `CH-${match[1].padStart(2, '0')}`;
    return String(cam.id ?? '—');
  };

  const filteredCameras = useMemo(() => {
    if (!cameras) return [];
    return cameras.filter(cam => {
      const zoneMatch = selectedZone === "All" || (cam.zone || '').toLowerCase().includes(selectedZone.toLowerCase());
      const searchMatch = !searchQuery || 
        (cam.name || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
        (cam.zone || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
        (cam.district || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
        (cam.camera_id || '').toLowerCase().includes(searchQuery.toLowerCase());
      return zoneMatch && searchMatch;
    });
  }, [cameras, selectedZone, searchQuery]);

  if (!cameras) return <PanelSkeleton lines={3} />;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* C2 Regional Tab Strip & Search Toolbar */}
      <div style={{
        padding: '6px 8px',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        borderBottom: '1px solid var(--border, #1F2937)',
        background: 'var(--bg-elevated, #111827)',
      }}>
        {/* Search input: 30px height */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          background: 'var(--bg-base, #0A0E1A)',
          border: '1px solid var(--border-default, #374151)',
          borderRadius: 'var(--radius-sm, 2px)',
          padding: '0 8px',
          height: 30,
        }}>
          <Icon name="search" size={13} style={{ color: 'var(--text-muted, #6B7280)' }} />
          <input
            type="text"
            placeholder={`Search ${cameras.length} cameras...`}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-primary, #F9FAFB)',
              fontSize: 11.5,
              width: '100%',
              outline: 'none',
              fontFamily: 'var(--font-sans, sans-serif)',
            }}
          />
        </div>

        {/* Regional tabs: [All (30)] [Central (5)] [Saurashtra (8)] [South (7)] [+] */}
        <div style={{ display: 'flex', gap: 4, overflowX: 'auto', paddingBottom: 2 }}>
          {ZONES.map(zone => {
            const isSelected = selectedZone === zone;
            const count = zone === "All" 
              ? cameras.length 
              : cameras.filter(c => (c.zone || '').toLowerCase().includes(zone.toLowerCase())).length;
            return (
              <button
                key={zone}
                onClick={() => setSelectedZone(zone)}
                style={{
                  background: isSelected ? 'var(--accent-primary, #2563EB)' : 'var(--bg-card, #1F2937)',
                  color: isSelected ? '#FFFFFF' : 'var(--text-secondary, #9CA3AF)',
                  border: isSelected ? '1px solid var(--accent-primary, #2563EB)' : '1px solid var(--border-default, #374151)',
                  borderRadius: 'var(--radius-sm, 2px)',
                  padding: '2px 7px',
                  fontSize: 10.5,
                  fontWeight: 500,
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                  transition: 'all 0.15s ease',
                }}
              >
                {zone === 'South Gujarat' ? 'South' : zone} ({count})
              </button>
            );
          })}
          <button
            type="button"
            title="Add Regional Filter"
            style={{
              background: 'var(--bg-card, #1F2937)',
              color: 'var(--text-muted, #6B7280)',
              border: '1px solid var(--border-default, #374151)',
              borderRadius: 'var(--radius-sm, 2px)',
              padding: '2px 7px',
              fontSize: 10.5,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            +
          </button>
        </div>
      </div>

      {/* Responsive Camera Card Grid */}
      <div
        className="panel-content"
        style={{
          flex: 1,
          overflowY: 'auto',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))',
          gap: 6,
          padding: 8,
          alignContent: 'start',
        }}
      >
        {filteredCameras.length === 0 ? (
          <div style={{ color: 'var(--text-muted, #6B7280)', fontSize: 11.5, padding: 20, textAlign: 'center', gridColumn: '1 / -1' }}>
            No cameras match filter "{selectedZone}".
          </div>
        ) : (
          filteredCameras.map((cam) => {
            const status = (cam.status || 'ONLINE').toUpperCase();
            const online = status === 'ONLINE';
            const camLabel = channelLabel(cam);

            return (
              <div
                key={cam.id}
                role="button"
                tabIndex={0}
                aria-label={`Open camera ${cam.name}`}
                onClick={() => onExpand?.(cam)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onExpand?.(cam); }
                }}
                style={{
                  background: 'var(--bg-card, #1F2937)',
                  border: '1px solid var(--border-default, #374151)',
                  borderRadius: 'var(--radius-md, 4px)',
                  padding: 6,
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 5,
                  transition: 'border-color 0.15s ease',
                }}
              >
                {/* Header: CH-ID + Status Pill */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{
                    fontFamily: 'var(--font-mono, monospace)',
                    fontSize: 10.5,
                    fontWeight: 600,
                    color: '#93C5FD',
                  }}>
                    {camLabel}
                  </span>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                    padding: '1px 5px',
                    borderRadius: 2,
                    background: 'var(--bg-base, #0A0E1A)',
                    border: '1px solid var(--border, #1F2937)',
                  }}>
                    <span style={{
                      width: 5,
                      height: 5,
                      borderRadius: '50%',
                      background: online ? 'var(--status-success, #10B981)' : 'var(--text-muted, #6B7280)',
                    }} />
                    <span style={{
                      fontFamily: 'var(--font-mono, monospace)',
                      fontSize: 8.5,
                      fontWeight: 600,
                      color: online ? 'var(--status-success, #10B981)' : 'var(--text-muted, #6B7280)',
                    }}>
                      {status}
                    </span>
                  </div>
                </div>

                {/* 16:9 Dark Thumbnail */}
                <div style={{
                  position: 'relative',
                  width: '100%',
                  aspectRatio: '16 / 9',
                  background: 'var(--bg-base, #0A0E1A)',
                  borderRadius: 2,
                  overflow: 'hidden',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px solid var(--border, #1F2937)',
                }}>
                  {online ? (
                    <CameraThumbnail camId={cam.id} online={online}
                      paused={!!expandedCameraId} />
                  ) : (
                    <Icon name="camera" size={18} style={{ color: 'var(--text-muted, #4B5563)' }} />
                  )}
                </div>

                {/* Footer: Location Name */}
                <div
                  title={cam.name}
                  style={{
                    fontSize: 10.5,
                    color: 'var(--text-secondary, #D1D5DB)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {cam.name}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
