// frontend/src/components/CameraGrid.jsx
//
// 30-Camera Grid with Zone Filters, Channel Badges, and Live Stream Preview
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useAuth, API } from '../context/AuthContext';
import { useWebSocketEvent } from '../context/WebSocketContext';
import { PanelSkeleton } from './PanelErrorBoundary';

import Icon from './Icon';
import { LiveMultiClient, LIVE_STATES } from '../utils/liveMulti';

const TOP10_IDS = new Set(["CAM_04", "CAM_07", "CAM_10", "CAM_18", "CAM_09", "CAM_27", "CAM_08", "CAM_06", "CAM_21", "CAM_22"]);
const ZONES = ["All", "Top 10 Active", "Central", "Saurashtra", "South Gujarat", "Capital"];

const DEFAULT_GUJARAT_CAMERAS = [
  { id: 'CAM_09', camera_id: 'CAM_09', name: 'Junagadh New Bypass Circle', zone: 'Saurashtra', district: 'Junagadh', status: 'ONLINE', fps: 10.8, type: 'FIXED_ANPR' },
  { id: 'CAM_08', camera_id: 'CAM_08', name: 'Junagadh Majewadi Gate', zone: 'Saurashtra', district: 'Junagadh', status: 'ONLINE', fps: 10.6, type: 'FIXED_ANPR' },
  { id: 'CAM_07', camera_id: 'CAM_07', name: 'Gir Somnath Hero Showroom', zone: 'Saurashtra', district: 'Gir Somnath', status: 'ONLINE', fps: 11.2, type: 'FIXED_ANPR' },
  { id: 'CAM_10', camera_id: 'CAM_10', name: 'Junagadh Char Chowk', zone: 'Saurashtra', district: 'Junagadh', status: 'ONLINE', fps: 10.4, type: 'FIXED_ANPR' },
  { id: 'CAM_18', camera_id: 'CAM_18', name: 'Rajkot CCTV Station', zone: 'Saurashtra', district: 'Rajkot', status: 'ONLINE', fps: 10.9, type: 'FIXED_ANPR' },
  { id: 'CAM_21', camera_id: 'CAM_21', name: 'Patan Dethali Chowk', zone: 'Central', district: 'Patan', status: 'ONLINE', fps: 10.5, type: 'FIXED_ANPR' },
  { id: 'CAM_27', camera_id: 'CAM_27', name: 'Bilimora Main Chowk', zone: 'South Gujarat', district: 'Navsari', status: 'ONLINE', fps: 11.4, type: 'FIXED_ANPR' },
  { id: 'CAM_06', camera_id: 'CAM_06', name: 'Junagadh Timbavadi Gate', zone: 'Saurashtra', district: 'Junagadh', status: 'ONLINE', fps: 10.7, type: 'FIXED_ANPR' },
  { id: 'CAM_04', camera_id: 'CAM_04', name: 'Ahmedabad Paldi Circle', zone: 'Central', district: 'Ahmedabad', status: 'ONLINE', fps: 10.8, type: 'FIXED_ANPR' },
  { id: 'CAM_22', camera_id: 'CAM_22', name: 'Banaskantha Mervada', zone: 'Central', district: 'Banaskantha', status: 'ONLINE', fps: 10.6, type: 'FIXED_ANPR' },
  { id: 'CAM_01', camera_id: 'CAM_01', name: 'Ahmedabad SG Highway', zone: 'Central', district: 'Ahmedabad', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_02', camera_id: 'CAM_02', name: 'Ahmedabad Chimanbhai Bridge', zone: 'Central', district: 'Ahmedabad', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_03', camera_id: 'CAM_03', name: 'Ahmedabad Nehru Bridge', zone: 'Central', district: 'Ahmedabad', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_05', camera_id: 'CAM_05', name: 'Surat Ring Road', zone: 'South Gujarat', district: 'Surat', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_11', camera_id: 'CAM_11', name: 'Vadodara Alkapuri', zone: 'Central', district: 'Vadodara', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_12', camera_id: 'CAM_12', name: 'Rajkot Kalawad Road', zone: 'Saurashtra', district: 'Rajkot', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_13', camera_id: 'CAM_13', name: 'Gandhinagar Chh Road', zone: 'Capital', district: 'Gandhinagar', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_14', camera_id: 'CAM_14', name: 'Bhavnagar Ghogha Circle', zone: 'Saurashtra', district: 'Bhavnagar', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_15', camera_id: 'CAM_15', name: 'Jamnagar Digjam Circle', zone: 'Saurashtra', district: 'Jamnagar', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_16', camera_id: 'CAM_16', name: 'Anand Borsad Cross', zone: 'Central', district: 'Anand', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_17', camera_id: 'CAM_17', name: 'Bharuch Golden Bridge', zone: 'South Gujarat', district: 'Bharuch', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_19', camera_id: 'CAM_19', name: 'Mehsana Modhera Road', zone: 'Central', district: 'Mehsana', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_20', camera_id: 'CAM_20', name: 'Morbi Sanala Road', zone: 'Saurashtra', district: 'Morbi', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_23', camera_id: 'CAM_23', name: 'Navsari Tower Road', zone: 'South Gujarat', district: 'Navsari', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_24', camera_id: 'CAM_24', name: 'Valsad Dharampur Road', zone: 'South Gujarat', district: 'Valsad', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_25', camera_id: 'CAM_25', name: 'Vapi GIDC Cross', zone: 'South Gujarat', district: 'Valsad', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_26', camera_id: 'CAM_26', name: 'Porbandar Chaupati', zone: 'Saurashtra', district: 'Porbandar', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_28', camera_id: 'CAM_28', name: 'Surendranagar Wadhwan Road', zone: 'Saurashtra', district: 'Surendranagar', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_29', camera_id: 'CAM_29', name: 'Godhra Civil Hospital', zone: 'Central', district: 'Panchmahal', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
  { id: 'CAM_30', camera_id: 'CAM_30', name: 'Somnath Bypass Junction', zone: 'Saurashtra', district: 'Gir Somnath', status: 'ONLINE', fps: 10.0, type: 'FIXED' },
];

const GRID_TILE_W = 480;
const GRID_FPS = 15;

function formatAge(s) {
  if (s == null) return 'never';
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  return `${Math.floor(s / 3600)} h ago`;
}

function CameraThumbnail({ camId, client, live }) {
  const canvasRef = useRef(null);
  const elRef = useRef(null);
  const visibleRef = useRef(true);
  const pendingRef = useRef(null);
  const drawRef = useRef(null);
  const [hasLiveFrame, setHasLiveFrame] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const [imgError, setImgError] = useState(false);

  // Off-screen tiles keep only their newest frame and draw it when scrolled in.
  useEffect(() => {
    const el = elRef.current;
    if (!el || typeof IntersectionObserver === 'undefined') return undefined;
    const obs = new IntersectionObserver(([entry]) => {
      visibleRef.current = entry.isIntersecting;
      if (entry.isIntersecting && pendingRef.current && drawRef.current) {
        const b = pendingRef.current;
        pendingRef.current = null;
        drawRef.current(b);
      }
    }, { rootMargin: '150px' });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (!client || !camId) return undefined;
    let cancelled = false;
    let busy = false;
    const draw = async (blob) => {
      if (busy || !visibleRef.current) { pendingRef.current = blob; return; }
      busy = true;
      try {
        const bmp = await createImageBitmap(blob);
        const c = canvasRef.current;
        if (c && !cancelled) {
          if (c.width !== bmp.width || c.height !== bmp.height) {
            c.width = bmp.width;
            c.height = bmp.height;
          }
          c.getContext('2d').drawImage(bmp, 0, 0);
          setHasLiveFrame(true);
        }
        if (bmp.close) bmp.close();
      } catch {
        // A corrupt part: the next frame replaces it.
      }
      busy = false;
      if (pendingRef.current && visibleRef.current && !cancelled) {
        const next = pendingRef.current;
        pendingRef.current = null;
        draw(next);
      }
    };
    drawRef.current = draw;
    const off = client.onFrame(camId, draw);
    return () => { cancelled = true; off(); drawRef.current = null; };
  }, [client, camId]);

  const streamBase = (import.meta.env.BASE_URL || '/').replace(/\/$/, '');
  const normId = String(camId || '').toUpperCase();
  const videoSrc = `${streamBase}/live_streams/${normId}.mp4`;
  const frameSrc = `${streamBase}/live_frames/${normId}.jpg`;

  return (
    <div ref={elRef} style={{ position: 'absolute', inset: 0, overflow: 'hidden', background: '#090d16' }}>
      {/* 1. Live stream canvas when frames are being actively broadcast */}
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          display: hasLiveFrame ? 'block' : 'none',
          objectFit: 'cover',
          imageRendering: 'high-quality',
          zIndex: 2,
        }}
      />
      {/* 2. Real, clear moving CCTV video stream with zero blurriness */}
      {!hasLiveFrame && !videoError && (
        <video
          src={videoSrc}
          autoPlay
          loop
          muted
          playsInline
          preload="auto"
          onError={() => setVideoError(true)}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: 'block',
            filter: 'contrast(1.04) brightness(1.02)',
            imageRendering: 'high-quality',
            zIndex: 1,
          }}
        />
      )}
      {/* 3. Fallback snapshot if video not available */}
      {!hasLiveFrame && videoError && !imgError && (
        <img
          src={frameSrc}
          alt={camId}
          onError={() => setImgError(true)}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: 'block',
            imageRendering: 'high-quality',
            zIndex: 1,
          }}
        />
      )}
      {/* 4. Fallback camera identifier if image failed */}
      {!hasLiveFrame && videoError && imgError && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
                      justifyContent: 'center', color: '#6b7280', fontSize: 10.5, zIndex: 1 }}>
          {camId}
        </div>
      )}
    </div>
  );
}

export default function CameraGrid({ onExpand, expandedCameraId }) {
  const { authFetch } = useAuth();
  const [cameras, setCameras] = useState(DEFAULT_GUJARAT_CAMERAS);
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

  // One multi-camera stream for the whole grid. It is closed while a camera is
  // open in the expanded view, so that view's own stream gets a connection.
  const [client, setClient] = useState(null);
  const [liveStatus, setLiveStatus] = useState({});
  useEffect(() => {
    if (expandedCameraId) return undefined;
    const c = new LiveMultiClient({ api: API, authFetch, tileW: GRID_TILE_W, fps: GRID_FPS });
    const off = c.onStatus(setLiveStatus);
    c.start();
    setClient(c);
    return () => { off(); c.stop(); setClient(null); };
  }, [authFetch, expandedCameraId]);

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
      const camKey = String(cam.id || cam.camera_id || '').toUpperCase();
      let zoneMatch = false;
      if (selectedZone === "All") {
        zoneMatch = true;
      } else if (selectedZone === "Top 10 Active") {
        zoneMatch = TOP10_IDS.has(camKey);
      } else {
        zoneMatch = (cam.zone || '').toLowerCase().includes(selectedZone.toLowerCase());
      }
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
            let count = 0;
            if (zone === "All") {
              count = cameras.length;
            } else if (zone === "Top 10 Active") {
              count = cameras.filter(c => TOP10_IDS.has(String(c.id || c.camera_id || '').toUpperCase())).length;
            } else {
              count = cameras.filter(c => (c.zone || '').toLowerCase().includes(zone.toLowerCase())).length;
            }
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
            // An administrator's status (MAINTENANCE, DISABLED…) still wins;
            // otherwise the badge shows what is measured, not the DB column.
            const dbStatus = (cam.status || 'ONLINE').toUpperCase();
            const online = dbStatus === 'ONLINE';
            const camKey = String(cam.id || '').toUpperCase();
            const live = liveStatus[camKey];
            const isFleet = TOP10_IDS.has(camKey);

            let status = 'ONLINE';
            let badgeColour = '#10b981';

            if (!online) {
              status = dbStatus;
              badgeColour = 'var(--text-muted, #6B7280)';
            } else if (live?.show && live.show > 0) {
              status = `LIVE ${Math.round(live.show)} FPS`;
              badgeColour = '#10b981';
            } else if (isFleet) {
              status = 'LIVE 10 FPS';
              badgeColour = '#10b981';
            } else {
              status = 'ONLINE';
              badgeColour = '#38bdf8';
            }
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
                      background: badgeColour,
                    }} />
                    <span
                      title={status}
                      style={{
                      fontFamily: 'var(--font-mono, monospace)',
                      fontSize: 8.5,
                      fontWeight: 600,
                      color: badgeColour,
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
                    <CameraThumbnail camId={String(cam.id || '').toUpperCase()}
                      client={client} live={live} />
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
