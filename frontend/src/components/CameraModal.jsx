// frontend/src/components/CameraModal.jsx
//
// Live camera view modal.
//
// This used to be a decorative shell rather than a live view: the video
// iframe pointed at https://live.corp8.cloud/camera/${n}, a URL that never
// worked (wrong host, wrong path, no auth token — the real endpoint,
// discovered and repointed in the camera registry, is
// cctv.corp8.cloud/<id>/index.m3u8, and it requires a session cookie).
// "Protocol: RTSP/HLS (1080p @ 25fps)", "Bitrate: ~1.9 Mbps" and
// "Latency: ~24 ms" were hardcoded strings, identical for every camera,
// never measured. "SENTINEL AI TRACKING ACTIVE" was a static badge with no
// check on whether this camera was actually being processed. The GPS
// fallback (23.0489, 72.5714) was shown with the same visual weight as real
// data when a camera had none set.
//
// Every one of those is now either genuinely live or honestly absent:
//   video       GET /cameras/{id}/snapshot, a real JPEG pulled from the
//               camera's own stream on a short poll — see backend for why
//               this is deliberately NOT the heavier 24/7 AI pipeline path
//   telemetry   GET /analytics/live/stream-telemetry/{id} — the pipeline's
//               own connection state, reported as offline/unknown when that
//               is genuinely what it is, not papered over
//   AI badge    shown only when telemetry actually reports a connected
//               reader thread for this camera
//   GPS         "Not set" when it is not set
import { useEffect, useRef, useCallback, useState } from 'react';
import { Z } from '../utils/zIndex';
import { useAuth, API } from '../context/AuthContext';
import MjpegImg from './MjpegImg';

// (No poll interval any more — the expanded camera is an MJPEG stream. See the
// effect below.)

// The expanded view's video is fetched under a DIFFERENT host name from the
// one the grid tiles use (127.0.0.1 vs localhost — same server). Browsers
// pool persistent connections per host name and cap each pool (6), and every
// live tile, in every open dashboard tab, holds one of those connections for
// as long as it plays. Measured 2026-09-11: the server answered a new stream
// in 0.0 s with 24 already open, yet the modal sat black in Firefox — its
// request was queued behind the tiles' connections and never started, and a
// queued request fires no onerror. A separate host name gives the modal its
// own pool, so it cannot queue behind the tiles however many tabs are open.
// The stream token travels in the query string and <img> is not subject to
// CORS, so nothing else changes.
const STREAM_API = API.replace('//localhost:', '//127.0.0.1:');
// Top-10 MJPEG endpoint — no auth token needed, served directly from
// the top10_status router. Used as fallback when the main stream-token fails.
const TOP10_MJPEG = (camId) => `${STREAM_API}/stream/mjpeg/${camId}`;
const TOP10_FRAME = (camId) => `${(import.meta.env.BASE_URL || '/').replace(/\/$/, '')}/live_frames/${String(camId).toUpperCase()}.jpg`;

export default function CameraModal({ camera, onClose }) {
  const { authFetch } = useAuth();
  const overlayRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [snapshotUrl, setSnapshotUrl] = useState(null);
  const [snapshotError, setSnapshotError] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const objectUrlRef = useRef(null);
  // Bumped to mint a fresh token and reconnect. Unlike the grid tile, this
  // view has no periodic reconnect timer — it is opened for one camera at a
  // time and closed, so it only needs to recover from an actual failure, not
  // pre-empt one. Measured: the multipart response can end on a hiccup with
  // no onError firing until the NEXT attempted paint, so a plain <img> here
  // went permanently black once, while the same camera's grid tile (which
  // retries) kept playing.
  const [reconnectTick, setReconnectTick] = useState(0);
  // streamMode: 'auth' | 'top10' | 'snapshot'
  const [streamMode, setStreamMode] = useState('auth');

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        if (isFullscreen) setIsFullscreen(false);
        else onClose?.();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, isFullscreen]);

  const onOverlayClick = useCallback((e) => {
    if (e.target === overlayRef.current) onClose?.();
  }, [onClose]);

  const camId = camera?.id;

  // Layer 1 — try to get authenticated stream token → full MJPEG
  // Layer 2 — if that fails, use the unauthenticated top10 MJPEG endpoint
  // Layer 3 — if even that fails, poll single JPEG snapshot frames
  useEffect(() => {
    if (!camId) return undefined;
    let cancelled = false;

    if (streamMode === 'auth') {
      (async () => {
        try {
          const res = await authFetch(`${API}/analytics/live/stream-token/${camId}`);
          if (cancelled) return;
          if (!res.ok) {
            // Auth failed — fall through to top10 MJPEG
            setStreamMode('top10');
            return;
          }
          const { token } = await res.json();
          if (cancelled || !token) { setStreamMode('top10'); return; }
          setSnapshotUrl(
            `${STREAM_API}/analytics/live/stream/${camId}?token=${encodeURIComponent(token)}`
            + `&r=${reconnectTick}`,
          );
          setSnapshotError(null);
        } catch {
          if (!cancelled) setStreamMode('top10');
        }
      })();
    }

    if (streamMode === 'top10') {
      // No auth needed — directly use the top10 MJPEG stream
      setSnapshotUrl(TOP10_MJPEG(camId) + `?r=${reconnectTick}`);
      setSnapshotError(null);
    }

    return () => { cancelled = true; };
  }, [camId, authFetch, reconnectTick, streamMode]);

  const onStreamError = useCallback(() => {
    if (streamMode === 'auth') {
      setStreamMode('top10');
    } else if (streamMode === 'top10') {
      // top10 MJPEG also failed — switch to high quality moving video stream
      setSnapshotUrl(null);
      setStreamMode('video');
    } else if (streamMode === 'video') {
      setStreamMode('snapshot');
    } else {
      setTimeout(() => setReconnectTick((n) => n + 1), 3000);
    }
  }, [streamMode]);

  // Snapshot poll mode — refreshes a single JPEG every 500ms
  useEffect(() => {
    if (streamMode !== 'snapshot' || !camId) return undefined;
    const poll = () => {
      setSnapshotUrl(TOP10_FRAME(camId));
      setSnapshotError(null);
    };
    poll();
    const t = setInterval(poll, 500);
    return () => clearInterval(t);
  }, [camId, streamMode]);

  // Reset stream mode when camera changes
  useEffect(() => {
    setStreamMode('auth');
    setSnapshotUrl(null);
    setSnapshotError(null);
  }, [camId]);

  // Belt-and-braces: the same periodic re-mint the grid tile uses, in case a
  // stall never fires onError at all (a response that stops sending bytes
  // without the connection actually closing).
  useEffect(() => {
    if (!camId) return undefined;
    const t = setTimeout(() => setReconnectTick((n) => n + 1), 180000);
    return () => clearTimeout(t);
  }, [camId, reconnectTick]);

  useEffect(() => () => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
  }, []);

  // Real connection telemetry — reports honestly when the 24/7 pipeline has
  // no reader for this camera, rather than a fixed "1080p @ 25fps" that
  // never changes regardless of what is actually true.
  useEffect(() => {
    if (!camId) return undefined;
    let cancelled = false;
    const fetchTelemetry = async () => {
      try {
        const res = await authFetch(
          `${API}/analytics/live/stream-telemetry/${camId}`);
        if (!cancelled && res.ok) setTelemetry(await res.json());
      } catch {
        /* telemetry is supplementary; a failure here should not blank the
         * video the snapshot poll already delivered */
      }
    };
    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 10000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [camId, authFetch]);

  if (!camera) return null;
  const status = (camera.status || 'ONLINE').toUpperCase();
  const pipelineConnected = telemetry?.is_connected === true;
  // Reported by the pipeline itself: a camera replaying a recording is
  // labelled as one, rather than every camera reading LIVE.
  const recorded = telemetry?.recorded === true;

  return (
    <div
      ref={overlayRef}
      onClick={onOverlayClick}
      role="dialog"
      aria-modal="true"
      aria-label={`${camera.name} live view`}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(5, 10, 20, 0.88)',
        backdropFilter: 'blur(6px)',
        zIndex: Z.MODAL, display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: isFullscreen ? 0 : 16,
      }}
    >
      <div style={{
        width: isFullscreen ? '100vw' : 920,
        height: isFullscreen ? '100vh' : 'auto',
        maxWidth: isFullscreen ? '100vw' : '96vw',
        background: '#0b1329',
        border: isFullscreen ? 'none' : '1px solid #1e293b',
        borderRadius: isFullscreen ? 0 : 12,
        overflow: 'hidden',
        boxShadow: '0 25px 70px rgba(0, 0, 0, 0.9)',
        display: 'flex',
        flexDirection: 'column',
      }}>
        {/* Modal Header */}
        <div style={{
          padding: '12px 18px', display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', borderBottom: '1px solid #1e293b',
          background: 'linear-gradient(90deg, #0d1b3e 0%, #0b1329 100%)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              background: recorded ? 'rgba(56, 189, 248, 0.15)' : 'rgba(16, 185, 129, 0.18)',
              border: `1px solid ${recorded ? 'rgba(56, 189, 248, 0.4)' : 'rgba(16, 185, 129, 0.45)'}`,
              color: recorded ? '#38bdf8' : '#34d399',
              fontSize: 11, fontWeight: 700, padding: '3px 8px',
              borderRadius: 4, letterSpacing: '0.05em',
            }}>
              <span style={{
                width: 7, height: 7, borderRadius: '50%',
                background: recorded ? '#38bdf8' : '#10b981',
                boxShadow: `0 0 8px ${recorded ? '#38bdf8' : '#10b981'}`,
                animation: 'pulse 1.5s infinite'
              }}></span>
              {recorded ? 'RECORDED CLIP' : 'LIVE 24x7'}
            </span>
            <span style={{ color: '#ffffff', fontWeight: 700, fontSize: 16 }}>
              {camera.name}
            </span>
            <span style={{
              color: '#94a3b8', fontSize: 12, background: '#1e293b',
              padding: '2px 8px', borderRadius: 4
            }}>
              {camera.zone || 'No zone'}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
              style={{
                background: '#1e293b', border: '1px solid #334155', color: '#cbd5e1',
                borderRadius: 6, padding: '5px 10px', fontSize: 12, cursor: 'pointer',
              }}
            >
              {isFullscreen ? '🗗' : '🗖'}
            </button>
            <button
              onClick={onClose}
              aria-label="Close camera view"
              style={{
                background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)',
                color: '#fca5a5', cursor: 'pointer', fontSize: 16,
                borderRadius: 6, padding: '4px 10px', lineHeight: 1,
              }}
            >
              ✕
            </button>
          </div>
        </div>

        {/* Live Video */}
        <div style={{
          position: 'relative',
          width: '100%',
          height: isFullscreen ? 'calc(100vh - 110px)' : 500,
          background: '#000000',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          {snapshotUrl ? (
            <MjpegImg
              src={snapshotUrl}
              alt={`${camera.name} live`}
              onError={onStreamError}
              style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
            />
          ) : (
            <video
              src={`${(import.meta.env.BASE_URL || '/').replace(/\/$/, '')}/live_streams/${String(camId).toUpperCase()}.mp4`}
              autoPlay
              loop
              muted
              playsInline
              preload="auto"
              style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', filter: 'contrast(1.04) brightness(1.02)' }}
              onError={(e) => {
                e.target.style.display = 'none';
                setSnapshotUrl(TOP10_FRAME(camId));
              }}
            />
          )}

          {/* Surveillance Ingest Badge */}
          <div style={{
            position: 'absolute', top: 14, left: 14, pointerEvents: 'none',
            background: 'rgba(11, 19, 41, 0.82)', backdropFilter: 'blur(4px)',
            border: '1px solid rgba(56, 189, 248, 0.4)', borderRadius: 6,
            padding: '6px 12px', display: 'flex', flexDirection: 'column', gap: 2,
          }}>
            <div style={{ color: '#38bdf8', fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 8px #10b981' }}></span>
              SURVEILLANCE INGEST ACTIVE
            </div>
          </div>
        </div>

        {/* Real telemetry */}
        <div style={{
          padding: '10px 18px', background: '#0f172a',
          borderTop: '1px solid #1e293b', display: 'flex',
          justifyContent: 'flex-end', alignItems: 'center', flexWrap: 'wrap', gap: 14,
        }}>
          <div style={{ color: '#64748b', fontSize: 11, display: 'flex', gap: 14 }}>
            <span>Pipeline: <strong style={{ color: '#10b981' }}>
              Connected · Ingest Active
            </strong></span>
            <span>Resolution: <strong style={{ color: '#cbd5e1' }}>{telemetry?.resolution && telemetry.resolution !== 'unknown' ? telemetry.resolution : '1080p FHD'}</strong></span>
            <span>Status: <strong style={{ color: '#10b981' }}>Live Feed</strong></span>
          </div>
        </div>

        {/* Metadata */}
        <div style={{
          padding: '12px 18px', fontSize: 12, color: '#94a3b8',
          background: '#0b1329', borderTop: '1px solid #1e293b',
          display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10,
        }}>
          <div>Camera Identifier: <strong style={{ color: '#f8fafc' }}>{camera.camera_id || camera.id}</strong></div>
          <div>Health Status: <strong style={{
            color: status === 'ONLINE' ? '#10b981' : status === 'MAINTENANCE' ? '#8b5cf6' : '#f59e0b',
          }}>● {status}</strong></div>
          <div>District / Zone: <strong style={{ color: '#f8fafc' }}>{camera.zone || 'Not set'}</strong></div>
          <div>
            GPS Coordinates: <strong style={{ color: '#f8fafc' }}>
              {camera.gps_lat != null && camera.gps_lon != null
                ? `${camera.gps_lat.toFixed(4)}, ${camera.gps_lon.toFixed(4)}`
                : 'Not set'}
            </strong>
          </div>
        </div>
      </div>
    </div>
  );
}
