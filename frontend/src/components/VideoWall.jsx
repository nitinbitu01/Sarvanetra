// frontend/src/components/VideoWall.jsx
//
// Every camera the operator may see, moving at once, on ONE connection.
//
// A browser opens at most six connections per host, and each MJPEG tile holds
// one, so a grid of thirty streaming tiles cannot all move. The server composes
// the fleet's published frames into a single picture instead
// (GET /analytics/live/wall) and this page shows that one stream.
//
// What each tile says is measured, not configured: its state (LIVE, REPLAY for
// a network source delivering faster than real time, RECORDED, NO SIGNAL) and
// three rates from the worker heartbeats — src (frames decoded from the
// camera), show (frames published for the screen) and ai (frames analysed).
// The footer prints the rate the wall itself is achieving. A camera whose
// source sends fewer than 15 frames a second is shown at that rate; frames are
// never repeated to make a number look better.
import { useCallback, useEffect, useState } from 'react';
import { useAuth, API } from '../context/AuthContext';
import MjpegImg from './MjpegImg';

// Its own host name, so the wall's connection is not queued behind camera
// tiles open in other tabs (see CameraModal.jsx for the measurement).
const STREAM_API = API.replace('//localhost:', '//127.0.0.1:');
// Stream tokens last ten minutes; the stream itself is authorised at connect,
// so a fresh token is only needed to reconnect.
const TOKEN_REFRESH_MS = 8 * 60 * 1000;

const LEGEND = [
  ['LIVE', '#22c55e', 'network source at or below its own frame rate'],
  ['REPLAY', '#f59e0b', 'network source faster than real time — a recording, not live'],
  ['RECORDED', '#3b82f6', 'local clip'],
  ['NO SIGNAL', '#ef4444', 'no frame for more than 4 s'],
];

export default function VideoWall() {
  const { authFetch } = useAuth();
  const [src, setSrc] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [error, setError] = useState(null);
  const [fps, setFps] = useState(15);
  const [tileW, setTileW] = useState(320);
  const [tick, setTick] = useState(0);

  const connect = useCallback(async () => {
    try {
      const res = await authFetch(`${API}/analytics/live/wall-token`);
      if (!res.ok) {
        setError(`Could not authorise the video wall (HTTP ${res.status}).`);
        return;
      }
      const data = await res.json();
      setCameras(data.cameras || []);
      setError(null);
      const cols = (data.cameras || []).length > 24 ? 6 : 5;
      setSrc(
        `${STREAM_API}/analytics/live/wall?token=${encodeURIComponent(data.token)}`
        + `&cols=${cols}&tile_w=${tileW}&fps=${fps}&r=${Date.now()}`,
      );
    } catch (e) {
      setError(`Video wall unavailable: ${e.message}`);
    }
  }, [authFetch, fps, tileW]);

  useEffect(() => { connect(); }, [connect, tick]);

  useEffect(() => {
    const id = setInterval(() => setTick((n) => n + 1), TOKEN_REFRESH_MS);
    return () => clearInterval(id);
  }, []);

  const select = {
    background: '#0f172a', color: '#e2e8f0', border: '1px solid #334155',
    borderRadius: 6, padding: '4px 8px', fontSize: 12,
  };

  return (
    <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 12 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#e2e8f0' }}>Video Wall</div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>
            {cameras.length} cameras on one stream · rates on each tile are measured
          </div>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          <label style={{ fontSize: 12, color: '#94a3b8' }}>
            Target fps{' '}
            <select style={select} value={fps} onChange={(e) => setFps(Number(e.target.value))}>
              {[15, 20, 25].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label style={{ fontSize: 12, color: '#94a3b8' }}>
            Tile{' '}
            <select style={select} value={tileW} onChange={(e) => setTileW(Number(e.target.value))}>
              {[240, 320, 400].map((v) => <option key={v} value={v}>{v}px</option>)}
            </select>
          </label>
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14 }}>
        {LEGEND.map(([name, colour, meaning]) => (
          <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#94a3b8' }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: colour }} />
            <b style={{ color: '#cbd5e1' }}>{name}</b> {meaning}
          </div>
        ))}
        <div style={{ fontSize: 11, color: '#64748b' }}>
          src = decoded from camera · show = published to screen · ai = analysed
        </div>
      </div>

      {error && (
        <div style={{ color: '#fca5a5', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)',
                      borderRadius: 8, padding: '10px 12px', fontSize: 13 }}>
          {error}
        </div>
      )}

      {src && (
        <div style={{ overflowX: 'auto', background: '#020617', borderRadius: 8, border: '1px solid #1e293b' }}>
          <MjpegImg
            key={src}
            src={src}
            alt="Video wall of all authorised cameras"
            style={{ display: 'block', maxWidth: '100%', height: 'auto' }}
            onError={() => setTimeout(() => setTick((n) => n + 1), 3000)}
          />
        </div>
      )}
    </div>
  );
}
