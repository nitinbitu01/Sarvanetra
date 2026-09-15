// frontend/src/components/LiveANPRPanel.jsx
//
// Live ANPR reads — the plates the 24x7 pipeline has actually finished
// reading, most recent first.
//
// WHY THIS POLLS A REST ENDPOINT, NOT A WEBSOCKET
//   The ANPR-capable pipeline (Live24x7Pipeline) is meant to run as its own
//   process, separate from the API server — sharing a process measurably
//   starves request handling (see run_pipeline.py's own docstring: 30s
//   timeouts, 6.7GB held). A live push from that process would need either
//   running it in-process anyway, or a cross-process broker this environment
//   doesn't have running. The database is the channel both processes already
//   share — vehicle_track.plate_text is written the moment a track completes
//   — so polling GET /analytics/live/plate-reads is simpler and more robust
//   than a push would be here, not a fallback for lacking one.
//
// WHAT A "READ" MEANS HERE
//   A row exists once a track ends, so this lags the live frame by however
//   long that vehicle was in view. That is the FINAL voted read
//   (TemporalPlateVoter), not a single noisy frame guess — a more meaningful
//   number than the per-frame overlay badge on its own.
import { useCallback, useEffect, useRef, useState } from 'react';
import { useAuth, API } from '../context/AuthContext';
import '../styles/live-anpr.css';

const POLL_MS = 5000;

function timeAgo(iso) {
  if (!iso) return '';
  const s = Math.max(0, Math.floor((Date.now() - new Date(iso + 'Z').getTime()) / 1000));
  if (s < 5) return 'just now';
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.floor(m / 60)}h ago`;
}

// The time burnt into the footage the vehicle was seen in. Shown as a clock
// reading, never as an age: on a live stream the two agree, and on replayed
// footage the clock stays honest where an age would claim the recording is
// hours stale when the read is seconds old.
function footageClock(iso) {
  if (!iso) return '';
  const d = new Date(iso + 'Z');
  if (Number.isNaN(d.getTime())) return '';
  const sameDay = d.toDateString() === new Date().toDateString();
  const t = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return sameDay ? t : `${d.toLocaleDateString([], { day: 'numeric', month: 'short' })} ${t}`;
}

export default function LiveANPRPanel({ cameraId = null, limit = 20, height = 260 }) {
  const { authFetch } = useAuth();
  const [reads, setReads] = useState([]);
  const [pipelineRunning, setPipelineRunning] = useState(null);
  const [note, setNote] = useState(null);
  const [error, setError] = useState(null);
  const [, forceTick] = useState(0); // re-render every few seconds so "Xs ago" advances

  const load = useCallback(async () => {
    try {
      const qs = new URLSearchParams({ limit: String(limit) });
      if (cameraId) qs.set('camera_id', cameraId);
      const res = await authFetch(`${API}/analytics/live/plate-reads?${qs}`);
      if (!res?.ok) throw new Error(`HTTP ${res?.status}`);
      const data = await res.json();
      setReads(data.reads || []);
      setPipelineRunning(!!data.pipeline_running);
      setNote(data.note || null);
      setError(null);
    } catch (e) {
      setError(e.message || 'Could not load live ANPR reads');
    }
  }, [authFetch, cameraId, limit]);

  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  useEffect(() => {
    const id = setInterval(() => forceTick((n) => n + 1), 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height, minHeight: 0 }}>
      <div className="anpr-live-status">
        <span className={`anpr-live-dot anpr-live-dot--${pipelineRunning ? 'running' : 'stopped'}`} />
        <span className="anpr-live-status-text">
          {pipelineRunning === null
            ? 'Checking pipeline…'
            : pipelineRunning
              ? <><strong>Live</strong> — 24x7 pipeline running</>
              : <><strong>Not running</strong> — showing last completed reads</>}
        </span>
      </div>

      {note && !pipelineRunning && (
        <div className="anpr-live-empty" style={{ textAlign: 'left', fontSize: 11 }}>
          {note}
        </div>
      )}

      <div className="anpr-live-list">
        {error && <div className="anpr-live-empty">{error}</div>}
        {!error && reads.length === 0 && (
          <div className="anpr-live-empty">
            No plate reads yet{cameraId ? ' for this camera' : ''}.
          </div>
        )}
        {!error && reads.map((r, i) => (
          <div className="anpr-live-row" key={`${r.camera_id}-${r.last_seen}-${i}`}>
            <span className="anpr-plate">{r.plate}</span>
            <div className="anpr-live-meta">
              <span className="anpr-live-cam">{r.camera_name}</span>
              <span className="anpr-live-sub">
                {/* The timestamp is the FOOTAGE clock — when the vehicle
                    passed the camera — not when this row arrived. Replaying a
                    recording, every fresh read carries yesterday's time, and
                    rendering that as "23h ago" under a heading that says
                    REAL-TIME reads the wrong way round. Name it for what it
                    is instead. */}
                {r.vehicle_class || 'vehicle'} · {footageClock(r.footage_time || r.last_seen)}
                {r.confidence != null && ` · ${Math.round(r.confidence * 100)}%`}
              </span>
            </div>
            <span className={`anpr-grade anpr-grade--${r.grade}`}>
              {r.grade === 'committed' ? 'Committed' : 'Unconfirmed'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
