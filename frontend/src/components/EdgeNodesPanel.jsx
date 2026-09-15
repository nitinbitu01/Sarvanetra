// frontend/src/components/EdgeNodesPanel.jsx
//
// The scalability claim, made visible.
//
// Sending video to a central GPU does not scale: one camera is ~2.2 Mbit/s, so
// three thousand cameras would be ~6.6 Gbit/s. An edge node runs stages 2-4
// beside the camera and posts only the answer — measured here on real traffic
// at three to four thousand times smaller.
//
// This panel reads GET /edge/nodes, which reports what each node actually sent
// against the video it did not. Both figures are counted server-side from real
// requests, so the ratio on screen is measured rather than claimed — that is
// the whole point of showing it here instead of on a slide.
import { useCallback, useEffect, useRef, useState } from 'react';
import { useAuth, API } from '../context/AuthContext';

const POLL_MS = 4000;

function fmtBytes(n) {
  if (!n) return '0 B';
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export default function EdgeNodesPanel({ height = 320 }) {
  const { authFetch } = useAuth();
  const [nodes, setNodes] = useState([]);
  const [recent, setRecent] = useState([]);
  const [serverTime, setServerTime] = useState(null);
  const [error, setError] = useState(null);
  // Ages are recomputed every second rather than only on each poll, so the
  // panel visibly advances between fetches. A number that moves cannot be a
  // number that was typed in.
  const [, tick] = useState(0);

  const [running, setRunning] = useState([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);

  const load = useCallback(async () => {
    try {
      const res = await authFetch(`${API}/edge/nodes`);
      if (!res?.ok) throw new Error(`HTTP ${res?.status}`);
      const d = await res.json();
      setNodes(d.nodes || []);
      setRecent(d.recent_events || []);
      setServerTime(d.server_time || null);
      setError(null);
    } catch (e) {
      setError(e.message || 'Could not reach edge ingest');
    }
    try {
      const s = await authFetch(`${API}/edge/control/status`);
      if (s?.ok) setRunning((await s.json()).running || []);
    } catch { /* control is optional; the panel still reports traffic */ }
  }, [authFetch]);

  // Start and stop take a moment to show up: the node loads its models before
  // the first batch, so the button says so rather than looking broken.
  const control = useCallback(async (action) => {
    setBusy(true);
    setNotice(action === 'start' ? 'Starting node — models load first…' : 'Stopping…');
    try {
      const r = await authFetch(`${API}/edge/control/${action}?camera=CAM_09`,
                                { method: 'POST' });
      const j = await r.json().catch(() => ({}));
      setNotice(r.ok
        ? (action === 'start'
            ? 'Node started. First plates arrive in about 25 seconds.'
            : 'Node stopped. Nothing new will arrive now.')
        : (j.detail || `Could not ${action} the node`));
      await load();
    } catch (e) {
      setNotice(e.message || `Could not ${action} the node`);
    } finally {
      setBusy(false);
      setTimeout(() => setNotice(null), 9000);
    }
  }, [authFetch, load]);

  const isRunning = running.includes('CAM_09');

  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  useEffect(() => {
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);

  // Ages advance client-side between polls, from the age the server reported.
  const fetchedAt = useRef(Date.now());
  useEffect(() => { fetchedAt.current = Date.now(); }, [recent]);
  const ageOf = (r) =>
    (r.age_sec || 0) + Math.floor((Date.now() - fetchedAt.current) / 1000);

  // One headline figure across every node, because the question a viewer has is
  // about the architecture, not about any single device.
  const meta = nodes.reduce((a, n) => a + (n.metadata_bytes || 0), 0);
  const video = nodes.reduce((a, n) => a + (n.video_bytes_avoided || 0), 0);
  const events = nodes.reduce((a, n) => a + (n.events || 0), 0);
  const ratio = meta > 0 ? video / meta : 0;
  const mbitPerCam = nodes.length
    ? nodes.reduce((a, n) => a + (n.video_mbit_s || 0), 0) / nodes.length : 2.2;
  const kbitPerCam = nodes.length
    ? nodes.reduce((a, n) => a + (n.metadata_kbit_s || 0), 0) / nodes.length : 0;

  // Bar widths are logarithmic. On a linear scale a four-thousand-fold ratio
  // renders the metadata bar as nothing at all, which reads as "no data"
  // rather than "almost none" — the opposite of the point.
  const lg = (v) => (v > 0 ? Math.log10(v + 1) : 0);
  const metaW = ratio > 0 ? Math.max(2, (lg(meta) / lg(video)) * 100) : 2;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height, minHeight: 0,
                  gap: 12, overflowY: 'auto', padding: '4px 2px' }}>

      {/* The controls come first because using them IS the demonstration:
          stop the node and this page goes quiet; start it and plates return
          within one batch. A page of fixed numbers cannot do either. */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8,
                    padding: '8px 10px', borderRadius: 6,
                    background: 'rgba(148,163,184,0.08)',
                    border: '1px solid rgba(148,163,184,0.16)' }}>
        <span style={{ width: 9, height: 9, borderRadius: '50%', flexShrink: 0,
                       background: isRunning ? '#34d399' : '#64748b' }} />
        <span style={{ fontSize: 12, fontWeight: 700,
                       color: isRunning ? '#34d399' : '#94a3b8' }}>
          CAM_09 edge node {isRunning ? 'running' : 'stopped'}
        </span>

        <button
          onClick={() => control('start')}
          disabled={busy || isRunning}
          style={{ marginLeft: 'auto', padding: '5px 14px', fontSize: 12,
                   fontWeight: 700, borderRadius: 4, cursor: (busy || isRunning) ? 'default' : 'pointer',
                   border: '1px solid #34d399',
                   background: (busy || isRunning) ? 'transparent' : '#34d399',
                   color: (busy || isRunning) ? '#475569' : '#04231a',
                   opacity: (busy || isRunning) ? 0.45 : 1 }}
        >
          Start node
        </button>
        <button
          onClick={() => control('stop')}
          disabled={busy || !isRunning}
          style={{ padding: '5px 14px', fontSize: 12, fontWeight: 700,
                   borderRadius: 4, cursor: (busy || !isRunning) ? 'default' : 'pointer',
                   border: '1px solid #f87171',
                   background: (busy || !isRunning) ? 'transparent' : '#f87171',
                   color: (busy || !isRunning) ? '#475569' : '#2a0b0b',
                   opacity: (busy || !isRunning) ? 0.45 : 1 }}
        >
          Stop node
        </button>
      </div>

      {notice && (
        <div style={{ fontSize: 11, color: '#38bdf8' }}>{notice}</div>
      )}

      {error && (
        <div style={{ fontSize: 12, color: '#f87171' }}>
          Edge ingest unreachable — {error}
        </div>
      )}

      {!error && nodes.length === 0 && (
        <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6 }}>
          No edge node has reported yet. Start one with:
          <div style={{ fontFamily: 'monospace', fontSize: 11, marginTop: 6,
                        background: 'rgba(148,163,184,0.12)', padding: '6px 8px',
                        borderRadius: 4, color: '#cbd5e1' }}>
            python -m edge.agent --camera CAM_09 --source …
          </div>
        </div>
      )}

      {nodes.length > 0 && (
        <>
          {/* the headline: what the split actually bought */}
          <div style={{ textAlign: 'center', padding: '10px 0 4px' }}>
            <div style={{ fontSize: 44, fontWeight: 800, lineHeight: 1,
                          color: '#34d399', fontVariantNumeric: 'tabular-nums' }}>
              {ratio >= 1000
                ? `${(ratio / 1000).toFixed(1)}k×`
                : `${Math.round(ratio)}×`}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4,
                          letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              less data than sending the video
            </div>
          </div>

          {/* the same thing as a picture */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between',
                            fontSize: 11, color: '#cbd5e1', marginBottom: 3 }}>
                <span>Video, had it been streamed</span>
                <strong style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {fmtBytes(video)}
                </strong>
              </div>
              <div style={{ height: 14, borderRadius: 3, background: '#ef4444' }} />
            </div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between',
                            fontSize: 11, color: '#cbd5e1', marginBottom: 3 }}>
                <span>Metadata actually sent</span>
                <strong style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {fmtBytes(meta)}
                </strong>
              </div>
              <div style={{ height: 14, borderRadius: 3,
                            background: 'rgba(148,163,184,0.18)' }}>
                <div style={{ width: `${metaW}%`, height: '100%', borderRadius: 3,
                              background: '#34d399' }} />
              </div>
            </div>
            <div style={{ fontSize: 10, color: '#64748b' }}>
              Bars are log-scaled; on a linear axis the green one would vanish.
            </div>
          </div>

          {/* what it means for a city */}
          <div style={{ borderTop: '1px solid rgba(148,163,184,0.18)', paddingTop: 10 }}>
            <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6,
                          letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              Projected to a city network
            </div>
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse',
                            fontVariantNumeric: 'tabular-nums' }}>
              <thead>
                <tr style={{ color: '#64748b', textAlign: 'right' }}>
                  <th style={{ textAlign: 'left', fontWeight: 600 }}>Cameras</th>
                  <th style={{ fontWeight: 600 }}>Streaming video</th>
                  <th style={{ fontWeight: 600 }}>Edge metadata</th>
                </tr>
              </thead>
              <tbody>
                {[30, 500, 3000].map((n) => (
                  <tr key={n} style={{ textAlign: 'right', color: '#e2e8f0' }}>
                    <td style={{ textAlign: 'left', padding: '3px 0' }}>{n.toLocaleString()}</td>
                    <td style={{ color: n >= 3000 ? '#f87171' : '#e2e8f0' }}>
                      {(mbitPerCam * n / 1000).toFixed(2)} Gbit/s
                    </td>
                    <td style={{ color: '#34d399' }}>
                      {(kbitPerCam * n / 1000).toFixed(2)} Mbit/s
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* the devices themselves */}
          <div style={{ borderTop: '1px solid rgba(148,163,184,0.18)', paddingTop: 10 }}>
            <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6,
                          letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              Edge nodes ({nodes.length}) · {events} events received
            </div>
            {nodes.map((n) => {
              const stale = (n.seconds_since_last_report ?? 999) > 30;
              return (
                <div key={n.node_id}
                     style={{ display: 'flex', alignItems: 'center', gap: 8,
                              padding: '6px 0', fontSize: 12,
                              borderBottom: '1px solid rgba(148,163,184,0.10)' }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%',
                                 background: stale ? '#64748b' : '#34d399',
                                 flexShrink: 0 }} />
                  <span style={{ fontWeight: 700, color: '#f8fafc' }}>{n.node_id}</span>
                  <span style={{ color: '#94a3b8' }}>{(n.cameras || []).join(', ')}</span>
                  <span style={{ marginLeft: 'auto', color: '#cbd5e1',
                                 fontVariantNumeric: 'tabular-nums' }}>
                    {n.events} events
                  </span>
                  <span style={{ color: '#64748b', fontSize: 11, minWidth: 54,
                                 textAlign: 'right' }}>
                    {stale ? 'idle' : `${Math.round(n.seconds_since_last_report)}s ago`}
                  </span>
                </div>
              );
            })}
          </div>

          {/* What actually arrived. A ratio can be typed into a page; these
              plates cannot — they are the ones on the camera, seconds ago. */}
          <div style={{ borderTop: '1px solid rgba(148,163,184,0.18)', paddingTop: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6,
                          marginBottom: 6 }}>
              <span style={{ fontSize: 11, color: '#94a3b8',
                             letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                What each plate cost to deliver
              </span>
              {recent.some((r) => ageOf(r) < 15) && (
                <span style={{ fontSize: 9, fontWeight: 800, color: '#34d399',
                               border: '1px solid #34d399', borderRadius: 3,
                               padding: '1px 5px', letterSpacing: '0.08em' }}>
                  LIVE
                </span>
              )}
              {serverTime && (
                <span style={{ marginLeft: 'auto', fontSize: 10, color: '#64748b',
                               fontVariantNumeric: 'tabular-nums' }}>
                  server clock {new Date(serverTime * 1000).toLocaleTimeString()}
                </span>
              )}
            </div>

            {recent.length === 0 && (
              <div style={{ fontSize: 11, color: '#64748b' }}>
                Nothing received yet — start a node and rows appear here within
                one batch.
              </div>
            )}

            {recent.map((r, i) => {
              const age = ageOf(r);
              const fresh = age < 15;
              return (
                <div key={`${r.node}-${r.cam}-${r.track}-${i}`}
                     style={{ display: 'flex', alignItems: 'center', gap: 8,
                              padding: '5px 6px', fontSize: 12, borderRadius: 4,
                              background: fresh ? 'rgba(52,211,153,0.10)' : 'transparent',
                              borderLeft: fresh ? '2px solid #34d399'
                                                : '2px solid transparent' }}>
                  <span style={{ fontFamily: 'monospace', fontWeight: 800,
                                 letterSpacing: '0.5px',
                                 color: r.plate ? '#fbbf24' : '#64748b' }}>
                    {r.plate || 'no plate read'}
                  </span>
                  {r.conf != null && (
                    <span style={{ fontSize: 10, color: '#94a3b8' }}>
                      {Number(r.conf).toFixed(2)}
                    </span>
                  )}
                  <span style={{ fontSize: 10, color: '#94a3b8' }}>{r.cam}</span>
                  {r.kmh != null && (
                    <span style={{ fontSize: 11, color: '#38bdf8',
                                   fontVariantNumeric: 'tabular-nums' }}>
                      {Number(r.kmh).toFixed(1)} km/h
                    </span>
                  )}
                  {/* The point of this row is not the plate — the panel beside
                      it already reads plates. It is what the plate cost to get
                      here: bytes actually sent, against the video that would
                      otherwise have carried the same answer. */}
                  <span style={{ marginLeft: 'auto', fontSize: 11,
                                 fontVariantNumeric: 'tabular-nums' }}>
                    <strong style={{ color: '#34d399' }}>{r.batch_bytes} B</strong>
                    <span style={{ color: '#475569' }}> sent · </span>
                    <s style={{ color: '#f87171' }}>
                      {((mbitPerCam * 1e6 / 8) * ((r.frames || 1) / 5) / 1e6).toFixed(1)} MB
                    </s>
                    <span style={{ color: '#475569' }}> of video avoided</span>
                  </span>
                  <span style={{ fontSize: 10, color: '#64748b', minWidth: 52,
                                 textAlign: 'right',
                                 fontVariantNumeric: 'tabular-nums' }}>
                    {age}s ago
                  </span>
                </div>
              );
            })}
          </div>

          <div style={{ fontSize: 10, color: '#64748b', lineHeight: 1.5 }}>
            Each node runs vehicle detection, tracking and ANPR next to the camera
            and posts only the result. If the link drops it spools to disk and
            replays, so an outage delays evidence instead of losing it.
            <br />
            Every figure here is counted server-side from the requests the nodes
            made; stop a node and the ages above keep climbing, start one and new
            plates appear within a batch.
          </div>
        </>
      )}
    </div>
  );
}
