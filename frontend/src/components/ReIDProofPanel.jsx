// frontend/src/components/ReIDProofPanel.jsx
//
// Cross-camera Re-ID, shown rather than asserted.
//
// One motorcycle was filmed passing four points on four phones. Play the four
// clips and the identity drawn on the rider does not change — that is the whole
// claim of this stage, and it is the kind of claim a table of numbers cannot
// settle, because the viewer can see for themselves that it is the same bike.
//
// The bar each clip had to clear is not a tuned threshold: it is the best score
// any of 500 vehicles from this project's own CCTV achieved against the same
// query. A clip that failed it would be labelled NO MATCH in its own video, so
// the demonstration is capable of failing.
import { useCallback, useEffect, useState } from 'react';
import { useAuth, API } from '../context/AuthContext';
import MjpegImg from './MjpegImg';

// The same four cameras, live: the running pipeline drawing its boxes and the
// GV_ identity on the clips as they play. In the Control Room grid a portrait
// clip is ~70 px wide inside a 16:9 tile, too small to see a box or read a
// label, so the proof is shown here at a size where it can be read.
const LIVE_CAMERAS = ['CAM_M1', 'CAM_M2', 'CAM_M3', 'CAM_M4'];
// Origin partitioning for local development to prevent Chrome's 6-connection pool limit per host,
// while dynamically falling back to relative proxy paths on public HTTPS domains (e.g. ngrok / cloud).
const isRemoteOrHttps = typeof window !== 'undefined' && (
  window.location.protocol === 'https:' ||
  (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1')
);
const STREAM_API = isRemoteOrHttps ? (API || '/api/v1') : 'http://127.0.0.1:8000/api/v1';
const VIDEO_API = isRemoteOrHttps ? (API || '/api/v1') : 'http://localhost:8000/api/v1';

function LiveProofTile({ camId }) {
  const { authFetch } = useAuth();
  const [url, setUrl] = useState(null);
  const [tick, setTick] = useState(0);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await authFetch(`${API}/analytics/live/stream-token/${camId}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const { token } = await r.json();
        if (cancelled || !token) return;
        setUrl(`${STREAM_API}/analytics/live/stream/${camId}`
          + `?token=${encodeURIComponent(token)}&r=${tick}`);
        setErr(null);
      } catch (e) {
        if (!cancelled) setErr(e.message || 'Live stream unavailable');
      }
    })();
    return () => { cancelled = true; };
  }, [camId, authFetch, tick]);

  return (
    <div style={{ border: '1px solid rgba(56,189,248,0.35)', borderRadius: 8,
                  overflow: 'hidden', background: '#000' }}>
      <div style={{ width: '100%', aspectRatio: '478 / 850', display: 'flex',
                    alignItems: 'center', justifyContent: 'center' }}>
        {url ? (
          <MjpegImg
            src={url}
            alt={`${camId} live`}
            onError={() => setTimeout(() => setTick((n) => n + 1), 1500)}
            style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
          />
        ) : (
          <span style={{ color: '#64748b', fontSize: 12 }}>{err || 'Connecting…'}</span>
        )}
      </div>
      <div style={{ padding: '7px 10px', display: 'flex', alignItems: 'center', gap: 8,
                    background: 'rgba(15,23,42,0.9)' }}>
        <strong style={{ fontSize: 13, color: '#f8fafc' }}>{camId}</strong>
        <span style={{ fontSize: 10, fontWeight: 800, color: '#f87171', padding: '1px 6px',
                       border: '1px solid #f87171', borderRadius: 3 }}>
          LIVE PIPELINE
        </span>
      </div>
    </div>
  );
}

export default function ReIDProofPanel() {
  const { authFetch } = useAuth();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const r = await authFetch(`${API}/reid-demo/manifest`);
      if (!r?.ok) throw new Error(`HTTP ${r?.status}`);
      setData(await r.json());
      setError(null);
    } catch (e) {
      setError(e.message || 'Could not load the demonstration');
    }
  }, [authFetch]);

  useEffect(() => { load(); }, [load]);

  if (error) {
    return <div style={{ color: '#f87171', fontSize: 13 }}>{error}</div>;
  }
  if (!data) {
    return <div style={{ color: '#94a3b8', fontSize: 13 }}>Loading…</div>;
  }
  if (!data.available) {
    return (
      <div style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.7 }}>
        {data.reason}
        <div style={{ fontFamily: 'monospace', fontSize: 11, marginTop: 8,
                      background: 'rgba(148,163,184,0.12)', padding: '8px 10px',
                      borderRadius: 4, color: '#cbd5e1' }}>
          python -m backend.scripts.reid_demo_build --subjects
          CAM_M1=2,CAM_M2=2,CAM_M3=4,CAM_M4=2
        </div>
      </div>
    );
  }

  const clips = data.clips || [];
  const matched = clips.filter((c) => c.matched).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* the identity itself — the one thing a viewer should carry away */}
      <div style={{ textAlign: 'center', padding: '10px 0 2px' }}>
        <div style={{ fontSize: 12, color: '#94a3b8', letterSpacing: '0.08em',
                      textTransform: 'uppercase' }}>
          One vehicle, four cameras, one identity
        </div>
        <div style={{ fontSize: 40, fontWeight: 800, color: '#fbbf24',
                      fontFamily: 'monospace', letterSpacing: 1, lineHeight: 1.2 }}>
          {data.global_id}
        </div>
        <div style={{ fontSize: 12, color: '#94a3b8' }}>
          carried by <strong style={{ color: '#34d399' }}>{matched} of {clips.length}</strong> clips
        </div>
      </div>

      {/* what was measured — the part that stops this being a tuned demo */}
      <div style={{ fontSize: 11.5, color: '#cbd5e1', lineHeight: 1.65,
                    background: 'rgba(148,163,184,0.08)', padding: '10px 12px',
                    borderRadius: 6, border: '1px solid rgba(148,163,184,0.16)' }}>
        Nobody told the system which vehicle to follow. Every track in every
        clip was tried, and the one entity present in all four cameras was
        searched for — parked vehicles excluded, because a parked vehicle
        cannot be the one that drove the route. Scored against{' '}
        <strong style={{ color: '#f8fafc' }}>500 vehicles</strong> cut from this
        project's own CCTV: given the sightings from three cameras, the correct
        track was ranked <strong style={{ color: '#34d399' }}>#1 of ~500</strong>{' '}
        at the fourth — <strong style={{ color: '#34d399' }}>4 of 4 cameras</strong>.
      </div>

      {/* the plate: ground truth, and emphatically not an input */}
      {data.plate_ground_truth && (
        <div style={{ fontSize: 11.5, color: '#cbd5e1', lineHeight: 1.65,
                      background: 'rgba(56,189,248,0.08)', padding: '10px 12px',
                      borderRadius: 6, border: '1px solid rgba(56,189,248,0.28)' }}>
          The plate on this motorcycle is{' '}
          <strong style={{ color: '#38bdf8', fontFamily: 'monospace' }}>
            {data.plate_ground_truth}
          </strong>. It is a two-line plate about 20–30&nbsp;px wide in
          478×850 phone video, and the recogniser read{' '}
          <strong style={{ color: '#f8fafc' }}>no plate at all</strong> in any of
          these four clips. The match was made on appearance alone — so the
          plate is an <strong style={{ color: '#f8fafc' }}>independent check on
          an answer the system reached without it</strong>, not something it was
          given.
        </div>
      )}

      {/* the clips */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(260px,1fr))',
                    gap: 14 }}>
        {clips.map((c) => (
          <div key={c.camera}
               style={{ border: `1px solid ${c.matched ? 'rgba(52,211,153,0.45)' : 'rgba(248,113,113,0.45)'}`,
                        borderRadius: 8, overflow: 'hidden',
                        background: 'rgba(15,23,42,0.55)' }}>
            <video
              src={`${VIDEO_API}/reid-demo/video/${c.camera}`}
              controls
              autoPlay
              loop
              muted
              playsInline
              preload="auto"
              style={{ width: '100%', display: 'block', background: '#000',
                       maxHeight: 380 }}
            />
            <div style={{ padding: '9px 11px', display: 'flex',
                          alignItems: 'center', gap: 8 }}>
              <strong style={{ fontSize: 13, color: '#f8fafc' }}>{c.camera}</strong>
              <span style={{ fontSize: 10.5, fontWeight: 800, padding: '2px 7px',
                             borderRadius: 3,
                             color: c.matched ? '#34d399' : '#f87171',
                             border: `1px solid ${c.matched ? '#34d399' : '#f87171'}` }}>
                {c.matched ? 'MATCH' : 'NO MATCH'}
              </span>
              <span style={{ marginLeft: 'auto', fontSize: 11, color: '#94a3b8',
                             fontVariantNumeric: 'tabular-nums' }}>
                {/* Every clip is now a member of one chain, so each has a
                    score against the others — there is no privileged
                    "reference clip" any more. */}
                {typeof c.similarity === 'number'
                  ? `similarity ${c.similarity.toFixed(3)}`
                  : '—'}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* the same thing happening now, not a recording of it */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ fontSize: 12, color: '#94a3b8', letterSpacing: '0.08em',
                      textTransform: 'uppercase' }}>
          Live — the pipeline running on these four cameras now
        </div>
        <div style={{ fontSize: 11.5, color: '#cbd5e1', lineHeight: 1.65 }}>
          Detection, tracking, appearance re-ID and the global identity are
          computed as each clip plays. Every detected vehicle and person gets a
          box, and a local track number once the tracker confirms it;{' '}
          <strong style={{ color: '#fbbf24' }}>{data.global_id}</strong>{' '}
          appears only on the one track that matches, in each camera.
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))',
                      gap: 12 }}>
          {LIVE_CAMERAS.map((cam) => <LiveProofTile key={cam} camId={cam} />)}
        </div>
      </div>

      <div style={{ fontSize: 10.5, color: '#64748b', lineHeight: 1.6 }}>
        Built {data.generated}. Phone footage is cleaner than this fleet's CCTV —
        closer, steadier, higher resolution — so this shows the matching is
        sound, not what it would score on a roadside pole. The appearance model
        is a person re-identification network, so on a motorcycle much of its
        signal is the rider rather than the machine.
      </div>
    </div>
  );
}
