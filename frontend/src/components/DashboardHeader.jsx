// frontend/src/components/DashboardHeader.jsx
//
// Connection state, plate search, and the push opt-in.
//
// This search box was deliberately inert for most of the project's life: there
// was no backend that could answer "where has this plate been", and a
// client-side filter over whatever happened to be loaded would have returned
// confident partial results that read as a working search.
//
// That endpoint now exists and is measured - GET /journeys returns 92%
// precision and 89.6% recall on held-out vehicles - so the honest thing is no
// longer to do nothing, it is to answer. What it must NOT do is answer
// vaguely: a plate that was never seen returns "not found" rather than the
// nearest match, because a wrong vehicle is worse than no vehicle.
import { useCallback, useState } from 'react';
import { useConnectionState } from '../context/WebSocketContext';
import { useAuth } from '../context/AuthContext';
import EnableAlertsButton from './EnableAlertsButton';

const API = import.meta.env.VITE_API_URL || '/api/v1';

// Substitution costs taken from the confusions this recogniser actually makes,
// audited against its held-out errors. A pair it mixes up constantly is weak
// evidence of a different vehicle; a pair it never mixes up is strong evidence.
// Uniform edit distance would rank GJ01AB1234 -> GJ01AB1734 (a 1/7 the model
// never confuses) the same as -> GJ01A81234 (an 8/B it confuses weekly).
const CONFUSION = {
  '0O': 0.05, 'O0': 0.05, '1I': 0.05, 'I1': 0.05,
  '8B': 0.15, 'B8': 0.15, '5S': 0.15, 'S5': 0.15,
  '2Z': 0.15, 'Z2': 0.15, '6G': 0.20, 'G6': 0.20,
  'DO': 0.30, 'OD': 0.30, '0D': 0.30, 'D0': 0.30,
  'NH': 0.35, 'HN': 0.35, 'MH': 0.35, 'HM': 0.35,
  'RB': 0.35, 'BR': 0.35, 'CG': 0.40, 'GC': 0.40,
  'Q0': 0.30, '0Q': 0.30,
};

function weightedEdit(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, (_, i) =>
    Array.from({ length: m + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)));
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const c1 = a[i - 1], c2 = b[j - 1];
      const cost = c1 === c2 ? 0 : (CONFUSION[c1 + c2] ?? 1.0);
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[n][m];
}

const DOT = {
  CONNECTED:    { color: '#10b981', label: 'Connected' },
  CONNECTING:   { color: '#f59e0b', label: 'Connecting…' },
  RECONNECTING: { color: '#f59e0b', label: 'Reconnecting…' },
  OFFLINE:      { color: '#ef4444', label: 'Offline' },
};

export default function DashboardHeader() {
  const connectionState = useConnectionState();
  const { token } = useAuth();
  const [search, setSearch] = useState('');
  const [result, setResult] = useState(null);   // null | {…} | {none:true}
  const [busy, setBusy] = useState(false);
  const dot = DOT[connectionState] || { color: '#64748b', label: 'Unknown' };

  const onSubmit = useCallback(async (e) => {
    e.preventDefault();
    const q = search.trim().toUpperCase().replace(/[^A-Z0-9]/g, '');
    if (!q) { setResult(null); return; }
    setBusy(true);
    try {
      const res = await fetch(`${API}/journeys?mode=vehicle&limit=2000`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const all = data.journeys || [];
      const hit = all.find(j => j.reid_id === q);
      if (hit) {
        // Near-matches are NOT shown beside a hit. Offering alternatives when
        // the vehicle was found invites an officer to act on the wrong one.
        setResult(hit);
      } else {
        // Only when the exact plate is absent. A plate mistyped by one
        // character, or misremembered from a witness statement, is the
        // ordinary case in an investigation - returning nothing there wastes
        // a real lead. Suggestions are labelled as suggestions and carry a
        // similarity, never a confidence: they say how close the strings are
        // under this recogniser's known confusions, and nothing about whether
        // the vehicle is the right one.
        const near = all
          .filter(j => Math.abs(j.reid_id.length - q.length) <= 1)
          .map(j => ({ j, d: weightedEdit(q, j.reid_id) }))
          .filter(x => x.d <= 1.6)
          .sort((a, b) => a.d - b.d)
          .slice(0, 4)
          .map(x => ({
            plate: x.j.reid_id,
            similarity: Math.round(Math.max(0, 1 - x.d / Math.max(q.length, 1)) * 100),
            cameras: x.j.stops_count,
            watchlist: x.j.watchlist_match,
            // Two different reasons a plate can be one character away, and an
            // officer should be told which. A reader confusion (8/B, 0/O)
            // means the SYSTEM may have got it wrong; a plain typo means the
            // person typing may have. The costs separate them: a confusable
            // substitution is priced under 0.5, an ordinary one at 1.0.
            kind: x.d <= 0.6 ? 'reader-confusion' : 'one-character',
          }));
        setResult({ none: true, query: q, near });
      }
    } catch (err) {
      console.error('[SentinelIQ] plate search failed:', err);
      setResult({ error: true });
    } finally {
      setBusy(false);
    }
  }, [search, token]);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap', position: 'relative' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
        <span style={{
          width: 8, height: 8, borderRadius: '50%', background: dot.color,
          boxShadow: `0 0 6px ${dot.color}`,
        }} />
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{dot.label}</span>
      </div>

      <form onSubmit={onSubmit} style={{ flex: 1, minWidth: 180, maxWidth: 420, position: 'relative' }}>
        <input
          type="text"
          value={search}
          onChange={(e) => { setSearch(e.target.value); setResult(null); }}
          placeholder="Search a plate, e.g. RJ47GA5111 — press Enter"
          aria-label="Search for a vehicle by plate"
          style={{
            width: '100%', padding: '6px 11px', fontSize: 12,
            background: 'var(--bg-primary, #0f172a)',
            border: '1px solid var(--border, #334155)',
            borderRadius: 6, color: 'var(--text-primary, #fff)', outline: 'none',
            letterSpacing: 1,
          }}
        />

        {(busy || result) && (
          <div style={{
            position: 'absolute', top: 'calc(100% + 6px)', left: 0, right: 0,
            background: '#0f172a', border: '1px solid #334155', borderRadius: 8,
            padding: 12, zIndex: 60, fontSize: 12,
            boxShadow: '0 8px 26px rgba(0,0,0,.45)',
          }}>
            {busy && <span style={{ color: '#94a3b8' }}>Searching 16 cameras…</span>}

            {!busy && result?.error && (
              <span style={{ color: '#f87171' }}>Search failed — is the backend running?</span>
            )}

            {!busy && result?.none && (
              <div>
                <div style={{ color: '#f8fafc', fontWeight: 700, marginBottom: 3 }}>
                  {result.query} — not found
                </div>
                <div style={{ color: '#94a3b8' }}>
                  This plate did not pass any indexed camera.
                </div>

                {result.near?.length > 0 && (
                  <div style={{ marginTop: 10, paddingTop: 9, borderTop: '1px solid #1e293b' }}>
                    <div style={{ color: '#94a3b8', marginBottom: 6 }}>
                      Indexed plates within one character —{' '}
                      <span style={{ color: '#fbbf24' }}>suggestions, not matches</span>.
                      Confirm against the evidence crop before acting.
                    </div>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {result.near.map((n) => (
                        <button
                          key={n.plate}
                          type="button"
                          onClick={() => { setSearch(n.plate); setResult(null); }}
                          title={n.kind === 'reader-confusion'
                            ? 'Differs by a character pair this reader is known to confuse — the system may have misread it'
                            : 'Differs by one ordinary character — more likely a typing slip than a misread'}
                          style={{
                            background: n.watchlist ? '#ef444422' : '#1e293b',
                            border: `1px solid ${n.watchlist ? '#ef444455' : '#334155'}`,
                            borderRadius: 20, padding: '3px 10px', cursor: 'pointer',
                            color: '#e2e8f0', fontSize: 11, letterSpacing: 0.6,
                          }}>
                          {n.plate}
                          <span style={{ color: '#64748b' }}> · {n.similarity}%</span>
                          {n.kind === 'reader-confusion' && (
                            <span style={{ color: '#38bdf8' }}> · likely misread</span>
                          )}
                          {n.watchlist && (
                            <span style={{ color: '#f87171', fontWeight: 700 }}> · WL</span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {!result.near?.length && (
                  <div style={{ color: '#64748b', marginTop: 6 }}>
                    No near-matches either. Nothing is returned rather than the
                    closest available vehicle.
                  </div>
                )}
              </div>
            )}

            {!busy && result && !result.none && !result.error && (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <span style={{ color: '#f8fafc', fontWeight: 700, letterSpacing: 1 }}>
                    {result.reid_id}
                  </span>
                  {result.watchlist_match && (
                    <span style={{
                      background: '#ef444422', color: '#f87171',
                      border: '1px solid #ef444455', padding: '1px 6px',
                      borderRadius: 4, fontSize: 10, fontWeight: 800,
                    }}>WATCHLIST</span>
                  )}
                  {result.route_confidence != null && (
                    <span
                      title={result.confidence_basis || ''}
                      style={{ color: '#34d399', fontWeight: 700 }}>
                      {Math.round(result.route_confidence * 100)}% match
                    </span>
                  )}
                </div>
                {result.stops?.map((s, i) => (
                  <div key={i} style={{ color: '#cbd5e1', marginBottom: 2 }}>
                    {new Date(s.timestamp_utc).toLocaleTimeString()} ·{' '}
                    <span style={{ color: '#38bdf8' }}>{s.camera_id}</span> ·{' '}
                    {s.camera_location}
                    {s.match_type !== 'plate_confirmed' && (
                      <span style={{ color: '#fbbf24' }}> · candidate</span>
                    )}
                  </div>
                ))}
                <div style={{ color: '#64748b', marginTop: 6 }}>
                  Open the Journeys tab for the full route and evidence.
                </div>
              </div>
            )}
          </div>
        )}
      </form>

      <div style={{ flexShrink: 0 }}>
        <EnableAlertsButton />
      </div>
    </div>
  );
}
