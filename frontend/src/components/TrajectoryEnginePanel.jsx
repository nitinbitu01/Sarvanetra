import React, { useCallback, useEffect, useState } from 'react';
import { useAuth, API } from '../context/AuthContext';

/**
 * TrajectoryEnginePanel — what the trajectory engine can do RIGHT NOW.
 *
 * Every figure on this screen is read from /journeys/pipeline-status at the
 * moment it is opened, against the live database. Nothing is written into the
 * page. A stage with insufficient data reports that instead of a number, which
 * is the point: a panel that always shows six green ticks proves nothing,
 * because it would show six green ticks on an empty database too.
 */

const STATE_STYLE = {
  live: { bg: 'rgba(52,211,153,0.12)', border: 'rgba(52,211,153,0.45)', fg: '#34d399', label: 'LIVE' },
  default: { bg: 'rgba(251,191,36,0.12)', border: 'rgba(251,191,36,0.45)', fg: '#fbbf24', label: 'COLLECTING' },
};

function stateStyle(s) {
  return s === 'live' ? STATE_STYLE.live : STATE_STYLE.default;
}

function Evidence({ data }) {
  if (!data) return null;
  const rows = Object.entries(data).filter(
    ([, v]) => v !== null && v !== undefined && typeof v !== 'object',
  );
  if (!rows.length) return null;
  return (
    <div style={{
      marginTop: 8, display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit,minmax(190px,1fr))', gap: '3px 14px',
      fontSize: 11, fontVariantNumeric: 'tabular-nums',
    }}>
      {rows.map(([k, v]) => (
        <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
          <span style={{ color: '#64748b' }}>{k.replace(/_/g, ' ')}</span>
          <span style={{ color: '#cbd5e1', fontWeight: 600, textAlign: 'right' }}>
            {typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(4)) : String(v)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function TrajectoryEnginePanel() {
  const { authFetch } = useAuth();
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);

  // authFetch rather than bare fetch: every other panel uses it, and it keeps
  // working if this endpoint later takes an auth dependency.
  const load = useCallback(async () => {
    setBusy(true);
    try {
      const r = await authFetch(`${API}/journeys/pipeline-status`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setData(await r.json());
      setErr(null);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }, [authFetch]);

  useEffect(() => { load(); }, [load]);

  if (err) {
    return (
      <div style={{ padding: 16, color: '#f87171', fontSize: 13 }}>
        Could not read the engine status: {err}
      </div>
    );
  }
  if (!data) {
    return <div style={{ padding: 16, color: '#94a3b8', fontSize: 13 }}>Reading live state…</div>;
  }

  const live = data.stages.filter((s) => s.state === 'live').length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, padding: '4px 2px 24px' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 13, color: '#cbd5e1' }}>
          <strong style={{ color: '#34d399', fontSize: 18 }}>{live}</strong>
          <span style={{ color: '#64748b' }}> of {data.stages.length} stages live</span>
        </div>
        <div style={{ marginLeft: 'auto', fontSize: 11, color: '#64748b' }}>
          measured {new Date(`${data.generated_utc}Z`).toLocaleTimeString()}
          <button
            onClick={load}
            disabled={busy}
            style={{
              marginLeft: 10, padding: '3px 10px', fontSize: 11, borderRadius: 4,
              border: '1px solid rgba(148,163,184,0.35)', background: 'transparent',
              color: '#cbd5e1', cursor: busy ? 'wait' : 'pointer',
            }}
          >
            {busy ? '…' : 'Re-measure'}
          </button>
        </div>
      </div>

      {data.stages.map((s) => {
        const st = stateStyle(s.state);
        return (
          <div key={s.step} style={{
            border: `1px solid ${st.border}`, borderRadius: 8,
            background: 'rgba(15,23,42,0.5)', padding: '11px 13px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 9, flexWrap: 'wrap' }}>
              <span style={{
                fontFamily: 'monospace', fontSize: 11, color: '#64748b',
                border: '1px solid rgba(148,163,184,0.3)', borderRadius: 3,
                padding: '1px 6px',
              }}>
                STEP {s.step}
              </span>
              <strong style={{ fontSize: 13.5, color: '#f1f5f9' }}>{s.name}</strong>
              <span style={{
                marginLeft: 'auto', fontSize: 10, fontWeight: 800, letterSpacing: '0.06em',
                padding: '2px 8px', borderRadius: 3, background: st.bg,
                color: st.fg, border: `1px solid ${st.border}`,
              }}>
                {s.state === 'live' ? 'LIVE' : s.state.toUpperCase()}
              </span>
            </div>
            <div style={{ marginTop: 6, fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
              {s.detail}
            </div>
            <Evidence data={s.evidence} />
          </div>
        );
      })}

      {data.worked_example && (
        <div style={{
          border: '1px solid rgba(56,189,248,0.35)', borderRadius: 8,
          background: 'rgba(8,47,73,0.35)', padding: '12px 14px',
        }}>
          <div style={{ fontSize: 13.5, fontWeight: 700, color: '#f1f5f9' }}>
            Worked example — check the arithmetic yourself
          </div>
          <div style={{ fontSize: 11.5, color: '#94a3b8', marginTop: 4, lineHeight: 1.6 }}>
            One real vehicle from the database,{' '}
            <strong style={{ color: '#38bdf8', fontFamily: 'monospace' }}>
              {data.worked_example.reid_id}
            </strong>
            , recomputed a moment ago. Every input is shown next to the answer,
            so nothing has to be taken on trust.
          </div>

          <div style={{ overflowX: 'auto', marginTop: 10 }}>
            <table style={{
              width: '100%', borderCollapse: 'collapse', fontSize: 11.5,
              fontVariantNumeric: 'tabular-nums', minWidth: 720,
            }}>
              <thead>
                <tr style={{ color: '#64748b', textAlign: 'left' }}>
                  <th style={{ padding: '5px 8px' }}>Leg</th>
                  <th style={{ padding: '5px 8px' }}>Straight</th>
                  <th style={{ padding: '5px 8px' }}>By road</th>
                  <th style={{ padding: '5px 8px' }}>Time</th>
                  <th style={{ padding: '5px 8px' }}>Speed = distance ÷ time</th>
                  <th style={{ padding: '5px 8px' }}>Limit</th>
                  <th style={{ padding: '5px 8px' }}>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {data.worked_example.legs.map((l, i) => (
                  <tr key={i} style={{ borderTop: '1px solid rgba(148,163,184,0.15)' }}>
                    <td style={{ padding: '6px 8px', color: '#e2e8f0' }}>
                      {l.from_camera} → {l.to_camera}
                      {l.road_matched && l.detour_factor && (
                        <div style={{ fontSize: 10, color: '#64748b' }}>
                          road is {l.detour_factor}× the straight line
                        </div>
                      )}
                    </td>
                    <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{l.straight_km} km</td>
                    <td style={{ padding: '6px 8px', color: l.road_matched ? '#38bdf8' : '#64748b' }}>
                      {l.road_km != null ? `${l.road_km} km` : 'no route'}
                    </td>
                    <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{l.minutes} min</td>
                    <td style={{ padding: '6px 8px', color: '#e2e8f0', fontFamily: 'monospace' }}>
                      {l.arithmetic}
                    </td>
                    <td style={{ padding: '6px 8px', color: '#94a3b8' }}>
                      {l.speed_limit_kmh != null ? `${l.speed_limit_kmh} km/h` : '—'}
                    </td>
                    <td style={{ padding: '6px 8px' }}>
                      <span style={{
                        fontSize: 10, fontWeight: 800, padding: '2px 7px',
                        borderRadius: 3,
                        color: l.flag === 'OVERSPEED' ? '#f87171'
                          : l.flag === 'NORMAL' ? '#34d399' : '#94a3b8',
                        border: `1px solid ${l.flag === 'OVERSPEED' ? '#f87171'
                          : l.flag === 'NORMAL' ? '#34d399' : '#475569'}`,
                      }}>
                        {l.flag}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {data.worked_example.legs.some((l) => l.needs_verification) && (
            <div style={{
              marginTop: 9, padding: '8px 10px', borderRadius: 5,
              background: 'rgba(127,29,29,0.35)', border: '1px solid rgba(248,113,113,0.4)',
              fontSize: 11.5, color: '#fecaca', lineHeight: 1.6,
            }}>
              {data.worked_example.legs.find((l) => l.needs_verification).flag_reason}
            </div>
          )}

          {data.worked_example.evidence && (
            <div style={{ marginTop: 8, fontSize: 11, color: '#94a3b8' }}>
              Evidence checked on disk at load:{' '}
              <strong style={{ color: '#cbd5e1' }}>
                {data.worked_example.evidence.crop_files_on_disk}
              </strong>{' '}
              crop file(s) across{' '}
              {data.worked_example.evidence.sightings_with_crops} of{' '}
              {data.worked_example.evidence.sightings_checked} sightings.
            </div>
          )}
        </div>
      )}

      <div style={{
        fontSize: 11.5, color: '#94a3b8', lineHeight: 1.65,
        background: 'rgba(148,163,184,0.07)', padding: '10px 12px',
        borderRadius: 6, border: '1px solid rgba(148,163,184,0.16)',
      }}>
        Every number here is recomputed from the live database when this page
        loads — press <em>Re-measure</em> and watch them move. The legs above are
        built by the same functions the Journeys page uses: the same road
        router, the same speed classifier. A stage that lacks the data to
        support a figure reports that instead of showing one, which is why some
        read <em>COLLECTING</em> — the appearance bridge refuses to quote an
        accuracy until it has enough plate-labelled rows to measure one.
      </div>
    </div>
  );
}
