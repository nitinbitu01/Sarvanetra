import React, { useCallback, useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup, useMap } from 'react-leaflet';
import { useAuth, API } from '../context/AuthContext';

/**
 * TrajectoryProofPanel — the plain version of the evidence.
 *
 * A map with the route on it, and under every pin the photograph that camera
 * actually recorded. Nothing else. The argument this makes is the one an
 * officer or a judge can check without being told how it works: here is the
 * same vehicle, at four places, in four photographs, in time order.
 *
 * A stop with no recoverable footage says so and shows no picture. That case
 * is real — the plate journeys in this database came from the live pipeline,
 * which does not write to the frame index the clip runs built — and inventing
 * a stand-in image would be the one thing that would make this panel worthless.
 */

function FitAll({ points }) {
  const map = useMap();
  useEffect(() => {
    const pts = points.filter((p) => p[0] != null && p[1] != null);
    if (pts.length === 1) map.setView(pts[0], 16);
    else if (pts.length > 1) map.fitBounds(pts, { padding: [45, 45] });
  }, [map, points]);
  return null;
}

export default function TrajectoryProofPanel({ initialId = 'GJ27F V8122' }) {
  const { authFetch } = useAuth();
  const [id, setId] = useState(initialId);
  const [query, setQuery] = useState(initialId);
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [openClip, setOpenClip] = useState(null);

  const load = useCallback(async (target) => {
    if (!target) return;
    setBusy(true); setErr(null); setOpenClip(null);
    try {
      const r = await authFetch(`${API}/journeys/${encodeURIComponent(target)}/proof-stops`);
      if (!r.ok) throw new Error(r.status === 404 ? `No journey found for ${target}` : `HTTP ${r.status}`);
      setData(await r.json());
    } catch (e) {
      setErr(e.message); setData(null);
    } finally {
      setBusy(false);
    }
  }, [authFetch]);

  useEffect(() => { load(id); }, [load, id]);

  const stops = data?.stops || [];
  const pts = stops.filter((s) => s.lat != null && s.lon != null).map((s) => [s.lat, s.lon]);
  // Fit to the observed stretches as well as the pins, so a segment running
  // off the edge is not cropped out of the first view.
  const allPts = [
    ...pts,
    ...stops.flatMap((s) => s.observed_path || []),
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <form
        onSubmit={(e) => { e.preventDefault(); setId(query.trim()); }}
        style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}
      >
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Plate number, e.g. GJ27F V8122"
          style={{
            padding: '7px 11px', borderRadius: 5, fontSize: 13,
            border: '1px solid rgba(148,163,184,0.35)', background: 'rgba(15,23,42,0.7)',
            color: '#f1f5f9', minWidth: 210, fontFamily: 'monospace',
          }}
        />
        <button type="submit" disabled={busy} style={{
          padding: '7px 14px', borderRadius: 5, fontSize: 13, fontWeight: 600,
          border: '1px solid rgba(56,189,248,0.5)', background: 'rgba(56,189,248,0.15)',
          color: '#38bdf8', cursor: busy ? 'wait' : 'pointer',
        }}>
          {busy ? 'Loading…' : 'Trace'}
        </button>
        {data && (
          <span style={{ fontSize: 12, color: '#94a3b8' }}>
            seen at <strong style={{ color: '#f1f5f9' }}>{data.cameras}</strong> cameras ·{' '}
            <strong style={{ color: data.with_picture ? '#34d399' : '#fbbf24' }}>
              {data.with_picture}
            </strong> with footage
          </span>
        )}
      </form>

      {err && (
        <div style={{ fontSize: 13, color: '#f87171' }}>{err}</div>
      )}

      {/* Plain words, first.
        *
        * The version of this screen that came before was a list of internal
        * counters — cached legs, negative p99, threshold values. It was a
        * developer's diagnostic wearing the word "proof", and it failed the
        * only test that matters: nobody reading it could say what the system
        * had done. Four sentences that a person understands are worth more
        * than twenty numbers they skip. */}
      {data && stops.length > 0 && (
        <div style={{
          fontSize: 13, color: '#e2e8f0', lineHeight: 1.75,
          background: 'rgba(52,211,153,0.08)', padding: '12px 14px',
          borderRadius: 7, border: '1px solid rgba(52,211,153,0.3)',
        }}>
          <div>
            <strong style={{ color: '#f1f5f9', fontFamily: 'monospace' }}>
              {data.plate_ground_truth || data.reid_id}
            </strong>{' '}
            was seen by <strong>{data.cameras} cameras</strong>
            {stops[0] && stops[stops.length - 1] && (
              <> between{' '}
                <strong>{new Date(stops[0].timestamp_utc).toLocaleTimeString()}</strong>
                {' '}and{' '}
                <strong>{new Date(stops[stops.length - 1].timestamp_utc).toLocaleTimeString()}</strong>
              </>
            )}.
          </div>
          {stops.every((s) => s.match_type === 'appearance') ? (
            <div>
              <strong style={{ color: '#fbbf24' }}>No camera could read its
              number plate.</strong> It was identified by how the vehicle looks
              — matched against everything each camera saw
              {stops.some((s) => s.score) && (
                <> at {Math.min(...stops.filter((s) => s.score).map((s) => s.score)).toFixed(2)}
                  –{Math.max(...stops.filter((s) => s.score).map((s) => s.score)).toFixed(2)} similarity</>
              )}.
            </div>
          ) : (
            <div>Identified by its number plate at each camera.</div>
          )}
          <div>
            The <strong style={{ color: '#34d399' }}>green lines</strong> are the
            stretches each camera actually watched. Every corner on them is a
            point surveyed on the ground — nothing was drawn by guesswork.
          </div>
          <div style={{ color: '#94a3b8' }}>
            Speeds are not shown here: these clips carry no clock, so the times
            are the order they were filmed in, not when the vehicle passed.
          </div>
        </div>
      )}

      {/* A search by plate that lands on an appearance id has to explain
        * itself, or it looks like the wrong vehicle came back. */}
      {data?.resolved_from_plate && (
        <div style={{
          fontSize: 11.5, color: '#cbd5e1', lineHeight: 1.6,
          background: 'rgba(56,189,248,0.09)', padding: '9px 12px',
          borderRadius: 6, border: '1px solid rgba(56,189,248,0.3)',
        }}>
          You searched{' '}
          <strong style={{ color: '#38bdf8', fontFamily: 'monospace' }}>
            {data.searched_for}
          </strong>. No camera on this route could read that plate — it is
          roughly 25&nbsp;px wide in this footage — so the vehicle is filed under
          the identity the appearance matcher gave it,{' '}
          <strong style={{ color: '#34d399', fontFamily: 'monospace' }}>
            {data.reid_id}
          </strong>. The route below is that vehicle.
        </div>
      )}

      {pts.length > 0 && (
        <div style={{ height: 340, borderRadius: 8, overflow: 'hidden',
                      border: '1px solid rgba(148,163,184,0.25)' }}>
          <MapContainer center={pts[0]} zoom={15} style={{ height: '100%', width: '100%' }}>
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution="&copy; OpenStreetMap"
            />
            <FitAll points={allPts} />
            {/* The link between cameras: dashed, because nobody watched it. */}
            <Polyline
              positions={pts}
              pathOptions={{ color: '#38bdf8', weight: 2, opacity: 0.6, dashArray: '7,6' }}
            />
            {/* What each camera actually watched: solid, because it was seen.
              * Drawn over the dashed link so the observed stretches stand out
              * from the inferred ones. */}
            {stops.map((s) => (
              s.observed_path ? (
                <Polyline
                  key={`obs-${s.camera_id}`}
                  positions={s.observed_path}
                  pathOptions={{ color: '#34d399', weight: 5, opacity: 0.95 }}
                >
                  <Popup>
                    <div style={{ fontSize: 12 }}>
                      <strong>{s.camera_id}</strong> watched this stretch<br />
                      {s.position_basis}
                    </div>
                  </Popup>
                </Polyline>
              ) : null
            ))}
            {stops.map((s, i) => (
              s.lat != null && s.lon != null ? (
                <CircleMarker
                  key={s.camera_id}
                  center={[s.lat, s.lon]}
                  radius={9}
                  pathOptions={{ color: '#0f172a', fillColor: '#34d399', fillOpacity: 1, weight: 2 }}
                >
                  <Popup>
                    <div style={{ fontSize: 12 }}>
                      <strong>{i + 1}. {s.camera_name}</strong><br />
                      {new Date(s.timestamp_utc).toLocaleString()}
                    </div>
                  </Popup>
                </CircleMarker>
              ) : null
            ))}
          </MapContainer>
        </div>
      )}

      {/* The photographs — the actual point of this screen */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(250px,1fr))',
        gap: 12,
      }}>
        {stops.map((s, i) => (
          <div key={s.camera_id} style={{
            border: '1px solid rgba(148,163,184,0.25)', borderRadius: 8,
            overflow: 'hidden', background: 'rgba(15,23,42,0.55)',
          }}>
            <div style={{
              padding: '7px 10px', display: 'flex', alignItems: 'center', gap: 7,
              borderBottom: '1px solid rgba(148,163,184,0.18)',
            }}>
              <span style={{
                width: 20, height: 20, borderRadius: '50%', background: '#34d399',
                color: '#0f172a', fontSize: 11, fontWeight: 800,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>{i + 1}</span>
              <strong style={{ fontSize: 12.5, color: '#f1f5f9' }}>{s.camera_id}</strong>
              <span style={{ marginLeft: 'auto', fontSize: 11, color: '#94a3b8',
                             fontVariantNumeric: 'tabular-nums' }}>
                {new Date(s.timestamp_utc).toLocaleTimeString()}
              </span>
            </div>

            {openClip === s.camera_id && s.clip_url ? (
              <video
                src={`${API.replace('/api/v1', '')}${s.clip_url}`}
                controls autoPlay loop muted playsInline
                style={{ width: '100%', display: 'block', background: '#000' }}
              />
            ) : s.image_url ? (
              <img
                src={`${API.replace('/api/v1', '')}${s.image_url}`}
                alt={`${s.camera_id} sighting`}
                style={{ width: '100%', display: 'block', background: '#000' }}
              />
            ) : (
              <div style={{
                padding: '26px 12px', textAlign: 'center', fontSize: 11.5,
                color: '#fbbf24', lineHeight: 1.6,
              }}>
                No recoverable footage for this sighting.<br />
                <span style={{ color: '#64748b' }}>
                  The read exists in the database, but this camera has no indexed
                  frame to cut from — so nothing is shown rather than a stand-in.
                </span>
              </div>
            )}

            <div style={{ padding: '7px 10px', fontSize: 11, color: '#94a3b8',
                          display: 'flex', alignItems: 'center', gap: 8 }}>
              <span>{s.camera_name}</span>
              {s.clip_url && (
                <button
                  onClick={() => setOpenClip(openClip === s.camera_id ? null : s.camera_id)}
                  style={{
                    marginLeft: 'auto', padding: '2px 8px', fontSize: 10.5,
                    borderRadius: 3, border: '1px solid rgba(56,189,248,0.45)',
                    background: 'transparent', color: '#38bdf8', cursor: 'pointer',
                  }}
                >
                  {openClip === s.camera_id ? 'Show frame' : 'Play clip'}
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {data && (
        <div style={{
          fontSize: 11.5, color: '#94a3b8', lineHeight: 1.65,
          background: 'rgba(148,163,184,0.07)', padding: '10px 12px',
          borderRadius: 6, border: '1px solid rgba(148,163,184,0.16)',
        }}>
          <div>
            <strong style={{ color: '#34d399' }}>Solid green</strong> is the
            stretch a camera actually watched;{' '}
            <strong style={{ color: '#38bdf8' }}>dashed blue</strong> is the gap
            between cameras, which nobody saw. Each picture is cut from the
            footage that camera recorded — not a stock image and not a
            re-render. Where a stop has no indexed frame the card says so and
            stays empty.
          </div>
          {data.timing_caveat && (
            <div style={{ marginTop: 6, color: '#fbbf24' }}>{data.timing_caveat}</div>
          )}
        </div>
      )}
    </div>
  );
}
