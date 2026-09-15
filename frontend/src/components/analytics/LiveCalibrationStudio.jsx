// frontend/src/components/analytics/LiveCalibrationStudio.jsx
//
// Ground-plane calibration for a camera: mark points on the road, give each
// one its real position, and the solver turns pixels into metres.
//
// What this tool can and cannot do is worth stating, because the previous
// version blurred it. Scale cannot be recovered from a single image. No
// amount of processing tells you whether a road is 7 m or 14 m across, so
// somebody has to measure it — with a wheel, from a survey, or off satellite
// imagery — and type it in. Everything else is arithmetic.
//
// The version this replaces filled those measurements in by itself. Opening
// an uncalibrated camera pre-placed five points carrying a 14 m x 16 m
// rectangle, identical for every camera in the fleet; clicking the frame
// assigned each new point world_m = [0, 10 + n*3.5]; and the save path
// defaulted held-out error to 0.055 m and the quality gate to "good" when its
// own solve had not run. A camera could be marked calibrated to 5.5 cm
// without anyone having measured anything.

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { apiFetch } from '../../utils/authClient';
import '../../styles/calibration.css';

// Five is the smallest number of points that can be checked. Four determine
// an 8-DOF homography exactly, so the fit passes through all of them and the
// residual is zero whatever the points mean.
const MIN_POINTS_TO_VALIDATE = 5;

// Held-out error thresholds, in metres, matching the server's quality gate.
const GATE_GOOD_M = 0.5;
const GATE_DEGRADED_M = 1.0;

function fmtMetres(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return null;
  return v < 1 ? `${(v * 100).toFixed(1)} cm` : `${v.toFixed(2)} m`;
}

function GateBadge({ gate, heldOut }) {
  const label = {
    good: 'Validated',
    degraded: 'Degraded',
    rejected: 'Rejected',
    unvalidated: 'Not validated',
    uncalibrated: 'Not calibrated',
  }[gate] || gate || 'Unknown';

  return (
    <span className={`cal-gate cal-gate--${gate || 'unknown'}`}>
      {label}
      {fmtMetres(heldOut) && <em> · held-out {fmtMetres(heldOut)}</em>}
    </span>
  );
}

// A different question from GateBadge's "how good is this calibration": this
// answers "validated against what". A camera can be quality_gate=good and
// still only surveyed — a held-out ground point, spatial accuracy only — vs
// cross_validated, checked against an independent camera's leg (spatial
// accuracy AND the time axis, e.g. CAM_11's 27.7-vs-27.8 km/h). Showing only
// GateBadge would let a fresh survey camera read exactly like CAM_11 on
// screen; it should not.
function SpeedConfidenceBadge({ confidence }) {
  if (!confidence) return null;
  const copy = {
    surveyed: {
      label: 'Speed: surveyed',
      title: 'Speed comes from a held-out ground-control point — validates '
        + 'spatial accuracy. Not yet checked against an independent camera.',
    },
    cross_validated: {
      label: 'Speed: cross-validated',
      title: 'Speed checked against an independent camera seeing the same '
        + 'vehicle — validates spatial accuracy AND timing, not just position.',
    },
  }[confidence];
  if (!copy) return null;
  return (
    <span className={`cal-conf cal-conf--${confidence}`} title={copy.title}>
      {copy.label}
    </span>
  );
}

export default function LiveCalibrationStudio() {
  const [cameras, setCameras] = useState([]);
  const [camerasError, setCamerasError] = useState(null);
  const [camId, setCamId] = useState(null);

  const [detail, setDetail] = useState(null);
  const [detailError, setDetailError] = useState(null);
  const [points, setPoints] = useState([]);
  const [heldOutIndex, setHeldOutIndex] = useState(null);

  const [solve, setSolve] = useState(null);          // last solve result
  const [busy, setBusy] = useState(null);            // 'solve' | 'save' | 'auto'
  const [notice, setNotice] = useState(null);        // { kind, text }
  const [autoDetect, setAutoDetect] = useState(null);
  const [drift, setDrift] = useState(null);

  const [frameUrl, setFrameUrl] = useState(null);
  // Bumped by the "Refresh frame" button. The reference frame is deliberately
  // STILL — control points are placed by clicking the road, and an image that
  // moved under the cursor would make that impossible — but the frame you
  // happen to get can be unusable (this feed is a 12-hour recording played at
  // wall-clock position, so at night it may be too dark to see the kerb).
  // This pulls a newer still on demand without forcing a camera switch.
  const [frameNonce, setFrameNonce] = useState(0);
  const [frameError, setFrameError] = useState(null);
  const [natural, setNatural] = useState(null);
  const [showGrid, setShowGrid] = useState(true);
  const [dragIdx, setDragIdx] = useState(null);
  const imgRef = useRef(null);

  // Pixel coordinates are meaningless without the frame they were measured
  // in. Prefer the dimensions of the image actually on screen — that is what
  // the operator is clicking — and fall back to what the API reported.
  const frameW = natural?.w || detail?.frame_size?.[0] || null;
  const frameH = natural?.h || detail?.frame_size?.[1] || null;

  // ── data loading ────────────────────────────────────────────────────────

  const loadCameras = useCallback(async () => {
    try {
      const res = await apiFetch('/analytics/calibration/cameras');
      if (!res?.ok) throw new Error(`HTTP ${res?.status}`);
      const data = await res.json();
      const list = data.cameras || [];
      setCameras(list);
      setCamerasError(null);
      setCamId((cur) => cur || list[0]?.camera_id || null);
    } catch (e) {
      // No silent fallback list. A hardcoded fleet marked is_calibrated:true
      // used to appear here whenever this call failed, so a dead backend
      // looked like six calibrated cameras.
      setCameras([]);
      setCamerasError(e.message || 'Could not reach the calibration service');
    }
  }, []);

  const loadDetail = useCallback(async (id) => {
    if (!id) return;
    try {
      const res = await apiFetch(`/analytics/calibration/camera/${id}`);
      if (!res?.ok) throw new Error(`HTTP ${res?.status}`);
      const data = await res.json();
      setDetail(data);
      setDetailError(null);
      setPoints(data.ground_control_points || []);
      setHeldOutIndex(
        data.held_out_index ?? ((data.ground_control_points?.length || 0) - 1),
      );
      setSolve(null);
      setNotice(null);
      setAutoDetect(null);
    } catch (e) {
      setDetail(null);
      setDetailError(e.message || 'Could not load this camera');
    }
  }, []);

  const loadDrift = useCallback(async (id) => {
    if (!id) return;
    try {
      const res = await apiFetch(`/analytics/calibration/drift-status/${id}`);
      setDrift(res?.ok ? await res.json() : null);
    } catch {
      setDrift(null);
    }
  }, []);

  // The reference frame has to be fetched, not linked.
  //
  // It was an <img src={`${API_BASE}/…/frame/${camId}`}>, and the endpoint
  // requires a bearer token. An <img> tag cannot send an Authorization
  // header, so every request came back 401 and the canvas showed "No frame
  // available for CAM_01" for a camera with five clips on disk. Putting the
  // token in the query string would work and would also write it into server
  // logs and browser history, so instead the bytes are fetched through the
  // same authenticated client as everything else and handed to the <img> as
  // an object URL.
  useEffect(() => {
    if (!camId) return undefined;
    let cancelled = false;
    let objectUrl = null;

    (async () => {
      setFrameUrl(null);
      setFrameError(null);
      setNatural(null);
      try {
        const res = await apiFetch(`/analytics/calibration/frame/${camId}`);
        if (!res?.ok) {
          let detailText = `HTTP ${res?.status}`;
          try { detailText = (await res.json()).detail || detailText; } catch { /* not JSON */ }
          if (!cancelled) setFrameError(detailText);
          return;
        }
        const blob = await res.blob();
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setFrameUrl(objectUrl);
      } catch (e) {
        if (!cancelled) setFrameError(e.message || 'Could not load the frame');
      }
    })();

    return () => {
      cancelled = true;
      // Object URLs pin the decoded image in memory until revoked, and this
      // component reloads one per camera click.
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [camId, frameNonce]);

  useEffect(() => { loadCameras(); }, [loadCameras]);

  useEffect(() => {
    loadDetail(camId);
    loadDrift(camId);
    // Sway is measured over a rolling window of processed frames, so it moves
    // on the scale of seconds, not milliseconds. Polling faster than the
    // pipeline produces frames only burns requests.
    const t = setInterval(() => loadDrift(camId), 10_000);
    return () => clearInterval(t);
  }, [camId, loadDetail, loadDrift]);

  // ── point editing ───────────────────────────────────────────────────────

  const toFramePx = useCallback((clientX, clientY) => {
    const el = imgRef.current;
    if (!el || !frameW || !frameH) return null;
    const r = el.getBoundingClientRect();
    return {
      x: Math.round(Math.max(0, Math.min(frameW, ((clientX - r.left) / r.width) * frameW))),
      y: Math.round(Math.max(0, Math.min(frameH, ((clientY - r.top) / r.height) * frameH))),
    };
  }, [frameW, frameH]);

  const addPoint = (e) => {
    if (dragIdx !== null) return;
    const p = toFramePx(e.clientX, e.clientY);
    if (!p) return;
    // world_m stays null. The operator supplies it; the tool does not guess.
    setPoints((prev) => [...prev, {
      label: `Point ${prev.length + 1}`,
      pixel: [p.x, p.y],
      world_m: null,
      source: 'operator',
    }]);
    setSolve(null);
  };

  useEffect(() => {
    if (dragIdx === null) return undefined;
    const move = (e) => {
      const p = toFramePx(e.clientX, e.clientY);
      if (!p) return;
      setPoints((prev) => prev.map((pt, i) =>
        i === dragIdx ? { ...pt, pixel: [p.x, p.y] } : pt));
    };
    const up = () => { setDragIdx(null); setSolve(null); };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    return () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
    };
  }, [dragIdx, toFramePx]);

  const setWorld = (idx, axis, raw) => {
    setPoints((prev) => prev.map((pt, i) => {
      if (i !== idx) return pt;
      const w = pt.world_m ? [...pt.world_m] : [null, null];
      w[axis] = raw === '' ? null : Number(raw);
      return { ...pt, world_m: w };
    }));
    setSolve(null);
  };

  const removePoint = (idx) => {
    setPoints((prev) => prev.filter((_, i) => i !== idx));
    setHeldOutIndex((h) => (h === null ? h : Math.max(0, Math.min(h, points.length - 2))));
    setSolve(null);
  };

  // A point is usable only once both of its ground coordinates are known.
  const complete = useMemo(() => points.filter(
    (p) => Array.isArray(p.world_m)
      && Number.isFinite(p.world_m[0]) && Number.isFinite(p.world_m[1]),
  ), [points]);

  const missing = points.length - complete.length;
  const canSolve = complete.length >= 4 && missing === 0;
  const canValidate = complete.length >= MIN_POINTS_TO_VALIDATE;

  // ── actions ─────────────────────────────────────────────────────────────

  const runSolve = async () => {
    setBusy('solve');
    setNotice(null);
    try {
      const res = await apiFetch('/analytics/calibration/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_id: camId,
          ground_control_points: points,
          held_out_index: heldOutIndex,
          frame_size: [frameW, frameH],
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setSolve(null);
        setNotice({ kind: 'error', text: data.detail || `HTTP ${res.status}` });
        return;
      }
      setSolve(data);
      setNotice({ kind: data.quality_gate === 'good' ? 'ok' : 'warn', text: data.message });
    } catch (e) {
      setNotice({ kind: 'error', text: e.message || 'Network error' });
    } finally {
      setBusy(null);
    }
  };

  const runSave = async () => {
    setBusy('save');
    setNotice(null);
    try {
      // The server re-solves from the points and computes its own error and
      // gate; nothing derived is sent from here. That is deliberate — this
      // component used to post held_out_error_m: 0.055 and quality_gate:
      // 'good' as defaults, and the server stored them.
      const res = await apiFetch('/analytics/calibration/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_id: camId,
          ground_control_points: points,
          held_out_index: heldOutIndex,
          frame_size: [frameW, frameH],
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setNotice({ kind: 'error', text: data.detail || `HTTP ${res.status}` });
        return;
      }
      setNotice({
        kind: 'ok',
        text: data.hot_reloaded
          ? `Saved. The running pipeline picked it up immediately.`
          : `Saved. ${data.note || ''}`,
      });
      loadCameras();
      loadDetail(camId);
    } catch (e) {
      setNotice({ kind: 'error', text: e.message || 'Network error' });
    } finally {
      setBusy(null);
    }
  };

  const runAutoDetect = async () => {
    setBusy('auto');
    setNotice(null);
    setAutoDetect(null);
    try {
      const res = await apiFetch(
        `/analytics/calibration/auto-detect/${camId}`, { method: 'POST' },
      );
      const data = await res.json();
      if (!res.ok) {
        setNotice({ kind: 'error', text: data.detail || `HTTP ${res.status}` });
        return;
      }
      setAutoDetect(data);
      if (!data.auto_detection?.success) {
        setNotice({ kind: 'warn', text: data.auto_detection?.reason || 'No vanishing point found.' });
      }
    } catch (e) {
      setNotice({ kind: 'error', text: e.message || 'Network error' });
    } finally {
      setBusy(null);
    }
  };

  const adoptSuggested = () => {
    const s = autoDetect?.auto_detection?.suggested_gcps || [];
    if (!s.length) return;
    // Positions only. Their ground coordinates are blank and the Solve button
    // stays disabled until they are filled in — adopting used to hand over a
    // fixed rectangle of invented metres and immediately solve on it.
    setPoints(s.map((p, i) => ({
      label: `Point ${i + 1}`,
      pixel: p.pixel,
      world_m: null,
      source: 'auto-detected road segment',
    })));
    setHeldOutIndex(s.length - 1);
    setSolve(null);
    setNotice({
      kind: 'warn',
      text: `${s.length} points placed on the detected road. Enter the ground `
        + `coordinates of each before solving — they cannot be derived from `
        + `the image.`,
    });
  };

  // ── render ──────────────────────────────────────────────────────────────

  const calibratedCount = cameras.filter((c) => c.is_calibrated).length;

  return (
    <div className="cal-studio">
      <header className="cal-head">
        <div>
          <h3>Ground-plane calibration</h3>
          <p>
            Mark points on the road and give each its real position. Distances
            on the ground have to be measured — they cannot be recovered from
            the image.
          </p>
        </div>
        <div className="cal-fleet-summary">
          {camerasError ? (
            <span className="cal-gate cal-gate--rejected">{camerasError}</span>
          ) : (
            <span>
              <strong>{calibratedCount}</strong> of <strong>{cameras.length}</strong> cameras calibrated
            </span>
          )}
        </div>
      </header>

      {cameras.length > 0 && (
        <div className="cal-camera-strip" role="tablist" aria-label="Cameras">
          {cameras.map((c) => (
            <button
              key={c.camera_id}
              role="tab"
              aria-selected={c.camera_id === camId}
              className={`cal-cam ${c.camera_id === camId ? 'is-active' : ''}`}
              onClick={() => setCamId(c.camera_id)}
            >
              <span className={`cal-dot cal-dot--${c.quality_gate || 'uncalibrated'}`} />
              {c.camera_id}
            </button>
          ))}
        </div>
      )}

      {detailError && <div className="cal-notice cal-notice--error">{detailError}</div>}

      {detail && (
        <div className="cal-body">
          <div className="cal-canvas-wrap">
            <div className="cal-canvas-meta">
              <strong>{detail.location || camId}</strong>
              <GateBadge gate={detail.quality_gate} heldOut={detail.held_out_error_m} />
              <SpeedConfidenceBadge confidence={detail.speed_confidence} />
              {frameW
                ? <span className="cal-dim">{frameW} × {frameH}</span>
                : <span className="cal-dim cal-dim--warn">frame size unknown</span>}
              {detail.calibrated_by && (
                <span className="cal-dim">by {detail.calibrated_by}</span>
              )}
              <button
                type="button"
                className="btn btn-secondary"
                style={{ marginLeft: 'auto', fontSize: 11, padding: '3px 9px' }}
                onClick={() => setFrameNonce((n) => n + 1)}
                title="Pull a newer still from this camera. The frame stays still on purpose — control points are placed by clicking it."
              >
                ↻ Refresh frame
              </button>
            </div>

            {frameUrl ? (
              <div className="cal-canvas" onClick={addPoint}>
                <img
                  ref={imgRef}
                  src={frameUrl}
                  alt={`Reference frame from ${camId}`}
                  onLoad={(e) => setNatural({
                    w: e.currentTarget.naturalWidth,
                    h: e.currentTarget.naturalHeight,
                  })}
                  draggable={false}
                />
                {frameW && (
                  <svg viewBox={`0 0 ${frameW} ${frameH}`} preserveAspectRatio="none">
                    {showGrid && (solve?.grid_polylines || detail.grid_polylines || []).map((poly, i) => (
                      <polyline
                        key={i}
                        className="cal-grid"
                        points={poly.map(([x, y]) => `${x},${y}`).join(' ')}
                      />
                    ))}
                    {autoDetect?.auto_detection?.lane_lines?.map((l, i) => (
                      <line key={`ln${i}`} className="cal-lane"
                        x1={l[0]} y1={l[1]} x2={l[2]} y2={l[3]} />
                    ))}
                    {points.map((p, i) => {
                      const done = Array.isArray(p.world_m)
                        && Number.isFinite(p.world_m[0]) && Number.isFinite(p.world_m[1]);
                      const err = solve?.loo_point_errors_m?.[i];
                      return (
                        <g
                          key={i}
                          className={`cal-pt ${done ? '' : 'cal-pt--incomplete'} ${i === heldOutIndex ? 'cal-pt--heldout' : ''}`}
                          onMouseDown={(e) => { e.stopPropagation(); setDragIdx(i); }}
                        >
                          <circle cx={p.pixel[0]} cy={p.pixel[1]} r={Math.max(6, frameW / 160)} />
                          <text x={p.pixel[0] + frameW / 90} y={p.pixel[1] - frameW / 160}>
                            {i + 1}{Number.isFinite(err) ? ` · ${fmtMetres(err)}` : ''}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                )}
              </div>
            ) : frameError ? (
              <div className="cal-notice cal-notice--warn">
                <strong>No frame from {camId}.</strong> {frameError}
                {' '}Calibration is done against a real frame from this camera,
                so it cannot proceed until one is available.
              </div>
            ) : (
              <div className="cal-canvas cal-canvas--loading" role="status">
                Loading a frame from {camId}…
              </div>
            )}

            <div className="cal-canvas-tools">
              <label>
                <input type="checkbox" checked={showGrid}
                  onChange={() => setShowGrid((v) => !v)} />
                Metric grid
              </label>
              <span className="cal-hint">Click the road to add a point · drag to adjust</span>
            </div>
          </div>

          <aside className="cal-side">
            <section>
              <h4>Control points</h4>
              {points.length === 0 && (
                <p className="cal-empty">
                  None yet. Click a feature on the road you can also identify
                  on the ground or on satellite imagery — a kerb corner, a
                  stop line end, a manhole.
                </p>
              )}

              {points.map((p, i) => {
                const w = p.world_m || [null, null];
                const err = solve?.loo_point_errors_m?.[i];
                return (
                  <div key={i} className="cal-point-row">
                    <span className="cal-point-n">{i + 1}</span>
                    <span className="cal-point-px">
                      {p.pixel[0]}, {p.pixel[1]} px
                    </span>
                    <input
                      type="number" step="0.1" placeholder="X m"
                      value={Number.isFinite(w[0]) ? w[0] : ''}
                      onChange={(e) => setWorld(i, 0, e.target.value)}
                      aria-label={`Point ${i + 1} lateral position in metres`}
                    />
                    <input
                      type="number" step="0.1" placeholder="Y m"
                      value={Number.isFinite(w[1]) ? w[1] : ''}
                      onChange={(e) => setWorld(i, 1, e.target.value)}
                      aria-label={`Point ${i + 1} distance in metres`}
                    />
                    <button
                      className={`cal-holdout ${i === heldOutIndex ? 'is-on' : ''}`}
                      onClick={() => setHeldOutIndex(i)}
                      title="Hold this point out of the fit and predict it"
                    >
                      hold out
                    </button>
                    {Number.isFinite(err) && (
                      <span className={`cal-err ${err > GATE_DEGRADED_M ? 'is-bad' : err > GATE_GOOD_M ? 'is-warn' : ''}`}>
                        {fmtMetres(err)}
                      </span>
                    )}
                    <button className="cal-remove" onClick={() => removePoint(i)}
                      aria-label={`Remove point ${i + 1}`}>×</button>
                  </div>
                );
              })}

              {points.length > 0 && (
                <p className={`cal-readiness ${canValidate ? 'is-ok' : 'is-warn'}`}>
                  {missing > 0
                    ? `${missing} point${missing > 1 ? 's' : ''} still need ground coordinates.`
                    : canValidate
                      ? `${complete.length} points — enough to fit and validate.`
                      : `${complete.length} points. Four fix the homography exactly, `
                        + `so the error would be zero by construction. Add `
                        + `${MIN_POINTS_TO_VALIDATE - complete.length} more to measure accuracy.`}
                </p>
              )}

              <div className="cal-actions">
                <button onClick={runSolve} disabled={!canSolve || busy === 'solve'}>
                  {busy === 'solve' ? 'Solving…' : 'Solve'}
                </button>
                <button
                  onClick={runSave}
                  disabled={!canValidate || busy === 'save' || solve?.quality_gate === 'rejected'}
                  title={!canValidate
                    ? `${MIN_POINTS_TO_VALIDATE} validated points are required before saving`
                    : undefined}
                >
                  {busy === 'save' ? 'Saving…' : 'Save calibration'}
                </button>
              </div>
            </section>

            {solve && (
              <section className="cal-result">
                <h4>Fit</h4>
                <dl>
                  <dt>Held-out point</dt>
                  <dd>{fmtMetres(solve.held_out_error_m) || 'not measured'}</dd>
                  <dt>Leave-one-out RMS</dt>
                  <dd>{fmtMetres(solve.loo_rms_error_m) || '—'}</dd>
                  <dt>Condition number</dt>
                  <dd>{solve.condition_number?.toExponential(1)}</dd>
                  <dt>Horizon clearance</dt>
                  <dd>{Math.round(solve.horizon_distance_px)} px</dd>
                </dl>
                <GateBadge gate={solve.quality_gate} heldOut={solve.held_out_error_m} />
              </section>
            )}

            <section className="cal-auto">
              <h4>Find the road automatically</h4>
              <p>
                Detects the road direction from lane markings and kerbs. It
                places points on the road; their ground coordinates still have
                to be measured.
              </p>
              <button onClick={runAutoDetect} disabled={busy === 'auto'}>
                {busy === 'auto' ? 'Looking…' : 'Detect road'}
              </button>
              {autoDetect?.auto_detection?.success && (
                <>
                  <dl>
                    <dt>Vanishing point</dt>
                    <dd>
                      {autoDetect.auto_detection.vp[0]}, {autoDetect.auto_detection.vp[1]}
                    </dd>
                    <dt>Segments agreeing</dt>
                    <dd>
                      {autoDetect.auto_detection.segments_used}
                      {' '}(RMS {autoDetect.auto_detection.vp_rms_px} px)
                    </dd>
                  </dl>
                  <button onClick={adoptSuggested}>
                    Use these {autoDetect.auto_detection.suggested_gcps.length} positions
                  </button>
                </>
              )}
            </section>

            <section className="cal-drift">
              <h4>Camera steadiness</h4>
              {!drift ? (
                <p className="cal-empty">No reading.</p>
              ) : drift.measured ? (
                <>
                  <p className={`cal-drift-status cal-drift--${drift.status?.toLowerCase()}`}>
                    {drift.status} · {drift.jitter_rms_px} px RMS
                  </p>
                  <dl>
                    <dt>Largest shift</dt>
                    <dd>{drift.max_shift_px} px</dd>
                    <dt>Frames with movement</dt>
                    <dd>{drift.moving_frame_pct}%</dd>
                    <dt>Window</dt>
                    <dd>{drift.samples} frames</dd>
                  </dl>
                  <p className="cal-hint">
                    Measured from the frame-to-frame motion the tracker already
                    solves for. A camera that moves more than a few pixels
                    between frames invalidates its calibration.
                  </p>
                </>
              ) : (
                <p className="cal-empty">{drift.reason || 'Not measured yet.'}</p>
              )}
            </section>
          </aside>
        </div>
      )}

      {notice && (
        <div className={`cal-notice cal-notice--${notice.kind}`} role="status">
          {notice.text}
        </div>
      )}
    </div>
  );
}
