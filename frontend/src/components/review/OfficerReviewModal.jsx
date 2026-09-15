// frontend/src/components/review/OfficerReviewModal.jsx
// Human-In-The-Loop officer review modal with:
//   - Server-authoritative 4-second time gate (anti-fraud)
//   - AI confidence score HIDDEN during review (revealed in receipt)
//   - BroadcastChannel multi-tab lock guard
//   - Tab-blur PII protection (images blur when tab loses focus)
//   - Keyboard shortcuts: 1=Confirm, 2=Reject, 3=Escalate
//   - FocusTrap for accessibility
//   - Uses existing AuthContext (useAuth) — no Zustand

import { useState, useEffect, useRef, useCallback } from 'react';
import FocusTrap from 'focus-trap-react';
import toast from 'react-hot-toast';
import { useAuth } from '../../context/AuthContext';
import CountdownTimer from './CountdownTimer';
import DecisionReceipt from './DecisionReceipt';

const API = import.meta.env.VITE_API_URL || '/api/v1';
const GATE_DURATION = 4.0;

const ENV_LABELS = {
  DAY:         '☀️ Day',
  NIGHT:       '🌙 Night',
  NIGHT_GLARE: '💡 Night Glare',
  MONSOON:     '🌧️ Monsoon',
  DUST_STORM:  '🌪️ Dust Storm',
};

const ENV_COLORS = {
  DAY:         '#f59e0b',
  NIGHT:       '#6366f1',
  NIGHT_GLARE: '#f97316',
  MONSOON:     '#3b82f6',
  DUST_STORM:  '#92400e',
};

export default function OfficerReviewModal({ alert, isOpen, onClose, onReviewSubmitted }) {
  // ── State Machine ────────────────────────────────────────────────────────────
  // IDLE → LOADING → GATING → READY → SUBMITTING → SUCCESS | ERROR
  const [phase, setPhase]           = useState('IDLE');
  const [sessionData, setSessionData] = useState(null);
  const [gateUnlocked, setGateUnlocked] = useState(false);
  const [receipt, setReceipt]       = useState(null);
  const [submitError, setSubmitError] = useState(null);
  const [imagesBlurred, setImagesBlurred] = useState(false);
  const [probeError, setProbeError] = useState(false);
  const [galleryError, setGalleryError] = useState(false);
  const [isOnline, setIsOnline]     = useState(navigator.onLine);

  const { authFetch, user }         = useAuth();
  const broadcastRef                = useRef(null);
  const submittedRef                = useRef(false);

  // ── Network status ───────────────────────────────────────────────────────────
  useEffect(() => {
    const on  = () => setIsOnline(true);
    const off = () => setIsOnline(false);
    window.addEventListener('online',  on);
    window.addEventListener('offline', off);
    return () => { window.removeEventListener('online', on); window.removeEventListener('offline', off); };
  }, []);

  // ── Tab visibility blur (PII protection) ────────────────────────────────────
  useEffect(() => {
    const handler = () => setImagesBlurred(document.hidden);
    document.addEventListener('visibilitychange', handler);
    return () => document.removeEventListener('visibilitychange', handler);
  }, []);

  // ── Keyboard shortcuts ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (!gateUnlocked || phase !== 'READY') return;
      if (e.key === '1') handleSubmit('CONFIRMED');
      if (e.key === '2') handleSubmit('REJECTED');
      if (e.key === '3') handleSubmit('UNCERTAIN');
      if (e.key === 'Escape' && phase === 'LOADING') handleClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, gateUnlocked, phase]);

  // ── Open: Start session ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!isOpen || !alert) return;
    submittedRef.current = false;
    setPhase('IDLE');
    setGateUnlocked(false);
    setReceipt(null);
    setSubmitError(null);
    setProbeError(false);
    setGalleryError(false);

    // Multi-tab guard via BroadcastChannel
    try {
      broadcastRef.current = new BroadcastChannel('sentinel_review_lock');
      broadcastRef.current.postMessage({ type: 'REVIEW_LOCK', alert_id: alert.alert_id, officer_id: user?.username });
      broadcastRef.current.onmessage = (ev) => {
        if (ev.data?.type === 'REVIEW_LOCK'
            && ev.data?.alert_id === alert.alert_id
            && ev.data?.officer_id !== user?.username) {
          toast.error('⚠️ This alert is already being reviewed in another tab.');
          handleClose();
        }
      };
    } catch { /* BroadcastChannel unavailable on older browsers */ }

    initiateSession();

    return () => {
      broadcastRef.current?.close();
      if (!submittedRef.current && ['GATING', 'READY'].includes(phase)) {
        abandonSession();
      }
    };
  }, [isOpen, alert?.alert_id]);

  // ── Start review session ─────────────────────────────────────────────────────
  const initiateSession = async () => {
    setPhase('LOADING');
    try {
      const res = await authFetch(`${API}/review/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          alert_id: alert.alert_id,
          officer_id: user?.username,
          client_timestamp: new Date().toISOString(),
        }),
      });

      if (res.status === 409) {
        toast.error('⚠️ You already have an active review session.');
        handleClose(); return;
      }
      if (res.status === 423) {
        toast.error('🔒 Alert is currently locked by another officer.');
        handleClose(); return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      setSessionData(data);
      setPhase('GATING');
    } catch (err) {
      setPhase('ERROR_LOAD');
    }
  };

  // ── Gate complete callback ───────────────────────────────────────────────────
  const handleGateComplete = useCallback(() => {
    setGateUnlocked(true);
    setPhase('READY');
  }, []);

  // ── Submit decision ──────────────────────────────────────────────────────────
  const handleSubmit = async (decision) => {
    if (phase !== 'READY' || !gateUnlocked) return;
    if (!isOnline) { toast.error('🔴 No network — cannot submit. Please wait for connection.'); return; }

    setPhase('SUBMITTING');
    setSubmitError(null);

    try {
      const res = await authFetch(`${API}/review/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          review_session_id: sessionData?.review_session_id,
          officer_id: user?.username,
          decision,
          idempotency_key: crypto.randomUUID(),
        }),
      });

      if (res.status === 429) {
        // Server rejected — gate bypass attempt
        setGateUnlocked(false);
        setPhase('GATING');
        toast.error('⚠️ Submission rejected: minimum review time not met.');
        return;
      }
      if (res.status === 410) {
        toast.error('⏱️ Review session expired. Please reopen the alert.');
        handleClose(); return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      submittedRef.current = true;
      setReceipt({ ...data, decision });
      setPhase('SUCCESS');
      onReviewSubmitted?.(data);
    } catch (err) {
      setSubmitError('Network error — your decision was not saved. Please retry.');
      setPhase('READY');
    }
  };

  // ── Abandon session ──────────────────────────────────────────────────────────
  const abandonSession = async () => {
    if (!sessionData?.review_session_id) return;
    try {
      await authFetch(`${API}/review/abandon`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          review_session_id: sessionData.review_session_id,
          officer_id: user?.username,
        }),
      });
    } catch { /* silent — backend TTL cleans up */ }
  };

  // ── Close handler ────────────────────────────────────────────────────────────
  const handleClose = useCallback(() => {
    broadcastRef.current?.close();
    setPhase('IDLE');
    setSessionData(null);
    setReceipt(null);
    setSubmitError(null);
    setGateUnlocked(false);
    onClose();
  }, [onClose]);

  if (!isOpen) return null;

  return (
    <FocusTrap focusTrapOptions={{ allowOutsideClick: false }}>
      <div
        className="modal-overlay"
        role="dialog"
        aria-modal="true"
        aria-labelledby="review-modal-title"
        style={{
          position: 'fixed', inset: 0, zIndex: 999,
          background: 'rgba(0,0,0,0.85)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: 16,
        }}
      >
        <div style={{
          background: 'var(--bg-card, #1f2937)',
          border: '1px solid var(--border, #374151)',
          borderRadius: 20,
          width: '100%', maxWidth: 900,
          maxHeight: '92vh', overflowY: 'auto',
          boxShadow: '0 25px 60px rgba(0,0,0,0.6)',
        }}>

          {/* ── Header ── */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '18px 24px',
            borderBottom: '1px solid var(--border, #374151)',
            position: 'sticky', top: 0,
            background: 'var(--bg-card, #1f2937)',
            zIndex: 10,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <h2 id="review-modal-title" style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>
                🔍 Officer Review
              </h2>
              <span style={{ fontFamily: 'monospace', color: 'var(--text-muted)', fontSize: 12 }}>
                #{alert?.alert_id}
              </span>
              {sessionData?.environmental_condition && (
                <span style={{
                  padding: '2px 10px',
                  borderRadius: 20,
                  fontSize: 12,
                  fontWeight: 600,
                  background: `${ENV_COLORS[sessionData.environmental_condition]}22`,
                  color: ENV_COLORS[sessionData.environmental_condition] ?? '#9ca3af',
                  border: `1px solid ${ENV_COLORS[sessionData.environmental_condition] ?? '#374151'}`,
                }}>
                  {ENV_LABELS[sessionData.environmental_condition] ?? sessionData.environmental_condition}
                </span>
              )}
            </div>
            <button
              onClick={handleClose}
              disabled={phase === 'SUBMITTING'}
              aria-label="Close review modal"
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                color: 'var(--text-muted)', fontSize: 20, padding: 6,
                opacity: phase === 'SUBMITTING' ? 0.3 : 1,
              }}
            >
              ✕
            </button>
          </div>

          {/* ── Body ── */}
          <div style={{ padding: 24 }}>

            {/* LOADING */}
            {phase === 'LOADING' && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '60px 0', gap: 16 }}
                   role="status" aria-label="Starting review session">
                <div className="loading-spinner" style={{ width: 48, height: 48 }} />
                <p style={{ color: 'var(--text-muted)' }}>Starting secure review session…</p>
              </div>
            )}

            {/* LOAD ERROR */}
            {phase === 'ERROR_LOAD' && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '60px 0', gap: 16 }}
                   role="alert">
                <span style={{ fontSize: 48 }}>⚠️</span>
                <p style={{ color: '#ef4444', fontWeight: 600 }}>Failed to start review session</p>
                <p style={{ color: 'var(--text-muted)', fontSize: 13, textAlign: 'center' }}>
                  {!isOnline ? 'No network connection.' : 'Server error — please try again.'}
                </p>
                <button className="btn-primary" onClick={initiateSession} disabled={!isOnline}>
                  Retry
                </button>
              </div>
            )}

            {/* SUCCESS RECEIPT */}
            {phase === 'SUCCESS' && receipt && (
              <DecisionReceipt receipt={receipt} alert={alert} onClose={handleClose} />
            )}

            {/* MAIN REVIEW INTERFACE */}
            {['GATING', 'READY', 'SUBMITTING'].includes(phase) && sessionData && (
              <>
                {/* Camera context */}
                <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
                  {sessionData.camera_name && (
                    <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                      📷 {sessionData.camera_name}
                    </span>
                  )}
                  {sessionData.zone && (
                    <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                      📍 {sessionData.zone}
                    </span>
                  )}
                </div>

                {/* Side-by-side images */}
                <div style={{
                  display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24,
                  filter: imagesBlurred ? 'blur(20px)' : 'none',
                  transition: 'filter 0.3s ease',
                  pointerEvents: imagesBlurred ? 'none' : 'auto',
                }}
                  aria-hidden={imagesBlurred}
                >
                  {/* Probe */}
                  <ImagePanel
                    label="📹 CCTV Detected Probe"
                    src={sessionData.probe_image_url}
                    alt="Person detected by CCTV — probe crop for review"
                    hasError={probeError}
                    onError={() => setProbeError(true)}
                    errorMsg="Probe image unavailable"
                    errorNote="⚠️ Proceeding without probe image"
                  />
                  {/* Gallery */}
                  <ImagePanel
                    label="🎯 Watchlist Reference"
                    src={sessionData.gallery_image_url}
                    alt="Watchlist reference image for identity comparison"
                    hasError={galleryError}
                    onError={() => setGalleryError(true)}
                    errorMsg="Reference image unavailable"
                    errorNote="Contact system admin"
                  />
                </div>

                {/* Blur overlay warning */}
                {imagesBlurred && (
                  <div role="alert" style={{
                    textAlign: 'center', color: '#f59e0b', fontSize: 13,
                    padding: '10px 16px', background: 'rgba(245,158,11,0.1)',
                    borderRadius: 8, marginBottom: 16,
                  }}>
                    🔒 Images hidden — return to this tab to continue review
                  </div>
                )}

                {/* Offline warning */}
                {!isOnline && (
                  <div role="alert" style={{
                    textAlign: 'center', color: '#ef4444', fontSize: 13,
                    padding: '10px 16px', background: 'rgba(239,68,68,0.1)',
                    borderRadius: 8, marginBottom: 16,
                  }}>
                    🔴 No network connection — decision submission blocked
                  </div>
                )}

                {/* Countdown timer */}
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 20 }}>
                  <CountdownTimer
                    duration={GATE_DURATION}
                    onComplete={handleGateComplete}
                    isComplete={phase === 'READY' || phase === 'SUBMITTING'}
                  />
                </div>

                {/* Submit error */}
                {submitError && (
                  <div role="alert" style={{
                    padding: '12px 16px', background: 'rgba(239,68,68,0.1)',
                    border: '1px solid #991b1b', borderRadius: 10,
                    color: '#ef4444', fontSize: 13, marginBottom: 16,
                  }}>
                    ⚠️ {submitError}
                  </div>
                )}

                {/* Legal notice */}
                <p style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: 11, margin: '0 0 16px' }}>
                  Your decision is permanently recorded under your Officer ID.
                  Biometric data governed by DPDPA 2023 · IT Act 2000 S.72A.
                </p>

                {/* Decision buttons */}
                <div
                  style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}
                  role="group"
                  aria-label="Review decision options"
                  aria-describedby="gate-timer-status"
                >
                  <DecisionBtn
                    onClick={() => handleSubmit('CONFIRMED')}
                    disabled={phase !== 'READY' || !isOnline}
                    color="#16a34a" borderColor="#166534" bg="rgba(22,163,74,0.08)"
                    shortcut="1"
                  >
                    ✅ This IS the person
                    <small>Confirm Match</small>
                  </DecisionBtn>

                  <DecisionBtn
                    onClick={() => handleSubmit('REJECTED')}
                    disabled={phase !== 'READY' || !isOnline}
                    color="#dc2626" borderColor="#991b1b" bg="rgba(220,38,38,0.08)"
                    shortcut="2"
                  >
                    ❌ This is NOT the person
                    <small>False Alarm</small>
                  </DecisionBtn>

                  <DecisionBtn
                    onClick={() => handleSubmit('UNCERTAIN')}
                    disabled={phase !== 'READY' || !isOnline}
                    color="#d97706" borderColor="#92400e" bg="rgba(217,119,6,0.08)"
                    shortcut="3"
                  >
                    ⚠️ I'm not sure
                    <small>Escalate for Second Review</small>
                  </DecisionBtn>
                </div>

                {/* Submitting indicator */}
                {phase === 'SUBMITTING' && (
                  <div role="status" aria-live="polite" style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    gap: 10, marginTop: 20, color: 'var(--text-muted)', fontSize: 14,
                  }}>
                    <div className="loading-spinner" style={{ width: 18, height: 18 }} />
                    Saving your decision securely…
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </FocusTrap>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ImagePanel({ label, src, alt, hasError, onError, errorMsg, errorNote }) {
  return (
    <div>
      <p style={{
        fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
        letterSpacing: '0.05em', color: 'var(--text-muted)', margin: '0 0 8px',
      }}>
        {label}
      </p>
      <div style={{
        position: 'relative', background: '#111827', borderRadius: 12,
        overflow: 'hidden', aspectRatio: '1', border: '1px solid var(--border, #374151)',
      }}>
        {hasError ? (
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center',
            justifyContent: 'center', height: '100%', gap: 8,
            color: 'var(--text-muted)',
          }}>
            <span style={{ fontSize: 32 }}>🖼️</span>
            <p style={{ fontSize: 13, margin: 0 }}>{errorMsg}</p>
            <p style={{ fontSize: 11, color: '#f59e0b', margin: 0 }}>{errorNote}</p>
          </div>
        ) : (
          <img
            src={src}
            alt={alt}
            onError={onError}
            onContextMenu={(e) => e.preventDefault()}
            draggable={false}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        )}
      </div>
    </div>
  );
}

function DecisionBtn({ onClick, disabled, color, borderColor, bg, shortcut, children }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-disabled={disabled}
      aria-keyshortcuts={shortcut}
      style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', gap: 6,
        padding: '18px 12px',
        background: disabled ? 'rgba(255,255,255,0.02)' : bg,
        border: `1px solid ${disabled ? '#374151' : borderColor}`,
        borderRadius: 12, cursor: disabled ? 'not-allowed' : 'pointer',
        color: disabled ? '#4b5563' : color,
        fontSize: 14, fontWeight: 600,
        transition: 'all 0.2s ease',
        opacity: disabled ? 0.5 : 1,
        lineHeight: 1.4,
      }}
    >
      {children}
    </button>
  );
}
