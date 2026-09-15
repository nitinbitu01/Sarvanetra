// frontend/src/components/ProofClipModal.jsx
import React, { useState, useEffect, useRef } from 'react';
import { useAuth, API } from '../context/AuthContext';
import toast from 'react-hot-toast';

export default function ProofClipModal({ alert, alertId, onClose }) {
  const { authFetch, token } = useAuth();
  const [evidence, setEvidence] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [verifyState, setVerifyState] = useState('idle'); // idle | checking | intact | tampered
  const [copied, setCopied] = useState(false);
  const videoRef = useRef(null);

  const targetAlertId = alertId || alert?.id;

  useEffect(() => {
    let cancelled = false;
    async function fetchEvidenceMeta() {
      if (!targetAlertId) return;
      setLoading(true);
      setError(null);
      try {
        const res = await authFetch(`${API}/evidence/${targetAlertId}`);
        if (cancelled) return;
        if (!res.ok) {
          if (res.status === 404) {
            setError('Evidence clip is being generated or was not found.');
          } else {
            setError('Could not load evidence metadata.');
          }
          return;
        }
        const data = await res.json();
        setEvidence(data);
      } catch (err) {
        if (!cancelled) setError('Network error fetching evidence details.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    fetchEvidenceMeta();
    return () => { cancelled = true; };
  }, [targetAlertId, authFetch]);

  // Handle escape key to close
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const verifyIntegrity = async () => {
    setVerifyState('checking');
    try {
      const res = await authFetch(`${API}/alerts/${targetAlertId}/verify`);
      const data = await res.json();
      if (data.intact === true) {
        setVerifyState('intact');
        toast.success('Evidence SHA-256 seal is intact and verified!', { id: 'ev-verify' });
      } else if (data.intact === false) {
        setVerifyState('tampered');
        toast.error('Warning: Evidence checksum does not match stored hash!', { id: 'ev-verify' });
      } else {
        setVerifyState('idle');
      }
    } catch {
      setVerifyState('idle');
      toast.error('Integrity verification check failed.');
    }
  };

  const copyHash = (hash) => {
    if (!hash) return;
    navigator.clipboard?.writeText(hash);
    setCopied(true);
    toast.success('SHA-256 hash copied to clipboard');
    setTimeout(() => setCopied(false), 2500);
  };

  const tokenParam = token ? `?token=${encodeURIComponent(token)}` : '';
  const videoSrc = `${API}/evidence/${targetAlertId}/clip${tokenParam}`;
  // Only offer the player when evidence capture actually produced a clip.
  // A staged alert never has one, and a real alert can fail capture — the
  // endpoint reports status FAILED with a reason in both cases.
  const hasClip = !loading
    && !alert?.is_simulated
    && !!evidence
    && evidence.status !== 'FAILED'
    && !!(evidence.clip_path || evidence.download?.clip);
  const custodyPdfUrl = `${API}/evidence/${targetAlertId}/custody${tokenParam}`;

  const camName = alert?.camera_name || alert?.camera_id || evidence?.camera_id || 'CCTV Camera';
  const alertType = alert?.alert_type || evidence?.alert_type || 'INCIDENT_ALERT';
  const severity = (alert?.severity || evidence?.severity || 'CRITICAL').toUpperCase();
  const timeStr = alert?.created_at
    ? new Date(alert.created_at).toLocaleString()
    : evidence?.event_time
      ? new Date(evidence.event_time).toLocaleString()
      : 'Live Timestamp';

  const sb = alert?.score_breakdown || {};
  const cad = sb?.cad_dispatch;
  const ipcList = sb?.ipc_sections || sb?.mv_sections || [];
  const firNumber = sb?.fir_number || sb?.echallan_reference;

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 10000,
        background: 'rgba(5, 10, 20, 0.88)',
        backdropFilter: 'blur(12px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 16, animation: 'fadeIn 0.2s ease-out',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        style={{
          width: '100%', maxWidth: 880, maxHeight: '92vh',
          background: 'linear-gradient(180deg, #0f172a 0%, #080d1a 100%)',
          borderRadius: 16,
          border: '1px solid rgba(59, 130, 246, 0.35)',
          boxShadow: '0 25px 60px -12px rgba(0, 0, 0, 0.8), 0 0 35px rgba(59, 130, 246, 0.15)',
          overflowY: 'auto',
          display: 'flex', flexDirection: 'column',
        }}
      >
        {/* Header */}
        <div style={{
          padding: '16px 22px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: 'rgba(15, 23, 42, 0.75)',
          position: 'sticky', top: 0, zIndex: 10,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 26 }}>📹</span>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span style={{
                  fontSize: 11, fontWeight: 800, padding: '3px 8px', borderRadius: 4,
                  background: severity === 'CRITICAL' ? '#ef4444' : '#f59e0b',
                  color: '#fff', letterSpacing: 0.5,
                }}>
                  {severity}
                </span>
                <span style={{ fontSize: 16, fontWeight: 700, color: '#f8fafc' }}>
                  {alertType.replace(/_/g, ' ')}
                </span>
                {/* Out of 10, not 100. Every producer writes this on a 0-10
                    scale — the highest value anywhere in the database is 9.8,
                    on a stolen-vehicle hit — so "/100" rendered the most
                    serious alert the system raises as "9.5/100", which reads
                    as almost harmless. */}
                {alert?.danger_score != null && (
                  <span style={{ fontSize: 12, color: '#ef4444', fontWeight: 700, background: 'rgba(239,68,68,0.15)', padding: '2px 6px', borderRadius: 4 }}>
                    Severity: {Number(alert.danger_score).toFixed(1)}/10
                  </span>
                )}
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 3 }}>
                📷 {camName} {alert?.zone ? `· ${alert.zone}` : ''} {alert?.district ? `(${alert.district})` : ''} · ⏱️ {timeStr}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'rgba(255,255,255,0.08)', border: 'none',
              color: '#94a3b8', width: 34, height: 34, borderRadius: '50%',
              fontSize: 16, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 0.2s',
            }}
            title="Close (Esc)"
          >
            ✕
          </button>
        </div>

        {/* Video Player Section with AI Forensic Reticle */}
        <div style={{
          background: '#000',
          position: 'relative',
          width: '100%',
          maxHeight: 460,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          {/* The <video> element was rendered unconditionally, so an alert
              with no clip showed the browser's own "No video with supported
              format and MIME type found" over a large black rectangle — a
              broken player where an explanation belongs. Evidence capture
              already reports WHY there is no footage; show that instead. */}
          {hasClip ? (
            <video
              ref={videoRef}
              src={videoSrc}
              controls
              autoPlay
              loop
              playsInline
              style={{
                width: '100%',
                maxHeight: 460,
                objectFit: 'contain',
                display: 'block',
              }}
            >
              Your browser does not support the video tag.
            </video>
          ) : (
            <div style={{
              padding: '48px 32px', textAlign: 'center', color: '#94a3b8',
              maxWidth: 620,
            }}>
              <div style={{ fontSize: 34, marginBottom: 10 }}>🎞️</div>
              <div style={{ fontSize: 15, fontWeight: 700, color: '#e2e8f0', marginBottom: 8 }}>
                {alert?.is_simulated
                  ? 'No footage — this alert was staged for demonstration'
                  : 'No verifiable footage for this alert'}
              </div>
              <div style={{ fontSize: 12.5, lineHeight: 1.6 }}>
                {alert?.is_simulated
                  ? 'The detection logic is real and is exercised by this alert; '
                    + 'the triggering event was created deliberately, so there is '
                    + 'no recorded clip behind it. Everything else on this page '
                    + 'is what a genuine detection would carry.'
                  : (evidence?.failure_reason
                     || error
                     || 'Evidence capture did not produce a clip for this event.')}
              </div>
              {!alert?.is_simulated && evidence?.status && (
                <div style={{ fontSize: 11, marginTop: 10, color: '#64748b' }}>
                  evidence status: {evidence.status}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Forensic Suspect Details & Context Body */}
        <div style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Target Identification Box */}
          <div style={{
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid rgba(59, 130, 246, 0.25)',
            borderRadius: 10, padding: '14px 16px',
            display: 'flex', flexDirection: 'column', gap: 8,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#60a5fa', display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>🎯</span> TARGET IDENTIFICATION & SUSPECT PROFILE
              </div>
              {alert?.plate_text && (
                <div style={{
                  background: '#fef08a', color: '#854d0e', fontWeight: 800,
                  fontSize: 13, padding: '3px 8px', borderRadius: 4, border: '1px solid #eab308',
                }}>
                  ANPR: {alert.plate_text}
                </div>
              )}
            </div>

            <div style={{ fontSize: 15, fontWeight: 700, color: '#f8fafc' }}>
              {alert?.subject_label || 'Flagged CCTV Target'}
            </div>

            {/* Why This Alert Emerged (Forensic Rationale) */}
            <div style={{ fontSize: 13, color: '#cbd5e1', lineHeight: 1.5, background: 'rgba(15, 23, 42, 0.5)', padding: '10px 12px', borderRadius: 6 }}>
              <strong style={{ color: '#93c5fd' }}>Incident Rationale & Why Flagged:</strong> {alert?.description || 'Automated AI security tracking detected anomalous behavioral or biometric signature.'}
            </div>

            {/* Legal Charges & FIR Badges */}
            {(firNumber || (ipcList && ipcList.length > 0)) && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginTop: 4 }}>
                {firNumber && (
                  <span style={{ fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4, background: 'rgba(239, 68, 68, 0.15)', color: '#f87171', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                    📄 {firNumber}
                  </span>
                )}
                {ipcList.map((ipc, i) => (
                  <span key={i} style={{ fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4, background: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
                    ⚖️ {ipc}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Dial-112 CAD Patrol Unit Action */}
          {cad && (
            <div style={{
              background: 'rgba(16, 185, 129, 0.08)',
              border: '1px solid rgba(16, 185, 129, 0.25)',
              borderRadius: 8, padding: '10px 14px',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontSize: 20 }}>🚔</span>
                <div>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#34d399' }}>
                    DIAL-112 CAD PATROL DISPATCHED: {cad.call_sign} ({cad.unit_id})
                  </div>
                  <div style={{ fontSize: 12, color: '#a7f3d0' }}>
                    Officer: {cad.officer} · Distance: {cad.distance_km} km
                  </div>
                </div>
              </div>
              <div style={{
                background: '#059669', color: '#fff', fontWeight: 800, fontSize: 12,
                padding: '4px 10px', borderRadius: 6,
              }}>
                ETA: {cad.eta_minutes} MIN
              </div>
            </div>
          )}

          {/* Voice AI Spoken Alert Broadcast */}
          <div style={{
            background: 'rgba(59, 130, 246, 0.08)',
            border: '1px solid rgba(59, 130, 246, 0.25)',
            borderRadius: 8, padding: '10px 14px',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 20 }}>🎙️</span>
              <div>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#60a5fa' }}>
                  VOICE AI CONTROL ROOM BROADCAST (PA SYSTEM)
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>
                  Synthesized in Hindi, Gujarati, and English for officer dispatch
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                type="button"
                onClick={() => {
                  const a = new Audio(`${API}/alerts/${targetAlertId}/audio?lang=hi${tokenParam ? `&token=${encodeURIComponent(token)}` : ''}`);
                  a.play().catch(() => toast.error('Audio playback failed'));
                }}
                style={{
                  background: '#1e293b', border: '1px solid rgba(255,255,255,0.15)',
                  color: '#f8fafc', padding: '5px 10px', borderRadius: 6, fontSize: 11, fontWeight: 700, cursor: 'pointer'
                }}
              >
                🔊 Hindi
              </button>
              <button
                type="button"
                onClick={() => {
                  const a = new Audio(`${API}/alerts/${targetAlertId}/audio?lang=gu${tokenParam ? `&token=${encodeURIComponent(token)}` : ''}`);
                  a.play().catch(() => toast.error('Audio playback failed'));
                }}
                style={{
                  background: '#1e293b', border: '1px solid rgba(255,255,255,0.15)',
                  color: '#f8fafc', padding: '5px 10px', borderRadius: 6, fontSize: 11, fontWeight: 700, cursor: 'pointer'
                }}
              >
                🔊 Gujarati
              </button>
              <button
                type="button"
                onClick={() => {
                  const a = new Audio(`${API}/alerts/${targetAlertId}/audio?lang=en${tokenParam ? `&token=${encodeURIComponent(token)}` : ''}`);
                  a.play().catch(() => toast.error('Audio playback failed'));
                }}
                style={{
                  background: '#1e293b', border: '1px solid rgba(255,255,255,0.15)',
                  color: '#f8fafc', padding: '5px 10px', borderRadius: 6, fontSize: 11, fontWeight: 700, cursor: 'pointer'
                }}
              >
                🔊 English
              </button>
            </div>
          </div>

          {/* Cryptographic SHA-256 Hash Seal */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            background: 'rgba(0, 0, 0, 0.4)', border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: 8, padding: '12px 16px', flexWrap: 'wrap', gap: 10,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 260, flex: 1 }}>
              <span style={{ fontSize: 18 }}>🔒</span>
              <div style={{ overflow: 'hidden' }}>
                <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 700, letterSpacing: 0.5 }}>
                  CRYPTOGRAPHIC EVIDENCE SEAL (SHA-256):
                </div>
                <code
                  onClick={() => copyHash(evidence?.sha256 || alert?.evidence_hash)}
                  style={{
                    fontSize: 11, color: '#38bdf8', fontFamily: 'monospace',
                    cursor: 'pointer', wordBreak: 'break-all',
                  }}
                  title="Click to copy SHA-256 hash"
                >
                  {evidence?.sha256 || alert?.evidence_hash || 'Generating cryptographic seal…'}
                </code>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button
                onClick={verifyIntegrity}
                disabled={verifyState === 'checking'}
                style={{
                  background: verifyState === 'intact' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(59, 130, 246, 0.15)',
                  border: `1px solid ${verifyState === 'intact' ? '#22c55e' : '#3b82f6'}`,
                  color: verifyState === 'intact' ? '#4ade80' : '#60a5fa',
                  padding: '7px 14px', borderRadius: 6, fontSize: 12, fontWeight: 700,
                  cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6,
                }}
              >
                {verifyState === 'checking' ? '⏳ Verifying…' : verifyState === 'intact' ? '✅ Sealed & Intact' : '🛡️ Verify Hash'}
              </button>
            </div>
          </div>

          {/* Action Bar */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: 14,
            flexWrap: 'wrap', gap: 10,
          }}>
            <div style={{ fontSize: 12, color: '#64748b' }}>
              {evidence?.frame_count ? `Duration: ${evidence.captured_window?.duration_seconds || 6}s · Frames: ${evidence.frame_count}` : 'Court-Admissible Evidence Lock'}
            </div>

            <div style={{ display: 'flex', gap: 10 }}>
              <a
                href={videoSrc}
                download={`alert_${targetAlertId}_clip.mp4`}
                target="_blank"
                rel="noreferrer"
                style={{
                  background: 'rgba(255, 255, 255, 0.08)',
                  color: '#e2e8f0', textDecoration: 'none',
                  padding: '8px 14px', borderRadius: 8, fontSize: 13, fontWeight: 600,
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  display: 'flex', alignItems: 'center', gap: 6,
                }}
              >
                ⬇️ Download MP4
              </a>

              <a
                href={custodyPdfUrl}
                download={`alert_${targetAlertId}_custody.pdf`}
                target="_blank"
                rel="noreferrer"
                style={{
                  background: '#2563eb',
                  color: '#fff', textDecoration: 'none',
                  padding: '8px 14px', borderRadius: 8, fontSize: 13, fontWeight: 600,
                  border: '1px solid #3b82f6',
                  display: 'flex', alignItems: 'center', gap: 6,
                }}
              >
                📄 Custody PDF
              </a>

              <button
                onClick={onClose}
                style={{
                  background: 'transparent',
                  color: '#94a3b8', border: '1px solid rgba(255, 255, 255, 0.15)',
                  padding: '8px 14px', borderRadius: 8, fontSize: 13, cursor: 'pointer',
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
