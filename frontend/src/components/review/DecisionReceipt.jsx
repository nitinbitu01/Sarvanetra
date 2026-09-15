// frontend/src/components/review/DecisionReceipt.jsx
// Shown after a successful review submission.
// Reveals the confidence score (hidden during review to prevent bias).

export default function DecisionReceipt({ receipt, alert, onClose }) {
  const DECISION_CONFIG = {
    CONFIRMED: {
      icon: '✅',
      label: 'Confirmed Match',
      color: '#22c55e',
      bg: 'var(--receipt-confirm-bg, rgba(34,197,94,0.08))',
      border: '#166534',
    },
    REJECTED: {
      icon: '❌',
      label: 'Rejected — False Alarm',
      color: '#ef4444',
      bg: 'var(--receipt-reject-bg, rgba(239,68,68,0.08))',
      border: '#991b1b',
    },
    UNCERTAIN: {
      icon: '⚠️',
      label: 'Escalated for Second Review',
      color: '#f59e0b',
      bg: 'var(--receipt-uncertain-bg, rgba(245,158,11,0.08))',
      border: '#92400e',
    },
  };

  const cfg = DECISION_CONFIG[receipt?.decision] ?? DECISION_CONFIG.UNCERTAIN;

  const trustColor = {
    HIGH:   '#22c55e',
    MEDIUM: '#f59e0b',
    LOW:    '#ef4444',
  }[receipt?.trust_level] ?? '#6b7280';

  const confidencePct = receipt?.confidence_score != null
    ? `${(receipt.confidence_score * 100).toFixed(1)}%`
    : '—';

  return (
    <div
      className="decision-receipt"
      role="status"
      aria-live="assertive"
      style={{
        background: cfg.bg,
        border: `1px solid ${cfg.border}`,
        borderRadius: 16,
        padding: 28,
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 20 }}>
        <span style={{ fontSize: 40 }} aria-hidden="true">{cfg.icon}</span>
        <div>
          <h3 style={{ color: cfg.color, fontSize: 18, fontWeight: 700, margin: 0 }}>
            Decision Recorded: {cfg.label}
          </h3>
          <p style={{ color: 'var(--text-muted)', fontSize: 12, margin: '4px 0 0' }}>
            Alert #{alert?.alert_id} · {new Date().toLocaleTimeString('en-IN')}
          </p>
        </div>
      </div>

      {/* Metadata Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 12,
        marginBottom: 20,
      }}>
        {receipt?.vault_entry_id && (
          <ReceiptField
            label="Vault Entry ID"
            value={receipt.vault_entry_id}
            mono
          />
        )}
        {receipt?.trust_level && (
          <ReceiptField
            label="Trust Level"
            value={receipt.trust_level}
            color={trustColor}
          />
        )}
        {receipt?.next_action && (
          <ReceiptField
            label="Vault Action"
            value={receipt.next_action.replace(/_/g, ' ')}
          />
        )}
        {receipt?.outcome_message && (
          <ReceiptField
            label="Outcome"
            value={receipt.outcome_message}
          />
        )}
      </div>

      {/* Confidence Score (revealed post-decision) */}
      <div style={{
        padding: '12px 16px',
        background: 'rgba(255,255,255,0.04)',
        borderRadius: 8,
        marginBottom: 20,
      }}>
        <p style={{ color: 'var(--text-muted)', fontSize: 11, margin: '0 0 6px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          🤖 Model Confidence (revealed post-decision)
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            flex: 1,
            height: 8,
            background: '#374151',
            borderRadius: 4,
            overflow: 'hidden',
          }}>
            <div style={{
              width: confidencePct,
              height: '100%',
              background: trustColor,
              borderRadius: 4,
              transition: 'width 0.8s ease',
            }} />
          </div>
          <span style={{ color: trustColor, fontWeight: 700, fontSize: 16, minWidth: 60, textAlign: 'right' }}>
            {confidencePct}
          </span>
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: 11, margin: '6px 0 0' }}>
          Score was hidden during review to prevent confirmation bias.
        </p>
      </div>

      {/* Legal Notice */}
      <p style={{ color: 'var(--text-muted)', fontSize: 11, margin: '0 0 20px' }}>
        🔒 Decision permanently logged to audit chain under your Officer ID.
        Biometric data governed by DPDPA 2023 · IT Act 2000 S.72A.
      </p>

      {/* Close Button */}
      <button
        onClick={onClose}
        className="btn-primary"
        style={{ width: '100%' }}
        autoFocus
      >
        ✓ Close & Return to Alert Feed
      </button>
    </div>
  );
}

function ReceiptField({ label, value, mono, color }) {
  return (
    <div style={{
      padding: '10px 12px',
      background: 'rgba(255,255,255,0.03)',
      borderRadius: 8,
      border: '1px solid #374151',
    }}>
      <p style={{ color: 'var(--text-muted)', fontSize: 11, margin: '0 0 4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label}
      </p>
      <p style={{
        color: color ?? 'var(--text-primary, #f9fafb)',
        fontSize: 13,
        fontWeight: 600,
        fontFamily: mono ? 'monospace' : 'inherit',
        margin: 0,
        wordBreak: 'break-all',
      }}>
        {value}
      </p>
    </div>
  );
}
