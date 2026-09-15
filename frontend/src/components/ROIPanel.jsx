// frontend/src/components/ROIPanel.jsx
//
// ILLUSTRATIVE FIGURES. Not computed, not fetched, not derived from anything
// in this system.
//
// Do not wire these to a backend calculation. The numbers a real ROI claim
// would need — officer hours, false-alarm rates, response times — are not
// measured anywhere in this product, and a figure that LOOKS computed is
// worse than one that is obviously a placeholder: a reviewer would
// reasonably take it as evidence.
//
// The caption below is part of the component, not decoration. If these ever
// become real numbers, delete the caption in the same commit.
//
// This component fetches nothing, subscribes to nothing, and takes no props,
// so it cannot fail at runtime — which is why the dashboard does not wrap it
// in an error boundary.
const ROI_STATS = [
  { label: 'Officer hrs saved / week', value: '42 hrs' },
  { label: 'False alarm reduction', value: '68%' },
  { label: 'Avg response improvement', value: '4.2 min' },
  { label: 'Estimated annual saving', value: '₹18.4 L' },
];

export default function ROIPanel() {
  return (
    <div
      className="panel-content"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        padding: '10px 12px',
        height: '100%',
      }}
    >
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 8,
          flex: 1,
        }}
      >
        {ROI_STATS.map((s) => (
          <div
            key={s.label}
            style={{
              background: 'var(--bg-elevated, #111827)',
              border: '1px solid var(--border, #1F2937)',
              borderRadius: 'var(--radius-md, 4px)',
              padding: '12px 14px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
            }}
          >
            <div
              style={{
                fontFamily: 'var(--font-mono, monospace)',
                fontSize: 21,
                fontWeight: 500,
                color: 'var(--text-primary, #F9FAFB)',
                lineHeight: 1.1,
              }}
            >
              {s.value}
            </div>
            <div
              style={{
                fontFamily: 'var(--font-sans, sans-serif)',
                fontSize: 10.5,
                color: 'var(--text-secondary, #9CA3AF)',
                marginTop: 4,
                lineHeight: 1.3,
              }}
            >
              {s.label}
            </div>
          </div>
        ))}
      </div>
      <div
        style={{
          fontSize: 10,
          color: 'var(--text-muted, #6B7280)',
          fontStyle: 'italic',
          textAlign: 'center',
          paddingTop: 2,
        }}
      >
        Illustrative estimates — not measured by this system
      </div>
    </div>
  );
}
