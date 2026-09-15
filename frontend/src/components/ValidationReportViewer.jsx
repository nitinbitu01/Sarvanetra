// frontend/src/components/ValidationReportViewer.jsx
// Admin-only view of the latest reid_validation_*.md report.
// Renders markdown inline or shows a download prompt if the file is not found.

import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';

const API = import.meta.env.VITE_API_URL || '/api/v1';

// Minimal markdown → HTML renderer (headings, bold, tables, blockquotes, code)
function renderMarkdown(md) {
  if (!md) return '';
  let html = md
    // Headings
    .replace(/^#### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Inline code
    .replace(/`(.+?)`/g, '<code>$1</code>')
    // Horizontal rule
    .replace(/^---$/gm, '<hr/>')
    // Blockquotes
    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
    // Simple tables (basic)
    .replace(/\|(.+)\|/g, (match) => {
      const cells = match.split('|').filter(c => c.trim() && !c.match(/^[-| ]+$/));
      return '<tr>' + cells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>';
    })
    // Line breaks
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>');
  return `<p>${html}</p>`;
}

export default function ValidationReportViewer() {
  const { authFetch } = useAuth();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('rendered'); // 'rendered' | 'raw'

  useEffect(() => {
    authFetch(`${API}/reid/validation-report`)
      .then(async r => {
        if (r.ok) {
          setReport(await r.text());
        } else {
          setError('Failed to load validation report. Run scripts/reid_validate.py first.');
        }
      })
      .catch(() => setError('Network error loading validation report.'))
      .finally(() => setLoading(false));
  }, [authFetch]);

  if (loading) {
    return (
      <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>
        <div style={{ fontSize: 24, marginBottom: 8 }}>⏳</div>
        Loading validation report...
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <div style={{
          padding: '16px 20px', borderRadius: 10, background: 'var(--accent-yellow)11',
          border: '1px solid var(--accent-yellow)44', color: 'var(--accent-yellow)',
          marginBottom: 20,
        }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>⚠️ No Validation Report</div>
          <div style={{ fontSize: 13 }}>{error}</div>
        </div>
        <div className="card" style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
          <div style={{ fontWeight: 600, marginBottom: 10, color: 'var(--text-primary)' }}>How to generate a report:</div>
          <pre style={{
            background: 'var(--bg-elevated)', borderRadius: 8, padding: '12px 16px',
            fontFamily: 'var(--font-mono)', fontSize: 12, overflow: 'auto',
          }}>{`# With a labeled dataset:
python -m backend.scripts.reid_validate --dataset-dir data/test_pairs/

# Without a dataset (proxy/synthetic):
python -m backend.scripts.reid_validate`}</pre>
          <div style={{ marginTop: 10, color: 'var(--text-muted)' }}>
            The report will appear here after generation.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Toolbar */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 16,
      }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Latest reid_validation_*.md report
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={() => setViewMode(m => m === 'rendered' ? 'raw' : 'rendered')}
            style={{
              padding: '6px 12px', borderRadius: 6,
              border: '1px solid var(--border)', background: 'var(--bg-elevated)',
              color: 'var(--text-primary)', cursor: 'pointer', fontSize: 12,
            }}
          >
            {viewMode === 'rendered' ? '📄 Raw' : '🖥️ Rendered'}
          </button>
          <a
            href={`${API}/reid/validation-report`}
            download="reid_validation_report.md"
            target="_blank"
            rel="noreferrer"
            style={{
              padding: '6px 12px', borderRadius: 6,
              border: '1px solid var(--border)', background: 'var(--bg-elevated)',
              color: 'var(--text-primary)', cursor: 'pointer', fontSize: 12,
              textDecoration: 'none', display: 'inline-block',
            }}
          >
            ⬇️ Download
          </a>
        </div>
      </div>

      {/* Proxy warning banner */}
      {report?.includes('PROXY VALIDATION') && (
        <div style={{
          padding: '10px 16px', borderRadius: 8, marginBottom: 16,
          background: 'var(--accent-red)15', border: '1px solid var(--accent-red)44',
          color: 'var(--accent-red)', fontSize: 13, fontWeight: 600,
        }}>
          ⚠️ PROXY VALIDATION — This report is based on synthetic data, NOT real footage. Do not use for go-live decisions.
        </div>
      )}

      {/* Report content */}
      <div className="card" style={{ maxHeight: 700, overflow: 'auto' }}>
        {viewMode === 'raw' ? (
          <pre style={{
            fontFamily: 'var(--font-mono)', fontSize: 12,
            color: 'var(--text-secondary)', whiteSpace: 'pre-wrap',
            lineHeight: 1.6,
          }}>
            {report}
          </pre>
        ) : (
          <div
            style={{ lineHeight: 1.7, fontSize: 13 }}
            className="md-report"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(report) }}
          />
        )}
      </div>
    </div>
  );
}
