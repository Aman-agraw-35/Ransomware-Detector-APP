import { useMemo, useState } from "react";
import "./AttentionHeatmap.css";

function heatColor(weight, maxWeight) {
  const norm = maxWeight > 0 ? Math.max(0, Math.min(1, weight / maxWeight)) : 0;
  // Low weight -> cool blue, high weight -> hot orange/red
  if (norm < 0.33) {
    const t = norm / 0.33;
    return { bg: `rgba(59, 130, 246, ${0.1 + t * 0.25})`, text: "#7dd3fc", border: `rgba(59,130,246,${0.15 + t * 0.25})` };
  }
  if (norm < 0.66) {
    const t = (norm - 0.33) / 0.33;
    return { bg: `rgba(245, 158, 11, ${0.1 + t * 0.3})`, text: "#fbbf24", border: `rgba(245,158,11,${0.2 + t * 0.3})` };
  }
  const t = (norm - 0.66) / 0.34;
  return { bg: `rgba(239, 68, 68, ${0.15 + t * 0.35})`, text: "#f87171", border: `rgba(239,68,68,${0.25 + t * 0.35})` };
}

function normalizeLabel(label) {
  return label.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

export default function AttentionHeatmap({ attention }) {
  const [tooltip, setTooltip] = useState(null);

  const maxWeight = useMemo(() => {
    if (!attention?.weights?.length) return 1;
    return Math.max(...attention.weights);
  }, [attention]);

  if (!attention?.labels?.length) {
    return (
      <div id="attention-heatmap" className="attn-panel">
        <div className="panel-header">
          <span className="panel-header__icon" aria-hidden="true">
            <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="2" y="4" width="16" height="3" rx="1.5" fill="currentColor" opacity="0.9"/>
              <rect x="2" y="9" width="12" height="3" rx="1.5" fill="currentColor" opacity="0.6"/>
              <rect x="2" y="14" width="8" height="3" rx="1.5" fill="currentColor" opacity="0.3"/>
            </svg>
          </span>
          <h2 className="panel-header__title">Attention Heatmap</h2>
        </div>
        <p className="panel-empty">Run an analysis to view attention weights.</p>
      </div>
    );
  }

  return (
    <div id="attention-heatmap" className="attn-panel">
      <div className="panel-header">
        <span className="panel-header__icon" aria-hidden="true">
          <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="2" y="4" width="16" height="3" rx="1.5" fill="currentColor" opacity="0.9"/>
            <rect x="2" y="9" width="12" height="3" rx="1.5" fill="currentColor" opacity="0.6"/>
            <rect x="2" y="14" width="8" height="3" rx="1.5" fill="currentColor" opacity="0.3"/>
          </svg>
        </span>
        <h2 className="panel-header__title">Attention Heatmap</h2>
        <span className="panel-header__badge">{attention.labels.length} groups</span>
      </div>

      <div className="attn-legend">
        <span className="attn-legend__label">Low</span>
        <div className="attn-legend__bar" aria-hidden="true" />
        <span className="attn-legend__label">High</span>
      </div>

      <div className="attn-grid" role="list">
        {attention.labels.map((label, i) => {
          const weight = attention.weights[i] ?? 0;
          const colors = heatColor(weight, maxWeight);
          const normPct = maxWeight > 0 ? ((weight / maxWeight) * 100).toFixed(1) : "0.0";
          return (
            <div
              key={`${label}-${i}`}
              className="attn-cell fade-in-up"
              role="listitem"
              style={{
                backgroundColor: colors.bg,
                borderColor: colors.border,
                animationDelay: `${i * 0.03}s`,
                opacity: 0,
              }}
              onMouseEnter={() => setTooltip({ label, weight, normPct, index: i })}
              onMouseLeave={() => setTooltip(null)}
              title={`${normalizeLabel(label)}: ${weight.toFixed(4)}`}
            >
              <span className="attn-cell__label" style={{ color: colors.text }}>
                {normalizeLabel(label)}
              </span>
              <strong className="attn-cell__value" style={{ color: colors.text }}>
                {weight.toFixed(3)}
              </strong>
              <div
                className="attn-cell__bar"
                style={{ width: `${normPct}%`, backgroundColor: colors.text }}
                aria-hidden="true"
              />
            </div>
          );
        })}
      </div>

      {tooltip && (
        <div className="attn-tooltip" role="tooltip">
          <strong>{normalizeLabel(tooltip.label)}</strong>
          <span>Weight: <code>{tooltip.weight.toFixed(6)}</code></span>
          <span>Relative: <code>{tooltip.normPct}%</code> of max</span>
        </div>
      )}
    </div>
  );
}
