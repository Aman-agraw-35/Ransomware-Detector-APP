import { useMemo } from "react";
import "./ShapBarChart.css";

function formatFeatureName(name) {
  return name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function toFixed4(n) {
  return n >= 0 ? `+${n.toFixed(4)}` : n.toFixed(4);
}

export default function ShapBarChart({ shap }) {
  const maxAbs = useMemo(() => {
    if (!shap?.top_features?.length) return 1;
    return Math.max(...shap.top_features.map(f => Math.abs(f.value))) || 1;
  }, [shap]);

  if (!shap?.top_features?.length) {
    return (
      <div id="shap-chart" className="shap-panel">
        <div className="panel-header">
          <span className="panel-header__icon" aria-hidden="true">
            <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M3 17V10M7 17V7M11 17V12M15 17V4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </span>
          <h2 className="panel-header__title">SHAP Feature Explanations</h2>
        </div>
        <p className="panel-empty">Run an analysis to view SHAP contributions.</p>
      </div>
    );
  }

  return (
    <div id="shap-chart" className="shap-panel">
      <div className="panel-header">
        <span className="panel-header__icon" aria-hidden="true">
          <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 17V10M7 17V7M11 17V12M15 17V4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </span>
        <h2 className="panel-header__title">SHAP Feature Explanations</h2>
        <span className="panel-header__badge">Top {shap.top_features.length}</span>
      </div>

      <div className="shap-base-row">
        <span className="shap-base-label">Base value</span>
        <code className="shap-base-value">{shap.base_value.toFixed(4)}</code>
      </div>

      <div className="shap-legend">
        <span className="shap-legend__item shap-legend__item--pos">
          <span className="shap-legend__swatch" />
          Pushes toward Ransomware
        </span>
        <span className="shap-legend__item shap-legend__item--neg">
          <span className="shap-legend__swatch" />
          Pushes toward Benign
        </span>
      </div>

      <ol className="shap-list" aria-label="SHAP top features">
        {shap.top_features.map((feature, i) => {
          const isPos = feature.value >= 0;
          const widthPct = (Math.abs(feature.value) / maxAbs) * 100;
          return (
            <li key={feature.name} className="shap-row fade-in-up" style={{ animationDelay: `${i * 0.04}s`, opacity: 0 }}>
              <div className="shap-row__meta">
                <span className="shap-row__rank">#{i + 1}</span>
                <span className="shap-row__name" title={feature.name}>
                  {formatFeatureName(feature.name)}
                </span>
                <code className={`shap-row__value ${isPos ? "shap-row__value--pos" : "shap-row__value--neg"}`}>
                  {toFixed4(feature.value)}
                </code>
              </div>
              <div className="shap-row__bar-track">
                <div
                  className={`shap-row__bar ${isPos ? "shap-row__bar--pos" : "shap-row__bar--neg"}`}
                  style={{ "--target-width": `${widthPct}%`, width: `${widthPct}%` }}
                  role="progressbar"
                  aria-valuenow={Math.abs(feature.value)}
                  aria-valuemax={maxAbs}
                  aria-label={`${feature.name}: ${feature.value.toFixed(4)}`}
                />
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
