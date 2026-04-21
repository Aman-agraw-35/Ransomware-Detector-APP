import "./DiagnosticsPanel.css";

export default function DiagnosticsPanel({ diagnostics, filename }) {
  if (!diagnostics) return null;

  const { missing_features_count, missing_features_preview } = diagnostics;
  const severity = missing_features_count === 0 ? "good" : missing_features_count < 10 ? "warn" : "bad";

  return (
    <div id="diagnostics-panel" className={`diag-panel diag-panel--${severity} fade-in-up fade-in-up-delay-3`}>
      <div className="diag-panel__header">
        <svg className="diag-panel__icon" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="1.8"/>
          <path d="M10 6v4.5M10 14v.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        <h3 className="diag-panel__title">Scan Diagnostics</h3>
      </div>

      <dl className="diag-panel__grid">
        <dt>File</dt>
        <dd title={filename} className="diag-panel__filename">{filename}</dd>

        <dt>Unmapped features</dt>
        <dd className={`diag-panel__count diag-panel__count--${severity}`}>{missing_features_count}</dd>

        {missing_features_preview?.length > 0 && (
          <>
            <dt>First missing</dt>
            <dd className="diag-panel__missing-list">
              {missing_features_preview.slice(0, 6).map(f => (
                <code key={f} className="diag-panel__tag">{f}</code>
              ))}
              {missing_features_preview.length > 6 && (
                <span className="diag-panel__more">+{missing_features_preview.length - 6} more</span>
              )}
            </dd>
          </>
        )}
      </dl>

      {severity === "bad" && (
        <p className="diag-panel__warn-note">
          High number of unmapped features may reduce prediction accuracy.
        </p>
      )}
    </div>
  );
}
