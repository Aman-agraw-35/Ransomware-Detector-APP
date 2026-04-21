import "./LoadingOverlay.css";

const STEPS = [
  "Parsing PE headers…",
  "Extracting static features…",
  "Scaling feature vector…",
  "Running BiLSTM inference…",
  "Computing attention weights…",
  "Computing SHAP explanations…",
];

export default function LoadingOverlay({ visible, step }) {
  if (!visible) return null;

  const stepIndex = step ?? 0;

  return (
    <div id="loading-overlay" className="loading-overlay" role="status" aria-live="polite" aria-label="Analyzing file">
      <div className="loading-overlay__card">
        <div className="loading-scanner" aria-hidden="true">
          <div className="loading-scanner__ring loading-scanner__ring--outer" />
          <div className="loading-scanner__ring loading-scanner__ring--inner" />
          <div className="loading-scanner__core">
            <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M10 20h20M20 10v20" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
              <circle cx="20" cy="20" r="5" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </div>
        </div>

        <p className="loading-overlay__title">Analyzing Binary</p>
        <p className="loading-overlay__step">{STEPS[stepIndex % STEPS.length]}</p>

        <div className="loading-overlay__progress">
          {STEPS.map((_, i) => (
            <div
              key={i}
              className={`loading-overlay__dot ${i <= stepIndex ? "loading-overlay__dot--active" : ""}`}
              aria-hidden="true"
            />
          ))}
        </div>
      </div>
    </div>
  );
}
