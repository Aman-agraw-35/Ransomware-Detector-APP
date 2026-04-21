import "./VerdictCard.css";

function ConfidenceRing({ value, isRansomware }) {
  const radius = 40;
  const circ = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(1, value));
  const offset = circ * (1 - pct);

  return (
    <svg className="conf-ring" viewBox="0 0 100 100" aria-hidden="true">
      <circle className="conf-ring__track" cx="50" cy="50" r={radius} />
      <circle
        className={`conf-ring__fill ${isRansomware ? "conf-ring__fill--danger" : "conf-ring__fill--safe"}`}
        cx="50"
        cy="50"
        r={radius}
        strokeDasharray={circ}
        strokeDashoffset={offset}
      />
      <text className="conf-ring__text" x="50" y="50" textAnchor="middle" dominantBaseline="middle">
        {(pct * 100).toFixed(1)}%
      </text>
    </svg>
  );
}

export default function VerdictCard({ result }) {
  if (!result) return null;

  const isRansomware = result.verdict === "Ransomware";
  const confidence = result.confidence;
  const confidenceLabel = confidence >= 0.9 ? "Very High" : confidence >= 0.7 ? "High" : confidence >= 0.5 ? "Moderate" : "Low";

  return (
    <div id="verdict-card" className={`verdict-card fade-in-up ${isRansomware ? "verdict-card--danger" : "verdict-card--safe"}`}>
      <div className="verdict-card__glow" aria-hidden="true" />

      <div className="verdict-card__header">
        <div className="verdict-card__pill">
          <span className={`verdict-card__dot ${isRansomware ? "verdict-card__dot--danger" : "verdict-card__dot--safe"}`} />
          {isRansomware ? "THREAT DETECTED" : "FILE CLEAN"}
        </div>
        <span className="verdict-card__filename" title={result.filename}>{result.filename}</span>
      </div>

      <div className="verdict-card__body">
        <div className="verdict-card__label-block">
          <p className="verdict-card__verdict-text">{result.verdict}</p>
          <p className="verdict-card__verdict-sub">
            {isRansomware
              ? "This binary exhibits ransomware-like patterns."
              : "No ransomware signatures detected."}
          </p>
        </div>

        <div className="verdict-card__ring-block">
          <ConfidenceRing value={confidence} isRansomware={isRansomware} />
          <p className="verdict-card__ring-label">Confidence · <strong>{confidenceLabel}</strong></p>
        </div>
      </div>
    </div>
  );
}
