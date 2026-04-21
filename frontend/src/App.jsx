import { useEffect, useRef, useState } from "react";
import "./App.css";

import AttentionHeatmap from "./components/AttentionHeatmap";
import DiagnosticsPanel from "./components/DiagnosticsPanel";
import LoadingOverlay from "./components/LoadingOverlay";
import ShapBarChart from "./components/ShapBarChart";
import UploadZone from "./components/UploadZone";
import VerdictCard from "./components/VerdictCard";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const SCAN_STEPS = [
  "Parsing PE headers…",
  "Extracting static features…",
  "Scaling feature vector…",
  "Running BiLSTM inference…",
  "Computing attention weights…",
  "Computing SHAP explanations…",
];

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [inferenceMode, setInferenceMode] = useState(null); // 'heuristic' | 'bilstm_model' | null
  const stepTimerRef = useRef(null);
  const resultsRef = useRef(null);

  // Poll health to know inference mode
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => setInferenceMode(d.inference_mode ?? (d.mock_mode ? "heuristic" : "bilstm_model")))
      .catch(() => setInferenceMode(null));
  }, []);

  // Progress through loading steps while scanning
  useEffect(() => {
    if (loading) {
      setLoadingStep(0);
      let step = 0;
      stepTimerRef.current = setInterval(() => {
        step = Math.min(step + 1, SCAN_STEPS.length - 1);
        setLoadingStep(step);
      }, 800);
    } else {
      clearInterval(stepTimerRef.current);
      setLoadingStep(0);
    }
    return () => clearInterval(stepTimerRef.current);
  }, [loading]);

  // Scroll to results when they arrive
  useEffect(() => {
    if (result && resultsRef.current) {
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 200);
    }
  }, [result]);

  async function onAnalyze(event) {
    event.preventDefault();
    if (!file) {
      setError("Please upload an .exe or .dll file first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Analysis failed.");
      }
      setResult(payload);
    } catch (err) {
      setError(err.message || "Unexpected error while analyzing the file.");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setFile(null);
    setResult(null);
    setError("");
  }

  return (
    <>
      <LoadingOverlay visible={loading} step={loadingStep} />

      <div className="app-shell">
        {/* ── Header ── */}
        <header className="app-header">
          <div className="app-header__logo">
            <svg className="app-header__logo-icon" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
              <circle cx="18" cy="18" r="17" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M18 8v4M18 24v4M8 18h4M24 18h4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <circle cx="18" cy="18" r="6" stroke="currentColor" strokeWidth="1.8"/>
              <circle cx="18" cy="18" r="2" fill="currentColor"/>
            </svg>
            <span className="app-header__brand">SeqDefender</span>
          </div>
          <div className="app-header__meta">
            <span className="app-header__tag">BiLSTM + Attention</span>
            <span className="app-header__tag">SHAP Explainability</span>
            {inferenceMode === "heuristic" && (
              <span className="app-header__tag app-header__tag--heuristic" title="No trained model found. Using PE heuristic classifier based on academic research.">
                <svg width="13" height="13" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" style={{flexShrink:0}}>
                  <path d="M8 2L14 13H2L8 2z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
                  <path d="M8 7v3M8 12v.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
                Heuristic Mode
              </span>
            )}
            {inferenceMode === "bilstm_model" && (
              <span className="app-header__tag app-header__tag--live" aria-label="Status: BiLSTM Model Active">
                <span className="app-header__live-dot" aria-hidden="true" />
                BiLSTM Active
              </span>
            )}
            {inferenceMode === null && (
              <span className="app-header__tag app-header__tag--live" aria-label="Status: Live">
                <span className="app-header__live-dot" aria-hidden="true" />
                API Live
              </span>
            )}
          </div>
        </header>

        {/* ── Hero ── */}
        <section className="app-hero" aria-labelledby="hero-title">
          <div className="app-hero__content">
            <p className="app-hero__eyebrow">Static Ransomware Analysis · Phase II</p>
            <h1 id="hero-title" className="app-hero__title">
              Ransomware Scan<br/>
              <span className="app-hero__title-accent">Console</span>
            </h1>
            <p className="app-hero__subtitle">
              Upload a Windows PE binary (.exe / .dll). The BiLSTM model with Attention
              extracts static features, runs inference, and produces SHAP explanations
              in one request.
            </p>
          </div>
          <div className="app-hero__decoration" aria-hidden="true">
            <div className="hero-ring hero-ring--1" />
            <div className="hero-ring hero-ring--2" />
            <div className="hero-ring hero-ring--3" />
          </div>
        </section>

        {/* ── Upload + Scan ── */}
        <section className="app-card scan-card" aria-label="File upload and scan">
          <div className="scan-card__body">
            <div className="scan-card__upload-area">
              <h2 className="scan-card__section-title">
                <span className="scan-card__step-num">1</span>
                Select Binary
              </h2>
              <UploadZone
                file={file}
                onFile={(f) => { setFile(f); setError(""); setResult(null); }}
                onValidationError={setError}
              />
            </div>

            <div className="scan-card__action-area">
              <h2 className="scan-card__section-title">
                <span className="scan-card__step-num">2</span>
                Run Analysis
              </h2>
              <form onSubmit={onAnalyze}>
                <button
                  id="scan-btn"
                  className={`scan-btn ${loading ? "scan-btn--loading" : ""}`}
                  type="submit"
                  disabled={loading || !file}
                  aria-busy={loading}
                >
                  {loading ? (
                    <>
                      <span className="scan-btn__spinner" aria-hidden="true" />
                      Scanning…
                    </>
                  ) : (
                    <>
                      <svg className="scan-btn__icon" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                        <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="1.8"/>
                        <path d="M10 6v4.5M10 13.5v.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                      Analyze File
                    </>
                  )}
                </button>
              </form>

              {error && (
                <div className="error-banner" role="alert">
                  <svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" width="15" height="15">
                    <path d="M8 3L14 13H2L8 3z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
                    <path d="M8 8v2M8 11.5v.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  {error}
                </div>
              )}

              {inferenceMode === "heuristic" && (
                <div className="heuristic-notice" role="note">
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M8 5v4M8 11v.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  <span>
                    <strong>Heuristic mode</strong> — No trained model found. Analysis uses PE static indicators
                    (entropy, imports, resources, DLL characteristics) from academic research.
                    Train the model with <code>data_file.csv</code> to enable BiLSTM inference.
                  </span>
                </div>
              )}

              <div className="scan-card__feature-list" aria-label="Analysis features">
                {inferenceMode === "heuristic"
                  ? ["PE Entropy", "Import Analysis", "Resource Heuristics", "DLL Characteristics", "50 MB Max"].map(f => (
                      <span key={f} className="scan-card__feature-chip">{f}</span>
                    ))
                  : ["PE Header Parsing", "Bi-directional LSTM", "Attention Weights", "SHAP Values", "50 MB Max"].map(f => (
                      <span key={f} className="scan-card__feature-chip">{f}</span>
                    ))
                }
              </div>
            </div>
          </div>
        </section>

        {/* ── Results ── */}
        {result && (
          <div id="results" className="results-section" ref={resultsRef} aria-label="Analysis results">
            <div className="results-section__header">
              <h2 className="results-section__title">Analysis Results</h2>
              <button id="new-scan-btn" className="new-scan-btn" type="button" onClick={handleReset}>
                New Scan
              </button>
            </div>

            {/* Verdict */}
            <VerdictCard result={result} />

            {/* Dual column: Attention + SHAP */}
            <div className="results-twin-grid">
              <div className="app-card">
                <AttentionHeatmap attention={result.attention} />
              </div>
              <div className="app-card">
                <ShapBarChart shap={result.shap} />
              </div>
            </div>

            {/* Diagnostics */}
            <DiagnosticsPanel diagnostics={result.diagnostics} filename={result.filename} />
          </div>
        )}

        {/* ── Footer ── */}
        <footer className="app-footer">
          <span>SeqDefender · Static PE Analysis</span>
          <span className="app-footer__dot" aria-hidden="true" />
          <span>BiLSTM + Attention + SHAP</span>
          <span className="app-footer__dot" aria-hidden="true" />
          <span>For research use only</span>
        </footer>
      </div>
    </>
  );
}
