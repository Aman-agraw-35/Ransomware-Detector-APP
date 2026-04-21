import { useRef, useState } from "react";
import "./UploadZone.css";

const ALLOWED_EXT = [".exe", ".dll"];
const MAX_SIZE_MB = 50;

function formatBytes(bytes) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

function validateFile(f) {
  if (!f) return null;
  const ext = "." + f.name.split(".").pop().toLowerCase();
  if (!ALLOWED_EXT.includes(ext)) return `Only ${ALLOWED_EXT.join(", ")} files are supported.`;
  if (f.size > MAX_SIZE_MB * 1024 * 1024) return `File too large. Max size is ${MAX_SIZE_MB} MB.`;
  return null;
}

export default function UploadZone({ file, onFile, onValidationError }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function handleFiles(fileList) {
    const f = fileList[0];
    if (!f) return;
    const err = validateFile(f);
    if (err) {
      onValidationError(err);
      return;
    }
    onValidationError("");
    onFile(f);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  }

  function handleChange(e) {
    handleFiles(e.target.files);
    e.target.value = "";
  }

  return (
    <div
      id="upload-zone"
      className={`upload-zone ${dragging ? "upload-zone--drag" : ""} ${file ? "upload-zone--has-file" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      role="button"
      tabIndex={0}
      aria-label="Upload executable file"
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".exe,.dll"
        onChange={handleChange}
        aria-hidden="true"
        tabIndex={-1}
      />

      <div className="upload-zone__corner upload-zone__corner--tl" />
      <div className="upload-zone__corner upload-zone__corner--tr" />
      <div className="upload-zone__corner upload-zone__corner--bl" />
      <div className="upload-zone__corner upload-zone__corner--br" />

      {!file ? (
        <div className="upload-zone__idle">
          <div className="upload-zone__icon">
            <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
              <rect x="8" y="4" width="24" height="30" rx="3" fill="none" stroke="currentColor" strokeWidth="2.2"/>
              <path d="M32 4l8 8v28a2 2 0 01-2 2H10a2 2 0 01-2-2" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
              <path d="M32 4v8h8" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M24 22v10M20 28l4 4 4-4" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <p className="upload-zone__title">
            {dragging ? "Drop to scan" : "Drop your executable here"}
          </p>
          <p className="upload-zone__meta">
            <span className="upload-zone__badge">.exe</span>
            <span className="upload-zone__badge">.dll</span>
            <span>— Max {MAX_SIZE_MB} MB</span>
          </p>
          <p className="upload-zone__cta">or click to browse</p>
        </div>
      ) : (
        <div className="upload-zone__file-info">
          <div className="upload-zone__file-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round"/>
              <path d="M14 2v6h6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
              <path d="M9 12h6M9 16h4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
            </svg>
          </div>
          <div className="upload-zone__file-details">
            <span className="upload-zone__file-name">{file.name}</span>
            <span className="upload-zone__file-size">{formatBytes(file.size)}</span>
          </div>
          <button
            type="button"
            className="upload-zone__clear-btn"
            onClick={(e) => { e.stopPropagation(); onFile(null); onValidationError(""); }}
            aria-label="Remove file"
          >
            <svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
              <path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}
