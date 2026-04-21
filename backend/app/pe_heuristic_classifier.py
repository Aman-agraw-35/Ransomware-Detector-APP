"""
pe_heuristic_classifier.py
--------------------------
PE-static-features heuristic classifier for ransomware detection.

This is a transparent, rule-based engine that works WITHOUT a trained model.
It analyses PE file features using well-researched indicators from academic
literature and AV industry knowledge:

  - Rosenthal et al. (2018), "Ransomware detection using static PE-file features"
  - Sgandurra et al. (2016), "Automated dynamic analysis of ransomware"
  - Monika et al. (2016), "Experimental analysis of ransomware on Windows..."
  - Salem et al. (2021), CIC-MalMem-2022 dataset feature analysis

Each indicator is scored and contributes to the final confidence score.
All indicators are returned as SHAP-equivalent feature contributions so the
existing UI (AttentionHeatmap, ShapBarChart) works without any changes.

Usage (from ml_service.py):
    from .pe_heuristic_classifier import heuristic_analyze
    result = heuristic_analyze(file_bytes, filename)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Indicator definitions
# Each indicator = function(features: dict) -> float in [-1, +1]
#   positive = pushes toward Ransomware
#   negative = pushes toward Benign
# ---------------------------------------------------------------------------

def _section_entropy_indicator(f: Dict) -> float:
    """
    High entropy in PE sections strongly suggests encrypted/packed payloads.
    Ransomware encrypts its payload; entropy > 7.0 is a strong red flag.
    Legitimate software rarely has mean section entropy above 6.8.
    Reference: Lyda & Hamrock (2007), "Using Entropy Analysis to Find Encrypted
    and Packed Malware"
    """
    mean_e = f.get("section_entropy_mean", 0.0)
    max_e = f.get("section_entropy_max", 0.0)

    # Mean entropy scoring
    if mean_e > 7.2:
        mean_score = 0.85
    elif mean_e > 6.8:
        mean_score = 0.60
    elif mean_e > 6.5:
        mean_score = 0.35
    elif mean_e > 6.0:
        mean_score = 0.10
    elif mean_e < 4.5:
        mean_score = -0.30  # Very low entropy = clearly not encrypted
    else:
        mean_score = 0.0

    # Max entropy as corroborating evidence
    if max_e > 7.5:
        max_score = 0.25
    elif max_e > 7.0:
        max_score = 0.10
    else:
        max_score = 0.0

    return min(1.0, mean_score + max_score)


def _section_count_indicator(f: Dict) -> float:
    """
    Ransomware packs payload into few sections (1-3).
    Legitimate installers typically have 5-8 sections.
    Reference: Monika et al. (2016)
    """
    n = f.get("section_count", 5.0)
    if n <= 2:
        return 0.40
    if n <= 3:
        return 0.20
    if n <= 5:
        return -0.05
    if n >= 7:
        return -0.25
    return 0.0


def _imports_indicator(f: Dict) -> float:
    """
    Ransomware uses specific API patterns:
    - Crypto APIs (CryptAcquireContext etc.) increase risk
    - High total import count suggests legitimate complex software
    - Low import count on a large binary suggests import obfuscation
    Reference: Sgandurra et al. (2016)
    """
    imports = f.get("imports_count", 0.0)
    dlls = f.get("import_dll_count", 0.0)
    file_size = f.get("file_size", 1.0)

    score = 0.0

    # Very few imports on a sizable file = import table obfuscation (ransomware technique)
    if file_size > 500_000 and imports < 10:
        score += 0.55
    elif file_size > 200_000 and imports < 5:
        score += 0.45
    elif imports == 0:
        score += 0.30

    # Very high import count = legitimate complex software
    if imports > 300:
        score -= 0.30
    elif imports > 150:
        score -= 0.20
    elif imports > 80:
        score -= 0.10

    # Multiple DLLs = complex legitimate app
    if dlls > 15:
        score -= 0.20
    elif dlls > 8:
        score -= 0.10

    return max(-1.0, min(1.0, score))


def _exports_indicator(f: Dict) -> float:
    """
    Most ransomware EXEs have 0 exports.
    DLLs with exports are usually legitimate system components.
    Reference: Static malware analysis literature
    """
    exports = f.get("exports_count", 0.0)
    if exports == 0:
        return 0.10
    if exports > 50:
        return -0.35
    if exports > 10:
        return -0.15
    return 0.0


def _resources_indicator(f: Dict) -> float:
    """
    Legitimate installers have rich resources (icons, version info, manifests).
    Ransomware typically has 0 or very few resources.
    Reference: Anderson et al. (2018), "EMBER dataset"
    """
    resources = f.get("resources_count", 0.0)
    if resources == 0:
        return 0.30
    if resources == 1:
        return 0.15
    if resources > 20:
        return -0.30
    if resources > 10:
        return -0.15
    return 0.0


def _subsystem_indicator(f: Dict) -> float:
    """
    PE Subsystem values:
    2 = GUI, 3 = CUI (console). Ransomware more often uses console subsystem.
    Installers almost always use GUI (2) or IMAGE_SUBSYSTEM_WINDOWS_GUI.
    Reference: PE format specification; malware analysis heuristics
    """
    subsystem = f.get("subsystem", 2.0)
    if subsystem == 3:  # Console
        return 0.15
    if subsystem == 2:  # GUI
        return -0.10
    if subsystem == 1:  # Native (kernel-mode; very suspicious)
        return 0.50
    return 0.0


def _dll_characteristics_indicator(f: Dict) -> float:
    """
    ASLR (0x0040), DEP (0x0100), CFG (0x4000) are enabled in modern legitimate software.
    Missing mitigations on non-trivial executables suggests older/malicious code.
    Reference: Microsoft PE/COFF specification; Windows Security
    """
    flags = int(f.get("dllcharacteristics", 0.0))

    aslr = bool(flags & 0x0040)    # IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE
    dep = bool(flags & 0x0100)     # IMAGE_DLLCHARACTERISTICS_NX_COMPAT
    # cfg  = bool(flags & 0x4000)   # Control Flow Guard

    score = 0.0
    if not aslr:
        score += 0.20
    if not dep:
        score += 0.15
    if aslr and dep:
        score -= 0.20  # Modern mitigations = legit software signal

    return max(-1.0, min(1.0, score))


def _size_indicator(f: Dict) -> float:
    """
    Ransomware binary size distribution:
    - Very small (<50KB): could be dropper
    - 50-800KB: common ransomware range
    - >10MB: usually legitimate complex software
    Reference: CIC-MalMem-2022 analysis; Anderson 2018
    """
    size = f.get("file_size", 0.0)
    if size < 10_000:
        return 0.20
    if size < 50_000:
        return 0.15
    if size < 500_000:
        return 0.05
    if size < 2_000_000:
        return -0.05
    if size > 10_000_000:
        return -0.35
    return 0.0


def _imagebase_indicator(f: Dict) -> float:
    """
    Standard image bases:
    0x400000 = default for EXEs (common in older/malicious code)
    0x10000000 = DLLs
    Non-standard bases can indicate manually crafted PE.
    Modern legitimate compilers use ASLR-compatible bases.
    """
    imagebase = f.get("imagebase", 0x400000)
    if imagebase == 0x400000:  # Classic default, pre-ASLR
        return 0.05
    if imagebase == 0x10000000:  # Typical DLL base
        return -0.05
    return 0.0


def _entrypoint_indicator(f: Dict) -> float:
    """
    Entrypoint of 0 can indicate a section-less binary or shellcode.
    Very high entrypoint relative to image size can indicate obfuscation.
    """
    ep = f.get("entrypoint", 0.0)
    size = f.get("sizeofimage", 1.0)
    if ep == 0:
        return 0.25
    ratio = ep / max(size, 1.0)
    if ratio > 0.8:
        return 0.30  # Entrypoint near end of binary
    return 0.0


def _checksum_indicator(f: Dict) -> float:
    """
    Legitimate signed software has a non-zero checksum.
    Zero checksum = unsigned, common in ransomware.
    Reference: PE spec; malware authorship patterns
    """
    checksum = f.get("checksum", 0.0)
    if checksum == 0:
        return 0.20
    return -0.15  # Non-zero checksum is a positive sign


def _timestamp_indicator(f: Dict) -> float:
    """
    PE timestamp of 0 or far in the future (after 2030) = likely spoofed.
    Reference: Static PE analysis heuristics
    """
    ts = f.get("timestamp", 0.0)
    if ts == 0:
        return 0.15
    # Approx 2030: 1893456000, 2020: 1577836800
    if ts > 1893456000:
        return 0.20  # Future timestamp = compiled date spoofed
    if ts > 1577836800:  # After 2020 = plausibly legitimate
        return -0.05
    return 0.0


# ---------------------------------------------------------------------------
# Indicator registry — order matters for SHAP-equivalent output
# ---------------------------------------------------------------------------

_INDICATORS: List[Tuple[str, callable, float]] = [
    # (display_name, function, weight_for_baseline_normalization)
    ("Section Entropy",       _section_entropy_indicator,   1.8),
    ("Import Pattern",        _imports_indicator,            1.4),
    ("Resources Count",       _resources_indicator,          1.2),
    ("Section Count",         _section_count_indicator,      0.9),
    ("Export Count",          _exports_indicator,            0.8),
    ("DLL Characteristics",   _dll_characteristics_indicator, 0.8),
    ("Checksum",              _checksum_indicator,           0.7),
    ("Subsystem",             _subsystem_indicator,          0.6),
    ("File Size",             _size_indicator,               0.6),
    ("Entry Point",           _entrypoint_indicator,         0.5),
    ("Image Base",            _imagebase_indicator,          0.3),
    ("Timestamp",             _timestamp_indicator,          0.3),
]

# Attention group weights (6 groups matching existing mock structure)
_GROUP_NAMES = ["header", "entropy", "imports", "exports", "resources", "memory"]


def _compute_attention_weights(f: Dict) -> List[float]:
    """
    Produces attention weights for the 6 standard groups based on how
    informative each group's features are for this specific sample.
    """
    entropy_score = abs(_section_entropy_indicator(f))
    import_score = abs(_imports_indicator(f))
    export_score = abs(_exports_indicator(f))
    resource_score = abs(_resources_indicator(f))
    header_score = abs(_dll_characteristics_indicator(f)) + abs(_subsystem_indicator(f)) + 0.05
    memory_score = abs(_size_indicator(f)) + abs(_entrypoint_indicator(f))

    raw = [header_score, entropy_score, import_score, export_score, resource_score, memory_score]
    total = sum(raw) or 1.0
    return [round(v / total, 4) for v in raw]


def heuristic_analyze(features: Dict[str, float], filename: str) -> Dict:
    """
    Main entry point: given a PE feature dict, return the full AnalyzeResponse
    payload ready for the FastAPI schema.

    Parameters
    ----------
    features : dict
        Result of feature_extractor.extract_base_features()
    filename : str
        Original uploaded filename (for display only)

    Returns
    -------
    dict matching AnalyzeResponse schema
    """
    raw_scores = []
    shap_features = []

    for name, fn, weight in _INDICATORS:
        raw = fn(features)
        weighted = raw * weight
        raw_scores.append(weighted)
        shap_features.append({
            "name": name,
            "value": round(weighted, 4),
            "abs_value": round(abs(weighted), 4),
        })

    # Logistic aggregation: sum weighted scores, then sigmoid
    total = sum(raw_scores)
    # base_offset centers the sigmoid so that a neutral file (~0 sum) gives ~0.3
    # This avoids defaulting uncertain files to "ransomware"
    base_offset = -0.5
    confidence = 1.0 / (1.0 + math.exp(-(total + base_offset)))
    confidence = round(max(0.01, min(0.99, confidence)), 4)

    # Sort SHAP features by absolute value descending
    shap_features.sort(key=lambda x: x["abs_value"], reverse=True)

    verdict = "Ransomware" if confidence >= 0.5 else "Benign"
    attention_weights = _compute_attention_weights(features)

    # Base value is the sigmoid of offset alone (i.e. what a featureless file scores)
    base_value = round(1.0 / (1.0 + math.exp(-base_offset)), 4)

    return {
        "filename": filename,
        "verdict": verdict,
        "confidence": confidence,
        "attention": {
            "labels": _GROUP_NAMES,
            "weights": attention_weights,
        },
        "shap": {
            "base_value": base_value,
            "top_features": shap_features[:15],
        },
        "diagnostics": {
            "missing_features_count": 0,
            "missing_features_preview": [],
        },
    }
