from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pefile


# ── PE parsing helpers ────────────────────────────────────────────────────────

def _safe_stat(values: List[float], stat: str) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float({"mean": np.mean, "std": np.std,
                  "max": np.max, "min": np.min}[stat](arr))


def _resource_size(pe: pefile.PE) -> int:
    if not hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
        return 0
    total = 0
    stack = [pe.DIRECTORY_ENTRY_RESOURCE]
    while stack:
        node = stack.pop()
        for entry in getattr(node, "entries", []):
            if hasattr(entry, "directory"):
                stack.append(entry.directory)
            elif hasattr(entry, "data") and hasattr(entry.data, "struct"):
                total += getattr(entry.data.struct, "Size", 0)
    return total


def _resource_count(pe: pefile.PE) -> int:
    if not hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
        return 0
    count = 0
    stack = [pe.DIRECTORY_ENTRY_RESOURCE]
    while stack:
        node = stack.pop()
        for entry in getattr(node, "entries", []):
            count += 1
            if hasattr(entry, "directory"):
                stack.append(entry.directory)
    return count


def _imports_count(pe: pefile.PE) -> Tuple[int, int]:
    if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        return 0, 0
    dll_count    = len(pe.DIRECTORY_ENTRY_IMPORT)
    import_count = sum(len(getattr(e, "imports", [])) for e in pe.DIRECTORY_ENTRY_IMPORT)
    return import_count, dll_count


def _exports_count(pe: pefile.PE) -> int:
    if not hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        return 0
    return len(getattr(pe.DIRECTORY_ENTRY_EXPORT, "symbols", []))


def _debug_info(pe: pefile.PE) -> Tuple[int, int]:
    if not hasattr(pe, "DIRECTORY_ENTRY_DEBUG"):
        return 0, 0
    for dbg in pe.DIRECTORY_ENTRY_DEBUG:
        s = getattr(dbg, "struct", None)
        if s:
            return int(getattr(s, "SizeOfData", 0)), int(getattr(s, "AddressOfRawData", 0))
    return 0, 0


def _iat_rva(pe: pefile.PE) -> int:
    try:
        return int(pe.OPTIONAL_HEADER.DATA_DIRECTORY[12].VirtualAddress)
    except Exception:
        return 0


# ── DLL Characteristics bit flags ────────────────────────────────────────────
_DLLCHAR_BITS = {
    "dll_high_entropy_va": 0x0020,
    "dll_dynamic_base":    0x0040,   # ASLR
    "dll_nx_compat":       0x0100,   # DEP / NX
    "dll_no_seh":          0x0400,
    "dll_guard_cf":        0x4000,   # Control Flow Guard
}


def extract_base_features(file_bytes: bytes) -> Dict[str, float]:
    """
    Parse a PE binary and return features for BOTH inference paths:
      - Engineered features (v2 model): is_64bit, dll_* flags, presence flags, etc.
      - Raw Kaggle-column features (legacy / heuristic): exact column names from dataset.
      - Lowercase legacy features: section_entropy_mean, imports_count, etc.
    """
    try:
        pe = pefile.PE(data=file_bytes, fast_load=False)
        pe.parse_data_directories()
    except Exception as exc:
        raise ValueError("Invalid or unsupported PE executable.") from exc

    optional    = pe.OPTIONAL_HEADER
    file_header = pe.FILE_HEADER

    section_entropies = [float(s.get_entropy())     for s in pe.sections]
    section_raw_sizes = [float(s.SizeOfRawData)      for s in pe.sections]
    section_virt_sizes = [float(s.Misc_VirtualSize)  for s in pe.sections]

    imports_count, import_dll_count = _imports_count(pe)
    debug_size, debug_rva           = _debug_info(pe)
    iat_rva_val                     = _iat_rva(pe)
    raw_dll_char                    = int(getattr(optional, "DllCharacteristics", 0))

    export_dir  = pe.OPTIONAL_HEADER.DATA_DIRECTORY[0]
    export_rva  = int(export_dir.VirtualAddress)
    export_size = int(export_dir.Size)
    res_size    = _resource_size(pe)
    machine     = int(file_header.Machine)

    # ── Stack overflow cap matching training (99th pct ~= 4MB for this dataset)
    stack_reserve = min(float(getattr(optional, "SizeOfStackReserve", 0)), 4_194_304.0)
    resource_size_capped = min(float(res_size), 10_000_000.0)

    feats: Dict[str, float] = {}

    # ── Engineered features (v2 model, matching train_model.py) ──────────────
    feats["is_64bit"]           = float(machine == 34404)
    for fname, bit in _DLLCHAR_BITS.items():
        feats[fname]            = float((raw_dll_char & bit) != 0)
    feats["has_exports"]        = float(export_size > 0)
    feats["has_debug_info"]     = float(debug_size  > 0)
    feats["has_iat"]            = float(iat_rva_val > 0)
    feats["NumberOfSections"]   = float(file_header.NumberOfSections)
    feats["MajorLinkerVersion"] = float(getattr(optional, "MajorLinkerVersion", 0))
    feats["MinorLinkerVersion"] = float(getattr(optional, "MinorLinkerVersion", 0))
    feats["MajorImageVersion"]  = float(getattr(optional, "MajorImageVersion", 0))
    feats["MajorOSVersion"]     = float(getattr(optional, "MajorOperatingSystemVersion", 0))
    feats["ResourceSize"]       = resource_size_capped
    feats["SizeOfStackReserve"] = stack_reserve
    feats["BitcoinAddresses"]   = 0.0   # can't reliably detect via static parsing

    # ── Raw Kaggle column names (v1 / legacy fallback) ────────────────────────
    feats["Machine"]            = float(machine)
    feats["DebugSize"]          = float(debug_size)
    feats["DebugRVA"]           = float(debug_rva)
    feats["ExportRVA"]          = float(export_rva)
    feats["ExportSize"]         = float(export_size)
    feats["IatVRA"]             = float(iat_rva_val)
    feats["DllCharacteristics"] = float(raw_dll_char)

    # ── Legacy lowercase features (heuristic classifier) ─────────────────────
    feats["file_size"]               = float(len(file_bytes))
    feats["section_count"]           = float(len(pe.sections))
    feats["section_entropy_mean"]    = _safe_stat(section_entropies, "mean")
    feats["section_entropy_std"]     = _safe_stat(section_entropies, "std")
    feats["section_entropy_max"]     = _safe_stat(section_entropies, "max")
    feats["section_entropy_min"]     = _safe_stat(section_entropies, "min")
    feats["section_raw_size_mean"]   = _safe_stat(section_raw_sizes, "mean")
    feats["section_virtual_size_mean"] = _safe_stat(section_virt_sizes, "mean")
    feats["imports_count"]           = float(imports_count)
    feats["import_dll_count"]        = float(import_dll_count)
    feats["exports_count"]           = float(_exports_count(pe))
    feats["resources_count"]         = float(_resource_count(pe))
    feats["machine"]                 = float(machine)
    feats["characteristics"]         = float(file_header.Characteristics)
    feats["numberofsections"]        = float(file_header.NumberOfSections)
    feats["timestamp"]               = float(file_header.TimeDateStamp)
    feats["sizeofoptionalheader"]    = float(file_header.SizeOfOptionalHeader)
    feats["entrypoint"]              = float(optional.AddressOfEntryPoint)
    feats["baseofcode"]              = float(getattr(optional, "BaseOfCode", 0.0))
    feats["imagebase"]               = float(optional.ImageBase)
    feats["sectionalignment"]        = float(optional.SectionAlignment)
    feats["filealignment"]           = float(optional.FileAlignment)
    feats["sizeofimage"]             = float(optional.SizeOfImage)
    feats["sizeofheaders"]           = float(optional.SizeOfHeaders)
    feats["checksum"]                = float(optional.CheckSum)
    feats["subsystem"]               = float(optional.Subsystem)
    feats["dllcharacteristics"]      = float(raw_dll_char)
    feats["sizeofstackreserve"]      = float(getattr(optional, "SizeOfStackReserve", 0))
    feats["sizeofstackcommit"]       = float(getattr(optional, "SizeOfStackCommit", 0))
    feats["sizeofheapreserve"]       = float(getattr(optional, "SizeOfHeapReserve", 0))
    feats["sizeofheapcommit"]        = float(getattr(optional, "SizeOfHeapCommit", 0))
    feats["loaderflags"]             = float(optional.LoaderFlags)
    feats["numberofrvaandsizes"]     = float(optional.NumberOfRvaAndSizes)
    feats["sizeofcode"]              = float(optional.SizeOfCode)
    feats["sizeofinitializeddata"]   = float(optional.SizeOfInitializedData)
    feats["sizeofuninitializeddata"] = float(optional.SizeOfUninitializedData)

    return feats


def resolve_feature_value(feature_name: str, base_features: Dict[str, float]) -> float:
    """Exact-match first, then case-insensitive fallback."""
    if feature_name in base_features:
        return base_features[feature_name]
    lower = feature_name.lower()
    for k, v in base_features.items():
        if k.lower() == lower:
            return v
    return 0.0


def extract_feature_vector(file_bytes: bytes, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    base_features = extract_base_features(file_bytes)
    values: List[float] = []
    missing: List[str]  = []
    for name in feature_names:
        value = resolve_feature_value(name, base_features)
        values.append(float(value))
        if value == 0.0:
            missing.append(name)
    return np.asarray(values, dtype=np.float32).reshape(1, -1), missing
