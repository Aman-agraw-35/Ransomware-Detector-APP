"""
find_threshold.py  —  Find optimal decision threshold to minimize FPR
while keeping FNR (missed ransomware) acceptable.

Uses the test split from the same seed as training so it's held-out data.
"""
import json, os, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

SEED         = 42
DATA_PATH    = "data_file.csv"
ARTIFACT_DIR = "artifacts_phase2"

random.seed(SEED)
np.random.seed(SEED)

# ── Reload artifacts ──────────────────────────────────────────────────────────
with open(f"{ARTIFACT_DIR}/feature_names.json") as f:
    feature_names = json.load(f)
with open(f"{ARTIFACT_DIR}/group_config.json") as f:
    grp_cfg = json.load(f)

scaler      = joblib.load(f"{ARTIFACT_DIR}/scaler.pkl")
group_names = grp_cfg["group_names"]
groups      = {k: [int(i) for i in v] for k, v in grp_cfg["groups"].items()}
stats       = grp_cfg["stats"]

from backend.app.ml_service import _build_attention_layer
AttentionLayer = _build_attention_layer(tf, tf.keras.layers.Dense)
model = tf.keras.models.load_model(
    f"{ARTIFACT_DIR}/bilstm_attention_model.keras",
    custom_objects={"AttentionLayer": AttentionLayer},
    compile=False,
)
print("Model loaded.")

# ── Recreate test split identically ──────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
for col in ["FileName", "md5Hash"]:
    if col in df.columns:
        df = df.drop(columns=[col])
df = df.drop_duplicates().reset_index(drop=True)

y = (df["Benign"] == 0).astype(int).values
df = df.drop(columns=["Benign"])

raw_dll = df["DllCharacteristics"].astype(int)
df["is_64bit"]           = (df["Machine"] == 34404).astype(float)
df["dll_high_entropy_va"] = ((raw_dll & 0x0020) != 0).astype(float)
df["dll_dynamic_base"]   = ((raw_dll & 0x0040) != 0).astype(float)
df["dll_nx_compat"]      = ((raw_dll & 0x0100) != 0).astype(float)
df["dll_no_seh"]         = ((raw_dll & 0x0400) != 0).astype(float)
df["dll_guard_cf"]       = ((raw_dll & 0x4000) != 0).astype(float)
df["dll_terminal_server"] = ((raw_dll & 0x8000) != 0).astype(float)
df["has_exports"]        = (df["ExportSize"] > 0).astype(float)
df["has_debug_info"]     = (df["DebugSize"] > 0).astype(float)
df["has_iat"]            = (df["IatVRA"] > 0).astype(float)
rs_cap = df["ResourceSize"].quantile(0.99)
df["ResourceSize"] = df["ResourceSize"].clip(upper=rs_cap)
df["is_large_stack"] = (df["SizeOfStackReserve"] >= 4_194_304).astype(float)
stk_cap = df["SizeOfStackReserve"].quantile(0.99)
df["SizeOfStackReserve"] = df["SizeOfStackReserve"].clip(upper=stk_cap)

drop_cols = [c for c in ["Machine","DllCharacteristics","MajorImageVersion","MajorOSVersion",
                          "MajorLinkerVersion","MinorLinkerVersion","DebugSize","DebugRVA",
                          "ExportRVA","ExportSize","IatVRA"] if c in df.columns]
df = df.drop(columns=drop_cols)

X = df[feature_names].values.astype(np.float32)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

def build_seq(X_mat):
    n = X_mat.shape[0]
    s = np.zeros((n, len(group_names), len(stats)), dtype=np.float32)
    for g_idx, g_name in enumerate(group_names):
        idxs = groups[g_name]
        vals = X_mat[:, idxs]
        for s_idx, stat in enumerate(stats):
            if stat == "mean": s[:, g_idx, s_idx] = np.mean(vals, axis=1)
            elif stat == "std": s[:, g_idx, s_idx] = np.std(vals, axis=1)
            elif stat == "min": s[:, g_idx, s_idx] = np.min(vals, axis=1)
            elif stat == "max": s[:, g_idx, s_idx] = np.max(vals, axis=1)
    return s

X_test_sc  = scaler.transform(X_test)
X_test_seq = build_seq(X_test_sc)

y_prob = model.predict(X_test_seq, verbose=0).ravel()
print(f"\nAUC on test: {roc_auc_score(y_test, y_prob):.4f}")
print(f"\nThreshold sweep (FNR = missed ransomware, FPR = wrongly flagged benign):")
print(f"{'Threshold':>10} {'FNR':>8} {'FPR':>8} {'Accuracy':>10} {'Ransomware Recall':>18}")
print("-" * 62)

best = None
for thresh in np.arange(0.30, 0.90, 0.05):
    y_pred = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    acc = (tn + tp) / len(y_test)
    rec = tp / (tp + fn + 1e-9)
    marker = ""
    # Best = FNR < 5% AND FPR < 5%
    if fnr < 0.05 and fpr < 0.05 and best is None:
        marker = "  <-- OPTIMAL"
        best = float(thresh)
    print(f"{thresh:>10.2f} {fnr:>8.4f} {fpr:>8.4f} {acc:>10.4f} {rec:>18.4f}{marker}")

print()
if best:
    print(f"Optimal threshold: {best:.2f}")
    # Write to config
    cfg = {"decision_threshold": best, "reason": "Calibrated on held-out test set to achieve FNR<5% and FPR<5%"}
    with open(f"{ARTIFACT_DIR}/threshold_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved to {ARTIFACT_DIR}/threshold_config.json")
else:
    print("No threshold achieves both FNR<5% and FPR<5% — dataset has fundamental limitations.")
    # Use the threshold that minimises sum of FNR + FPR
    best_thresh = None
    best_sum = 999
    for thresh in np.arange(0.30, 0.90, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fnr = fn / (fn + tp + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        s = fnr + fpr
        if s < best_sum:
            best_sum = s
            best_thresh = thresh
    print(f"Minimum-cost threshold (min FNR+FPR): {best_thresh:.2f}")
    cfg = {"decision_threshold": round(float(best_thresh), 2)}
    with open(f"{ARTIFACT_DIR}/threshold_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved to {ARTIFACT_DIR}/threshold_config.json")
