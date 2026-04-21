"""Full ensemble test — BiLSTM + RandomForest, using the exact same pipeline as ml_service."""
import json, sys, os
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import tensorflow as tf
from backend.app.feature_extractor import extract_base_features, resolve_feature_value
from backend.app.ml_service import _build_attention_layer

ARTIFACT_DIR = "artifacts_phase2"

print("Loading artifacts...")
with open(f"{ARTIFACT_DIR}/feature_names.json") as f:
    feature_names = json.load(f)
with open(f"{ARTIFACT_DIR}/group_config.json") as f:
    grp_cfg = json.load(f)
with open(f"{ARTIFACT_DIR}/threshold_config.json") as f:
    thresh_cfg = json.load(f)

scaler       = joblib.load(f"{ARTIFACT_DIR}/scaler.pkl")
rf_model     = joblib.load(f"{ARTIFACT_DIR}/xgb_model.pkl")
group_names  = grp_cfg["group_names"]
groups       = {k: [int(i) for i in v] for k, v in grp_cfg["groups"].items()}
stats        = grp_cfg["stats"]
threshold    = thresh_cfg["decision_threshold"]
weights      = thresh_cfg.get("ensemble_weights", {"bilstm": 0.4, "random_forest": 0.6})

print(f"Features ({len(feature_names)}): {feature_names}")
print(f"Threshold: {threshold}  Weights: {weights}")
print(f"RF type: {type(rf_model).__name__}")

def build_sequence(X_mat):
    n = X_mat.shape[0]
    seq = np.zeros((n, len(group_names), len(stats)), dtype=np.float32)
    for g_idx, g_name in enumerate(group_names):
        idxs = groups[g_name]
        vals = X_mat[:, idxs]
        for s_idx, stat in enumerate(stats):
            if stat == "mean": seq[:, g_idx, s_idx] = np.mean(vals, axis=1)
            elif stat == "std": seq[:, g_idx, s_idx] = np.std(vals, axis=1)
            elif stat == "min": seq[:, g_idx, s_idx] = np.min(vals, axis=1)
            elif stat == "max": seq[:, g_idx, s_idx] = np.max(vals, axis=1)
    return seq

AttentionLayer = _build_attention_layer(tf, tf.keras.layers.Dense)
bilstm = tf.keras.models.load_model(
    f"{ARTIFACT_DIR}/bilstm_attention_with_weights.keras",
    custom_objects={"AttentionLayer": AttentionLayer},
    compile=False,
)
print("BiLSTM loaded.\n")

files = {
    "parsec-windows.exe":  r"C:\Users\asus laptop\Downloads\parsec-windows.exe",
    "notepad.exe":         r"C:\Windows\System32\notepad.exe",
    "calc.exe":            r"C:\Windows\System32\calc.exe",
    "cmd.exe":             r"C:\Windows\System32\cmd.exe",
    "powershell.exe":      r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
}

bilstm_w = weights.get("bilstm", 0.4)
rf_w = weights.get("random_forest", weights.get("xgboost", 0.6))

print(f"{'File':<35} {'BiLSTM':>8} {'RF':>8} {'Ensemble':>10} {'Verdict':<12} {'Status'}")
print("-" * 90)
for name, path in files.items():
    if not os.path.exists(path):
        print(f"{name:<35} NOT FOUND")
        continue
    with open(path, "rb") as f:
        data = f.read()
    try:
        base = extract_base_features(data)
        vals = [resolve_feature_value(fn, base) for fn in feature_names]
        X = np.array([vals], dtype=np.float32)
        X_sc = scaler.transform(X)
        seq = build_sequence(X_sc)

        pred, _ = bilstm.predict(seq, verbose=0)
        bilstm_prob = float(pred[0, 0])
        rf_prob     = float(rf_model.predict_proba(X_sc)[0, 1])
        ensemble    = bilstm_w * bilstm_prob + rf_w * rf_prob
        verdict     = "Ransomware" if ensemble >= threshold else "Benign"
        status      = "OK" if verdict == "Benign" else "FALSE POSITIVE"
        print(f"{name:<35} {bilstm_prob:>8.4f} {rf_prob:>8.4f} {ensemble:>10.4f} {verdict:<12} {status}")
    except Exception as e:
        import traceback
        print(f"{name}: ERROR {e}")
        traceback.print_exc()
