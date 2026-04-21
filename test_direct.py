"""Direct inference test — bypasses HTTP, loads model and runs prediction directly."""
import json, sys, os
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import tensorflow as tf
from backend.app.feature_extractor import extract_base_features, resolve_feature_value

ARTIFACT_DIR = "artifacts_phase2"

print("Loading artifacts...")
with open(f"{ARTIFACT_DIR}/feature_names.json") as f:
    feature_names = json.load(f)
with open(f"{ARTIFACT_DIR}/group_config.json") as f:
    grp_cfg = json.load(f)

scaler       = joblib.load(f"{ARTIFACT_DIR}/scaler.pkl")
group_names  = grp_cfg["group_names"]
groups       = {k: [int(i) for i in v] for k, v in grp_cfg["groups"].items()}
stats        = grp_cfg["stats"]

print(f"Feature names ({len(feature_names)}): {feature_names}")
print(f"Groups: {group_names}")

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

from backend.app.ml_service import _build_attention_layer
AttentionLayer = _build_attention_layer(tf, tf.keras.layers.Dense)

model = tf.keras.models.load_model(
    f"{ARTIFACT_DIR}/bilstm_attention_with_weights.keras",
    custom_objects={"AttentionLayer": AttentionLayer},
    compile=False,
)
print("Model loaded.\n")

files = {
    "parsec-windows.exe":   r"C:\Users\asus laptop\Downloads\parsec-windows.exe",
    "notepad.exe":          r"C:\Windows\System32\notepad.exe",
    "calc.exe":             r"C:\Windows\System32\calc.exe",
}

print(f"{'File':<35} {'Verdict':<12} {'Conf':>6}  Features")
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
        feat_str = "  ".join(f"{feature_names[i]}={vals[i]:.1f}" for i in range(len(feature_names)))

        X = np.array([vals], dtype=np.float32)
        X_sc = scaler.transform(X)
        seq = build_sequence(X_sc)

        pred, _ = model.predict(seq, verbose=0)
        conf = float(pred[0, 0])
        verdict = "Ransomware" if conf >= 0.5 else "Benign"
        result = "OK" if verdict == "Benign" else "FP!"
        print(f"{name:<35} {verdict:<12} {conf:>6.4f}  [{result}]")
        print(f"  {feat_str}")
        print()
    except Exception as e:
        print(f"{name}: ERROR {e}")
        import traceback; traceback.print_exc()
