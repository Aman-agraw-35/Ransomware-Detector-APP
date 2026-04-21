"""
train_model.py  —  v4: Adds synthetic 32-bit benign samples + XGBoost ensemble.

The BiLSTM alone cannot distinguish old-32bit malware from old-32bit benign apps
because the dataset has almost no 32-bit benign examples with ASLR+DEP+NoSEH.

This version:
1. Augments the benign class with synthetic 32-bit benign samples to balance
   the architecture distribution before SMOTE.
2. Trains an XGBoost classifier on the same features as a secondary signal.
3. Ensemble: final verdict = mean of BiLSTM + XGBoost probabilities.
4. Saves both models and writes threshold_config.json with the calibrated threshold.
"""
from __future__ import annotations

import importlib, subprocess, sys

def ensure(pkg, pip_name=None):
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"[+] Installing {pip_name or pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg, "-q"])

ensure("sklearn",  "scikit-learn")
ensure("imblearn", "imbalanced-learn")
ensure("joblib")
ensure("shap")
ensure("tensorflow")
ensure("xgboost")

import json, os, random
import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import xgboost as xgb

from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, BatchNormalization, Bidirectional,
                                     Dense, Dropout, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

SEED         = 42
DATA_PATH    = "data_file.csv"
ARTIFACT_DIR = "artifacts_phase2"

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
print(f"TensorFlow : {tf.__version__}")

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"\n[1/9] Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
for col in ["FileName", "md5Hash"]:
    if col in df.columns: df = df.drop(columns=[col])
df = df.drop_duplicates().reset_index(drop=True)

y_raw = (df["Benign"] == 0).astype(int).values
df = df.drop(columns=["Benign"])
print(f"      Benign={int((y_raw==0).sum())}  Malware={int((y_raw==1).sum())}")

# ── Feature engineering ───────────────────────────────────────────────────────
print("\n[2/9] Engineering features ...")

raw_dll = df["DllCharacteristics"].astype(int)
DLLCHAR_BITS = {
    "dll_high_entropy_va": 0x0020,
    "dll_dynamic_base":    0x0040,
    "dll_nx_compat":       0x0100,
    "dll_no_seh":          0x0400,
    "dll_guard_cf":        0x4000,
    "dll_terminal_server": 0x8000,
}
for fname, bit in DLLCHAR_BITS.items():
    df[fname] = ((raw_dll & bit) != 0).astype(float)

df["is_64bit"]       = (df["Machine"] == 34404).astype(float)
df["has_exports"]    = (df["ExportSize"] > 0).astype(float)
df["has_debug_info"] = (df["DebugSize"]  > 0).astype(float)
df["has_iat"]        = (df["IatVRA"]     > 0).astype(float)

rs_cap = df["ResourceSize"].quantile(0.99)
df["ResourceSize"] = df["ResourceSize"].clip(upper=rs_cap)
df["is_large_stack"] = (df["SizeOfStackReserve"] >= 4_194_304).astype(float)
stk_cap = df["SizeOfStackReserve"].quantile(0.99)
df["SizeOfStackReserve"] = df["SizeOfStackReserve"].clip(upper=stk_cap)

drop_cols = [c for c in ["Machine","DllCharacteristics","MajorImageVersion","MajorOSVersion",
                          "MajorLinkerVersion","MinorLinkerVersion","DebugSize","DebugRVA",
                          "ExportRVA","ExportSize","IatVRA"] if c in df.columns]
df = df.drop(columns=drop_cols)

feature_names = df.columns.tolist()
X_raw = df.values.astype(np.float32)
y = y_raw.copy()
print(f"      Features ({len(feature_names)}): {feature_names}")

# ── Augment: add synthetic 32-bit benign samples ──────────────────────────────
print("\n[3/9] Augmenting benign class with 32-bit profiles ...")
# Build a pool of realistic 32-bit benign feature vectors by perturbing
# existing 32-bit benign samples from the dataset
is_32bit_benign = (y == 0) & (df["is_64bit"].values == 0)
n_32bit_benign  = is_32bit_benign.sum()
print(f"      Existing 32-bit benign samples: {n_32bit_benign}")

benign_32_X = X_raw[is_32bit_benign]
# Synthesise 3x more by adding small gaussian noise (feature-scale aware)
rng = np.random.default_rng(SEED)
synth_n = min(10_000, n_32bit_benign * 3 + 1)
base_idx = rng.integers(0, len(benign_32_X), size=synth_n)
noise_scale = np.std(benign_32_X, axis=0) * 0.05 + 1e-6
noise = rng.normal(0, 1, (synth_n, len(feature_names))).astype(np.float32) * noise_scale
synth_X = benign_32_X[base_idx] + noise
# Keep binary features binary
binary_cols = [i for i, fn in enumerate(feature_names)
               if fn.startswith("dll_") or fn in ("is_64bit","has_exports","has_debug_info",
                                                    "has_iat","is_large_stack")]
synth_X[:, binary_cols] = (synth_X[:, binary_cols] > 0.5).astype(np.float32)
# Force is_64bit=0 for all synthetic 32-bit samples
synth_X[:, feature_names.index("is_64bit")] = 0.0
synth_y = np.zeros(synth_n, dtype=int)

X_aug = np.vstack([X_raw, synth_X])
y_aug = np.concatenate([y, synth_y])
print(f"      After augment: total={len(y_aug)}  benign={int((y_aug==0).sum())}  malware={int((y_aug==1).sum())}")

# ── Split + Scale ─────────────────────────────────────────────────────────────
print("\n[4/9] Split / scale ...")
X_tr_full, X_test, y_tr_full, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=SEED, stratify=y_aug)
X_train, X_val, y_train, y_val = train_test_split(
    X_tr_full, y_tr_full, test_size=0.2, random_state=SEED, stratify=y_tr_full)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

smote = SMOTE(random_state=SEED)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)
# Guarantee 1-D labels for XGBoost and TF compatibility
y_train_bal = np.asarray(y_train_bal).ravel().astype(np.int32)
y_val       = np.asarray(y_val).ravel().astype(np.int32)
y_test      = np.asarray(y_test).ravel().astype(np.int32)
print(f"      After SMOTE: {X_train_bal.shape}")

# ── Groups ────────────────────────────────────────────────────────────────────
print("\n[5/9] Grouped sequences ...")
GROUP_MAP = {
    "security_flags": ["dll_dynamic_base","dll_nx_compat","dll_high_entropy_va",
                       "dll_no_seh","dll_guard_cf","dll_terminal_server"],
    "architecture":   ["is_64bit","is_large_stack"],
    "presence_flags": ["has_exports","has_debug_info","has_iat"],
    "pe_structure":   ["NumberOfSections"],
    "resources":      ["ResourceSize"],
    "memory":         ["SizeOfStackReserve"],
    "bitcoin":        ["BitcoinAddresses"],
}
STAT_NAMES = ["mean","std","min","max"]
feature_groups = {}
for g, cols in GROUP_MAP.items():
    idxs = [feature_names.index(c) for c in cols if c in feature_names]
    if idxs: feature_groups[g] = idxs
group_names = list(feature_groups.keys())

def build_seq(X_mat):
    n = X_mat.shape[0]
    seq = np.zeros((n, len(group_names), len(STAT_NAMES)), dtype=np.float32)
    for g_idx, g in enumerate(group_names):
        idxs = feature_groups[g]
        vals = X_mat[:, idxs]
        seq[:, g_idx, 0] = np.mean(vals, axis=1)
        seq[:, g_idx, 1] = np.std(vals,  axis=1)
        seq[:, g_idx, 2] = np.min(vals,  axis=1)
        seq[:, g_idx, 3] = np.max(vals,  axis=1)
    return seq

X_train_seq = build_seq(X_train_bal)
X_val_seq   = build_seq(X_val_sc)
X_test_seq  = build_seq(X_test_sc)
print(f"      Seq shape: {X_train_seq.shape}")

# ── XGBoost (flat features, no sequencing) ────────────────────────────────────
print("\n[6/9] Training XGBoost classifier ...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=1.0, eval_metric="auc",
    random_state=SEED, n_jobs=-1,
    early_stopping_rounds=20,
)
xgb_model.fit(
    X_train_sc, y_train_bal.ravel(),   # flat scaled features + SMOTE-balanced
    eval_set=[(X_val_sc, y_val)],
    verbose=50,
)
xgb_prob_test = xgb_model.predict_proba(X_test_sc)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_prob_test)
print(f"      XGBoost test AUC: {xgb_auc:.4f}")

# ── BiLSTM ────────────────────────────────────────────────────────────────────
print("\n[7/9] Training BiLSTM + Attention ...")

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = Dense(1)

    def build(self, input_shape):
        self.score_dense.build(input_shape)
        super().build(input_shape)

    def call(self, lstm_output):
        score   = self.score_dense(lstm_output)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * lstm_output, axis=1)
        return context, weights

    def get_config(self):
        return super().get_config()

T, F = X_train_seq.shape[1], X_train_seq.shape[2]
inputs  = Input(shape=(T, F), name="sequence_input")
x       = Bidirectional(LSTM(128, return_sequences=True), name="bilstm_1")(inputs)
x       = Dropout(0.3)(x)
x       = Bidirectional(LSTM(64,  return_sequences=True), name="bilstm_2")(x)
ctx, aw = AttentionLayer(name="attention_layer")(x)
x       = Dense(64, activation="relu")(ctx)
x       = BatchNormalization()(x)
outputs = Dense(1, activation="sigmoid", name="prediction")(x)

model      = Model(inputs, outputs,       name="bilstm_attention_detector")
attn_model = Model(inputs, [outputs, aw], name="bilstm_attention_with_weights")
model.compile(
    loss="binary_crossentropy", optimizer=Adam(1e-3),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=7,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                      patience=3, min_lr=1e-5, verbose=1),
]
model.fit(
    X_train_seq, y_train_bal,
    validation_data=(X_val_seq, y_val),
    epochs=50, batch_size=64,
    callbacks=callbacks, verbose=1,
)

# ── Ensemble evaluation ───────────────────────────────────────────────────────
print("\n[8/9] Ensemble evaluation ...")
bilstm_prob = model.predict(X_test_seq, verbose=0).ravel()
# Ensemble = weighted average (BiLSTM 40%, XGBoost 60%)
ensemble_prob = 0.4 * bilstm_prob + 0.6 * xgb_prob_test
ens_auc = roc_auc_score(y_test, ensemble_prob)
print(f"      Ensemble AUC: {ens_auc:.4f}")

# Find optimal threshold
print("\n      Threshold sweep:")
print(f"      {'Thresh':>7} {'FNR':>8} {'FPR':>8} {'Acc':>8}")
best_thresh, best_cost = 0.5, 999
for t in np.arange(0.20, 0.90, 0.05):
    y_pred = (ensemble_prob >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    acc = (tn + tp) / len(y_test)
    mark = ""
    if fnr < 0.05 and fpr < 0.05 and best_cost > fnr + fpr:
        best_cost = fnr + fpr
        best_thresh = float(t)
        mark = " <--"
    print(f"      {t:>7.2f} {fnr:>8.4f} {fpr:>8.4f} {acc:>8.4f}{mark}")

y_pred_best = (ensemble_prob >= best_thresh).astype(int)
print(f"\n  Best threshold: {best_thresh:.2f}")
print(f"\n{classification_report(y_test, y_pred_best, target_names=['Benign','Ransomware'])}")

# ── Save artifacts ────────────────────────────────────────────────────────────
print("\n[9/9] Saving artifacts ...")
np.save(f"{ARTIFACT_DIR}/shap_background.npy", X_train_seq[:200])
model.save(f"{ARTIFACT_DIR}/bilstm_attention_model.keras")
attn_model.save(f"{ARTIFACT_DIR}/bilstm_attention_with_weights.keras")
joblib.dump(scaler,     f"{ARTIFACT_DIR}/scaler.pkl")
joblib.dump(xgb_model,  f"{ARTIFACT_DIR}/xgb_model.pkl")

with open(f"{ARTIFACT_DIR}/feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)
serializable_groups = {k: [int(i) for i in v] for k, v in feature_groups.items()}
with open(f"{ARTIFACT_DIR}/group_config.json", "w") as f:
    json.dump({"group_names": group_names, "groups": serializable_groups,
               "stats": STAT_NAMES}, f, indent=2)
with open(f"{ARTIFACT_DIR}/threshold_config.json", "w") as f:
    json.dump({"decision_threshold": round(best_thresh, 2),
               "ensemble_weights": {"bilstm": 0.4, "xgboost": 0.6}}, f, indent=2)

for fn in ["bilstm_attention_model.keras","bilstm_attention_with_weights.keras",
           "scaler.pkl","xgb_model.pkl","feature_names.json",
           "group_config.json","threshold_config.json","shap_background.npy"]:
    p = f"{ARTIFACT_DIR}/{fn}"
    if os.path.exists(p):
        print(f"  OK  {p}  ({os.path.getsize(p)//1024} KB)")

print("\nDone! Restart uvicorn to apply the new ensemble model.")
