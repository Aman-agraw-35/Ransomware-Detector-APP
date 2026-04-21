from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np

from .feature_extractor import extract_base_features, extract_feature_vector
from .pe_heuristic_classifier import heuristic_analyze


def _build_attention_layer(tf, Dense):
    """Factory that creates AttentionLayer only when TF is imported."""
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

    return AttentionLayer


class MLService:
    def __init__(self):
        root          = Path(__file__).resolve().parents[2]
        artifacts_dir = root / "artifacts_phase2"

        self.model_path           = Path(os.getenv("MODEL_PATH",           artifacts_dir / "bilstm_attention_model.keras"))
        self.attn_model_path      = Path(os.getenv("ATTN_MODEL_PATH",      artifacts_dir / "bilstm_attention_with_weights.keras"))
        self.scaler_path          = Path(os.getenv("SCALER_PATH",          artifacts_dir / "scaler.pkl"))
        self.xgb_path             = artifacts_dir / "xgb_model.pkl"   # RandomForest or XGBoost
        self.feature_names_path   = Path(os.getenv("FEATURE_NAMES_PATH",   artifacts_dir / "feature_names.json"))
        self.group_config_path    = Path(os.getenv("GROUP_CONFIG_PATH",     artifacts_dir / "group_config.json"))
        self.background_path      = Path(os.getenv("SHAP_BACKGROUND_PATH", artifacts_dir / "shap_background.npy"))
        self.threshold_config_path = artifacts_dir / "threshold_config.json"
        self.top_k                = int(os.getenv("SHAP_TOP_K", "15"))
        self._lock                = Lock()

        explicit_mock    = os.getenv("MODEL_MOCK", "0").strip().lower() in ("1", "true", "yes")
        artifacts_absent = not self.model_path.exists() and not self.attn_model_path.exists()
        self._mock       = explicit_mock or artifacts_absent

        if self._mock:
            self.model          = None
            self.attn_model     = None
            self.scaler         = None
            self.xgb_model      = None
            self.feature_names  = self._mock_feature_names()
            self.group_config   = self._mock_group_config()
            self.threshold      = 0.5
            self.bilstm_weight  = 0.4
            self.rf_weight      = 0.6
        else:
            import joblib
            import tensorflow as tf
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.models import Model

            self.model        = self._load_model(self.model_path, tf)
            self.attn_model   = self._load_attention_model(tf, Model)
            self.scaler        = joblib.load(self.scaler_path) if self.scaler_path.exists() else None
            self.xgb_model     = joblib.load(self.xgb_path) if self.xgb_path.exists() else None
            self.feature_names = self._load_feature_names()
            self.group_config  = self._load_group_config()
            self.threshold, self.bilstm_weight, self.rf_weight = self._load_threshold()
            self.shap_explainer = None

    # ── Threshold ────────────────────────────────────────────────────────────

    def _load_threshold(self) -> tuple:
        """Returns (threshold, bilstm_weight, ensemble_weight)."""
        if self.threshold_config_path.exists():
            with open(self.threshold_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            threshold  = float(cfg.get("decision_threshold", 0.5))
            ew         = cfg.get("ensemble_weights", {})
            bilstm_w   = float(ew.get("bilstm", 0.4))
            # support both key names
            rf_w       = float(ew.get("random_forest", ew.get("xgboost", 0.6)))
            return threshold, bilstm_w, rf_w
        return 0.5, 0.4, 0.6

    # ── Warm-up ──────────────────────────────────────────────────────────────

    def warm_up(self) -> None:
        """Pre-compile TF graph with a dummy input so first real request is fast."""
        if self._mock or self.attn_model is None:
            return
        try:
            cfg      = self.group_config
            n_groups = len(cfg.get("group_names", []))
            n_stats  = len(cfg.get("stats", ["mean", "std", "min", "max"]))
            if n_groups == 0:
                return
            dummy = np.zeros((1, n_groups, n_stats), dtype=np.float32)
            self.attn_model.predict(dummy, verbose=0)
        except Exception:  # noqa: BLE001
            pass

    # ── Mock helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _mock_feature_names() -> List[str]:
        return [
            "NumberOfSections", "SizeOfStackReserve", "ResourceSize", "BitcoinAddresses",
            "is_64bit", "dll_high_entropy_va", "dll_dynamic_base", "dll_nx_compat",
            "dll_no_seh", "dll_guard_cf", "dll_terminal_server",
            "has_exports", "has_debug_info", "has_iat", "is_large_stack",
        ]

    @staticmethod
    def _mock_group_config() -> Dict:
        groups = {
            "security_flags": [5, 6, 7, 8, 9, 10],
            "architecture":   [4, 14],
            "presence_flags": [11, 12, 13],
            "pe_structure":   [0],
            "resources":      [2],
            "memory":         [1],
            "bitcoin":        [3],
        }
        return {"groups": groups, "group_names": list(groups.keys()),
                "stats": ["mean", "std", "min", "max"]}

    def _mock_analyze(self, file_bytes: bytes, filename: str) -> Dict:
        """Heuristic analysis using actual PE feature extraction (no trained model)."""
        try:
            base_features = extract_base_features(file_bytes)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return heuristic_analyze(base_features, filename)

    # ── Real model helpers ────────────────────────────────────────────────────

    def _load_model(self, path: Path, tf):
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        from tensorflow.keras.layers import Dense  # noqa: PLC0415
        AttentionLayer_ = _build_attention_layer(tf, Dense)
        return tf.keras.models.load_model(
            path, custom_objects={"AttentionLayer": AttentionLayer_}, compile=False)

    def _load_attention_model(self, tf, Model):
        from tensorflow.keras.layers import Dense  # noqa: PLC0415
        AttentionLayer_ = _build_attention_layer(tf, Dense)
        if self.attn_model_path.exists():
            return tf.keras.models.load_model(
                self.attn_model_path,
                custom_objects={"AttentionLayer": AttentionLayer_},
                compile=False,
            )
        layer = self.model.get_layer("attention_layer")
        attn_output = layer.output
        if isinstance(attn_output, (list, tuple)) and len(attn_output) > 1:
            return Model(self.model.input, [self.model.output, attn_output[1]])
        raise RuntimeError("Attention model missing and could not be inferred.")

    def _load_feature_names(self) -> List[str]:
        if not self.feature_names_path.exists():
            raise FileNotFoundError(f"feature_names.json not found at {self.feature_names_path}")
        with open(self.feature_names_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(x) for x in data]

    def _load_group_config(self) -> Dict:
        if not self.group_config_path.exists():
            return {}
        with open(self.group_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["groups"]      = {k: [int(i) for i in v] for k, v in cfg.get("groups", {}).items()}
        cfg["group_names"] = [str(g) for g in cfg.get("group_names", list(cfg["groups"].keys()))]
        cfg["stats"]       = [str(s) for s in cfg.get("stats", ["mean", "std", "min", "max"])]
        return cfg

    def _build_grouped_sequences(self, X_matrix: np.ndarray) -> np.ndarray:
        groups      = self.group_config.get("groups", {})
        group_names = self.group_config.get("group_names", [])
        stats       = self.group_config.get("stats", ["mean", "std", "min", "max"])
        if not groups or not group_names:
            return X_matrix.reshape(X_matrix.shape[0], X_matrix.shape[1], 1)
        n         = X_matrix.shape[0]
        sequences = np.zeros((n, len(group_names), len(stats)), dtype=np.float32)
        for g_idx, g_name in enumerate(group_names):
            col_idx = groups.get(g_name, [])
            if not col_idx:
                continue
            values = X_matrix[:, col_idx]
            for s_idx, stat in enumerate(stats):
                if stat == "mean": sequences[:, g_idx, s_idx] = np.mean(values, axis=1)
                elif stat == "std": sequences[:, g_idx, s_idx] = np.std(values, axis=1)
                elif stat == "min": sequences[:, g_idx, s_idx] = np.min(values, axis=1)
                elif stat == "max": sequences[:, g_idx, s_idx] = np.max(values, axis=1)
        return sequences

    def _vector_to_sequence(self, vector: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            vector = self.scaler.transform(vector)
        return self._build_grouped_sequences(vector)

    def _feature_labels_for_sequence(self, seq_shape: Tuple[int, int, int]) -> List[str]:
        group_names = self.group_config.get("group_names", [])
        stats       = self.group_config.get("stats", ["mean", "std", "min", "max"])
        if group_names and stats and len(group_names) == seq_shape[1]:
            return [f"{g}_{s}" for g in group_names for s in stats]
        return [f"t{t}_f{f}" for t in range(seq_shape[1]) for f in range(seq_shape[2])]

    def _get_explainer(self, input_shape: Tuple[int, int, int]):
        if self.shap_explainer is not None:
            return self.shap_explainer
        import shap as _shap  # noqa: PLC0415
        if self.background_path.exists():
            background = np.load(self.background_path)
        else:
            background = np.zeros((32, input_shape[1], input_shape[2]), dtype=np.float32)
        self.shap_explainer = _shap.GradientExplainer(self.model, background)
        return self.shap_explainer

    def _compute_shap(self, sequence: np.ndarray) -> Dict:
        explainer   = self._get_explainer(sequence.shape)
        raw_values  = explainer.shap_values(sequence)
        shap_values = np.asarray(raw_values[0] if isinstance(raw_values, list) else raw_values)
        if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]
        flat_values = shap_values.reshape(shap_values.shape[0], -1)[0]
        labels      = self._feature_labels_for_sequence(sequence.shape)

        expected = getattr(explainer, "expected_value", None) or getattr(explainer, "expected_values", None)
        if expected is not None:
            base_value = float(np.asarray(expected).reshape(-1)[0])
        else:
            bg = np.load(self.background_path) if self.background_path.exists() else sequence
            baseline_pred = self.model.predict(bg[:min(64, len(bg))], verbose=0).reshape(-1)
            base_value = float(np.mean(baseline_pred))

        top_idx = np.argsort(np.abs(flat_values))[::-1][:self.top_k]
        top_features = [
            {"name": labels[i] if i < len(labels) else f"feature_{i}",
             "value": float(flat_values[i]),
             "abs_value": float(abs(flat_values[i]))}
            for i in top_idx
        ]
        return {"base_value": base_value, "top_features": top_features}

    # ── Main inference ────────────────────────────────────────────────────────

    def analyze_file(self, file_bytes: bytes, filename: str) -> Dict:
        if self._mock:
            return self._mock_analyze(file_bytes, filename)

        vector, missing_features = extract_feature_vector(file_bytes, self.feature_names)
        sequence                 = self._vector_to_sequence(vector)
        vector_scaled            = self.scaler.transform(vector) if self.scaler else vector

        with self._lock:
            prediction, attention = self.attn_model.predict(sequence, verbose=0)
            bilstm_prob = float(prediction[0, 0])

            # Ensemble: weights loaded from threshold_config.json (RF dominant to reduce FP)
            if self.xgb_model is not None:
                rf_prob    = float(self.xgb_model.predict_proba(vector_scaled)[0, 1])
                confidence = self.bilstm_weight * bilstm_prob + self.rf_weight * rf_prob
            else:
                confidence = bilstm_prob

            verdict    = "Ransomware" if confidence >= self.threshold else "Benign"
            shap_payload = self._compute_shap(sequence)

        attention_weights = attention[0, :, 0] if attention.ndim == 3 else attention.reshape(-1)
        group_names       = self.group_config.get("group_names", [f"step_{i}" for i in range(len(attention_weights))])

        return {
            "filename": filename,
            "verdict":  verdict,
            "confidence": confidence,
            "attention": {
                "labels":  group_names,
                "weights": [float(x) for x in attention_weights.tolist()],
            },
            "shap": shap_payload,
            "diagnostics": {
                "missing_features_count":   int(len(missing_features)),
                "missing_features_preview": missing_features[:20],
            },
        }
