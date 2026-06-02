"""
Complication Risk Predictor
Loads ht_model.pkl / dm_model.pkl and predicts CKD, Stroke, CAD risk.

Model pkl structure (confirmed from ht_model.pkl):
{
  "models":           { "co_ckd": XGBClassifier, "co_stroke": XGBClassifier, "co_cad": XGBClassifier },
  "scaler":           StandardScaler (14 features),
  "label_encoder_sex": LabelEncoder (FEMALE=0, MALE=1),
  "features":         [14 feature names in order],
  "targets":          ["co_ckd", "co_stroke", "co_cad"],
  "target_names":     {"co_ckd": "CKD", "co_stroke": "Stroke", "co_cad": "CAD"},
  "training_results": {...}
}
"""
from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Internal target keys (as stored in pkl)
_PKL_TARGETS = ["co_ckd", "co_stroke", "co_cad"]
# Public-facing keys used in API response
TARGETS = ["ckd", "stroke", "cad"]
_TARGET_MAP = dict(zip(_PKL_TARGETS, TARGETS))  # co_ckd → ckd, ...


# -------------------------------------------------------
# Risk thresholds
# -------------------------------------------------------
def risk_level(prob: float) -> str:
    if prob >= 0.60:
        return "HIGH"
    if prob >= 0.30:
        return "MEDIUM"
    return "LOW"


# -------------------------------------------------------
# Model bundle — matches confirmed pkl structure
# -------------------------------------------------------
class ModelBundle:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, dict) or "models" not in obj:
            raise ValueError(f"Unexpected model format in {path}")

        self.scaler: Any = obj["scaler"]
        self.label_encoder_sex: Any = obj.get("label_encoder_sex")
        self._models: dict[str, Any] = obj["models"]   # keys: co_ckd, co_stroke, co_cad
        self.features: list[str] = obj["features"]     # 14 feature names in order

    def encode_sex(self, sex_str: str) -> int:
        s = sex_str.strip().upper()
        if self.label_encoder_sex is not None:
            try:
                return int(self.label_encoder_sex.transform([s])[0])
            except Exception:
                pass
        return 1 if s == "MALE" else 0

    def predict_proba(self, raw_data: dict) -> dict[str, float]:
        """Build feature vector, scale, and return {ckd, stroke, cad: probability}."""
        sex_encoded = self.encode_sex(str(raw_data.get("sex", "")))

        def _f(key: str, fallback: float = 0.0) -> float:
            return float(raw_data.get(key) or fallback)

        sbp_mean   = _f("sbp_mean")
        dbp_mean   = _f("dbp_mean")
        hba1c_mean = _f("hba1c_mean")
        fpg_mean   = _f("fpg_mean")
        chol_mean  = _f("chol_mean")
        ldl_mean   = _f("ldl_mean")
        bmi_mean   = _f("bmi_mean")

        # Build row matching exact 33-feature order from ht_model.pkl:
        # vitalsign_sbp_mean/std/max, vitalsign_dbp_mean/std/max,
        # vitalsign_bmi_mean/std/max, lab_hba1c_mean/std/max,
        # lab_fpg_mean/std/max, lab_chol_mean/std/max, lab_ldl_mean/std/max,
        # age, sex, co_dm, co_stroke, co_cad, co_ckd, co_arrhythmias,
        # med_acei, med_arb, med_ccb, med_diuretics, med_beta_blocker
        value_map = {
            # SBP
            "vitalsign_sbp_mean": sbp_mean,
            "vitalsign_sbp_std":  _f("sbp_std"),
            "vitalsign_sbp_max":  _f("sbp_max", sbp_mean),
            # DBP
            "vitalsign_dbp_mean": dbp_mean,
            "vitalsign_dbp_std":  _f("dbp_std"),
            "vitalsign_dbp_max":  _f("dbp_max", dbp_mean),
            # BMI
            "vitalsign_bmi_mean": bmi_mean,
            "vitalsign_bmi_std":  _f("bmi_std"),
            "vitalsign_bmi_max":  _f("bmi_max", bmi_mean),
            # HbA1c
            "lab_hba1c_mean": hba1c_mean,
            "lab_hba1c_std":  _f("hba1c_std"),
            "lab_hba1c_max":  _f("hba1c_max", hba1c_mean),
            # FPG
            "lab_fpg_mean": fpg_mean,
            "lab_fpg_std":  _f("fpg_std"),
            "lab_fpg_max":  _f("fpg_max", fpg_mean),
            # Cholesterol
            "lab_chol_mean": chol_mean,
            "lab_chol_std":  _f("chol_std"),
            "lab_chol_max":  _f("chol_max", chol_mean),
            # LDL
            "lab_ldl_mean": ldl_mean,
            "lab_ldl_std":  _f("ldl_std"),
            "lab_ldl_max":  _f("ldl_max", ldl_mean),
            # Demographics
            "age": _f("age"),
            "sex": sex_encoded,
            # Comorbidities
            "co_dm":         float(raw_data.get("co_dm", 0) or 0),
            "co_stroke":     float(raw_data.get("co_stroke", 0) or 0),
            "co_cad":        float(raw_data.get("co_cad", 0) or 0),
            "co_ckd":        float(raw_data.get("co_ckd", 0) or 0),
            "co_arrhythmias": float(raw_data.get("co_arrhythmias", 0) or 0),
            # Current medications
            "med_acei":        float(raw_data.get("med_acei", 0) or 0),
            "med_arb":         float(raw_data.get("med_arb", 0) or 0),
            "med_ccb":         float(raw_data.get("med_ccb", 0) or 0),
            "med_diuretics":   float(raw_data.get("med_diuretics", 0) or 0),
            "med_beta_blocker": float(raw_data.get("med_beta_blocker", 0) or 0),
        }

        row = np.array([[value_map[f] for f in self.features]], dtype=np.float64)
        row_scaled = self.scaler.transform(row)

        result: dict[str, float] = {}
        for pkl_key, api_key in _TARGET_MAP.items():
            if pkl_key not in self._models:
                result[api_key] = 0.0
                continue
            proba = self._models[pkl_key].predict_proba(row_scaled)[0]
            result[api_key] = float(proba[1]) if len(proba) > 1 else float(proba[0])

        return result


# -------------------------------------------------------
# Load models once at import time
# -------------------------------------------------------
_HT_MODEL: ModelBundle | None = None
_DM_MODEL: ModelBundle | None = None


def _load_model(path: str, name: str) -> ModelBundle | None:
    if not os.path.exists(path):
        logger.warning("Model file not found: %s (%s predictions unavailable)", path, name)
        return None
    try:
        bundle = ModelBundle(path)
        logger.info("Loaded %s from %s", name, path)
        return bundle
    except Exception as exc:
        logger.error("Failed to load %s: %s", name, exc)
        return None


def load_all_models() -> None:
    global _HT_MODEL, _DM_MODEL
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _HT_MODEL = _load_model(os.path.join(_base, "model", "ht_model.pkl"), "ht_model")
    _DM_MODEL = _load_model(os.path.join(_base, "model", "dm_model.pkl"), "dm_model")


load_all_models()


# -------------------------------------------------------
# Combining formula  P = 1 - (1-P_HT)(1-P_DM)
# -------------------------------------------------------
def combine_probabilities(
    p_ht: dict[str, float], p_dm: dict[str, float]
) -> dict[str, float]:
    return {
        t: 1.0 - (1.0 - p_ht[t]) * (1.0 - p_dm[t])
        for t in TARGETS
    }


# -------------------------------------------------------
# Clinical explanation generator
# -------------------------------------------------------
def generate_explanation(data: dict, probs: dict[str, float], disease_group: str) -> str:
    factors: list[str] = []

    sbp   = float(data.get("sbp_mean")   or 0)
    hba1c = float(data.get("hba1c_mean") or 0)
    bmi   = float(data.get("bmi_mean")   or 0)
    age   = float(data.get("age")        or 0)

    if sbp >= 140:
        factors.append(f"elevated SBP ({sbp:.0f} mmHg)")
    if hba1c >= 7.0:
        factors.append(f"high HbA1c ({hba1c:.1f}%)")
    if bmi >= 30:
        factors.append(f"obesity (BMI {bmi:.1f})")
    if age >= 65:
        factors.append(f"advanced age ({int(age)} yr)")

    labels   = {"ckd": "CKD", "stroke": "Stroke", "cad": "CAD"}
    high     = [labels[t] for t in TARGETS if probs[t] >= 0.60]
    medium   = [labels[t] for t in TARGETS if 0.30 <= probs[t] < 0.60]

    if high:
        msg = f"HIGH risk detected for: {', '.join(high)}."
    elif medium:
        msg = f"MODERATE risk detected for: {', '.join(medium)}."
    else:
        msg = "Overall risk appears LOW across all complications."

    if factors:
        msg += f" Contributing factors: {', '.join(factors)}."

    msg += " Please consult a healthcare professional for clinical decision-making."
    return msg


# -------------------------------------------------------
# Main predict function (called by API route)
# -------------------------------------------------------
def predict(data: dict) -> dict:
    has_ht: bool = bool(data.get("has_ht", False))
    has_dm: bool = bool(data.get("has_dm", False))

    if not has_ht and not has_dm:
        raise ValueError("At least one of has_ht or has_dm must be True.")

    if has_ht and has_dm:
        if _HT_MODEL is None or _DM_MODEL is None:
            missing = "dm_model" if _DM_MODEL is None else "ht_model"
            raise RuntimeError(
                f"{missing}.pkl is not loaded — cannot compute combined prediction. "
                "Please ensure the model file exists in model/."
            )
        p_ht = _HT_MODEL.predict_proba(data)
        p_dm = _DM_MODEL.predict_proba(data)
        probs = combine_probabilities(p_ht, p_dm)
        disease_group = "HT_and_DM"

    elif has_ht:
        if _HT_MODEL is None:
            raise RuntimeError("ht_model.pkl is not loaded.")
        probs = _HT_MODEL.predict_proba(data)
        disease_group = "HT_only"

    else:
        if _DM_MODEL is None:
            raise RuntimeError("dm_model.pkl is not loaded.")
        probs = _DM_MODEL.predict_proba(data)
        disease_group = "DM_only"

    model_used_map = {"HT_only": "ht", "DM_only": "dm", "HT_and_DM": "combined"}

    return {
        "status":      "success",
        "model_used":  model_used_map.get(disease_group, disease_group),
        "risks":       {
            t: {
                "probability": round(probs[t], 4),
                "level":       risk_level(probs[t]),
            }
            for t in TARGETS
        },
        "explanation": generate_explanation(data, probs, disease_group),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
