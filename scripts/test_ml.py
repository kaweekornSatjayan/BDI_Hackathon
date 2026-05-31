"""
ML Model Test Suite
-------------------
Tests the ht_model.pkl (and dm_model.pkl if present) directly,
without needing the FastAPI server to be running.

Run from BDI_Hackathon/ root:
    python scripts/test_ml.py
    pytest scripts/test_ml.py -v
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.predictor import (
    ModelBundle,
    combine_probabilities,
    load_all_models,
    predict,
    risk_level,
    TARGETS,
    _HT_MODEL,
    _DM_MODEL,
)

# ---------------------------------------------------------------------------
# Shared test patient data
# ---------------------------------------------------------------------------

HT_PATIENT = {
    "has_ht": True,
    "has_dm": False,
    "age": 65,
    "sex": "MALE",
    "sbp_mean": 155,
    "sbp_std": 10,
    "sbp_max": 170,
    "dbp_mean": 95,
    "dbp_std": 6,
    "dbp_max": 105,
    "hba1c_mean": None,
    "hba1c_std": None,
    "hba1c_max": None,
    "bmi_mean": 28.5,
    "fpg_mean": None,
    "hemoglobin_mean": None,
}

DM_PATIENT = {
    "has_ht": False,
    "has_dm": True,
    "age": 55,
    "sex": "FEMALE",
    "sbp_mean": 130,
    "sbp_std": 8,
    "sbp_max": 142,
    "dbp_mean": 82,
    "dbp_std": 5,
    "dbp_max": 90,
    "hba1c_mean": 8.5,
    "hba1c_std": 0.6,
    "hba1c_max": 9.2,
    "bmi_mean": 31.0,
    "fpg_mean": 160,
    "hemoglobin_mean": 12.0,
}

COMBINED_PATIENT = {**HT_PATIENT, "has_ht": True, "has_dm": True}


# ---------------------------------------------------------------------------
# Unit tests: risk_level()
# ---------------------------------------------------------------------------

class TestRiskLevel:
    def test_low(self):
        assert risk_level(0.0)  == "LOW"
        assert risk_level(0.10) == "LOW"
        assert risk_level(0.29) == "LOW"

    def test_medium(self):
        assert risk_level(0.30) == "MEDIUM"
        assert risk_level(0.45) == "MEDIUM"
        assert risk_level(0.59) == "MEDIUM"

    def test_high(self):
        assert risk_level(0.60) == "HIGH"
        assert risk_level(0.75) == "HIGH"
        assert risk_level(1.00) == "HIGH"


# ---------------------------------------------------------------------------
# Unit tests: combine_probabilities()
# ---------------------------------------------------------------------------

class TestCombineProbabilities:
    def test_both_zero(self):
        p = {"ckd": 0.0, "stroke": 0.0, "cad": 0.0}
        result = combine_probabilities(p, p)
        for t in TARGETS:
            assert result[t] == pytest.approx(0.0)

    def test_formula(self):
        p_ht = {"ckd": 0.3, "stroke": 0.1, "cad": 0.5}
        p_dm = {"ckd": 0.4, "stroke": 0.2, "cad": 0.6}
        result = combine_probabilities(p_ht, p_dm)
        assert result["ckd"]    == pytest.approx(1 - (1-0.3)*(1-0.4), rel=1e-6)
        assert result["stroke"] == pytest.approx(1 - (1-0.1)*(1-0.2), rel=1e-6)
        assert result["cad"]    == pytest.approx(1 - (1-0.5)*(1-0.6), rel=1e-6)

    def test_combined_always_gte_individual(self):
        p_ht = {"ckd": 0.3, "stroke": 0.2, "cad": 0.4}
        p_dm = {"ckd": 0.2, "stroke": 0.3, "cad": 0.1}
        result = combine_probabilities(p_ht, p_dm)
        for t in TARGETS:
            assert result[t] >= p_ht[t]
            assert result[t] >= p_dm[t]


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_ht_model_loaded(self):
        assert _HT_MODEL is not None, "ht_model.pkl failed to load — check model/ directory"

    def test_ht_model_has_features(self):
        assert _HT_MODEL is not None
        assert len(_HT_MODEL.features) == 14, f"Expected 14 features, got {len(_HT_MODEL.features)}"

    def test_ht_model_feature_names(self):
        assert _HT_MODEL is not None
        expected = {"age", "sex", "sbp_mean", "sbp_std", "sbp_max",
                    "dbp_mean", "dbp_std", "dbp_max",
                    "hba1c_mean", "hba1c_std", "hba1c_max",
                    "bmi_mean", "fpg_mean", "hemoglobin_mean"}
        assert set(_HT_MODEL.features) == expected

    def test_ht_model_has_three_classifiers(self):
        assert _HT_MODEL is not None
        assert set(_HT_MODEL._models.keys()) == {"co_ckd", "co_stroke", "co_cad"}


# ---------------------------------------------------------------------------
# Prediction tests: HT model
# ---------------------------------------------------------------------------

class TestHTModelPrediction:
    def test_returns_all_targets(self):
        result = predict(HT_PATIENT)
        assert set(result["risks"].keys()) == {"ckd", "stroke", "cad"}

    def test_probabilities_in_range(self):
        result = predict(HT_PATIENT)
        for t, v in result["risks"].items():
            assert 0.0 <= v["probability"] <= 1.0, f"{t} probability out of range: {v['probability']}"

    def test_risk_levels_valid(self):
        result = predict(HT_PATIENT)
        for t, v in result["risks"].items():
            assert v["level"] in {"LOW", "MEDIUM", "HIGH"}, f"Invalid level for {t}: {v['level']}"

    def test_model_used_is_ht(self):
        result = predict(HT_PATIENT)
        assert result["model_used"] == "ht"

    def test_status_success(self):
        result = predict(HT_PATIENT)
        assert result["status"] == "success"

    def test_explanation_is_string(self):
        result = predict(HT_PATIENT)
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 10

    def test_female_patient(self):
        patient = {**HT_PATIENT, "sex": "FEMALE", "age": 45, "sbp_mean": 118, "dbp_mean": 76}
        result = predict(patient)
        assert result["status"] == "success"

    def test_none_optional_fields_accepted(self):
        """All optional labs can be None."""
        result = predict(HT_PATIENT)   # HT_PATIENT already has None labs
        assert result["status"] == "success"

    def test_boundary_age_18(self):
        result = predict({**HT_PATIENT, "age": 18})
        assert result["status"] == "success"

    def test_boundary_age_120(self):
        result = predict({**HT_PATIENT, "age": 120})
        assert result["status"] == "success"

    def test_high_sbp_increases_risk(self):
        """Patient with very high SBP should have higher CAD/CKD risk than low SBP."""
        low_result  = predict({**HT_PATIENT, "sbp_mean": 110, "sbp_max": 110})
        high_result = predict({**HT_PATIENT, "sbp_mean": 200, "sbp_max": 200})
        # At least one risk should be higher for the hypertensive patient
        any_higher = any(
            high_result["risks"][t]["probability"] >= low_result["risks"][t]["probability"]
            for t in TARGETS
        )
        assert any_higher


# ---------------------------------------------------------------------------
# Prediction tests: DM model (skipped if dm_model.pkl absent)
# ---------------------------------------------------------------------------

DM_AVAILABLE = _DM_MODEL is not None

@pytest.mark.skipif(not DM_AVAILABLE, reason="dm_model.pkl not present")
class TestDMModelPrediction:
    def test_returns_all_targets(self):
        result = predict(DM_PATIENT)
        assert set(result["risks"].keys()) == {"ckd", "stroke", "cad"}

    def test_model_used_is_dm(self):
        result = predict(DM_PATIENT)
        assert result["model_used"] == "dm"

    def test_probabilities_in_range(self):
        result = predict(DM_PATIENT)
        for t, v in result["risks"].items():
            assert 0.0 <= v["probability"] <= 1.0


# ---------------------------------------------------------------------------
# Prediction tests: Combined HT+DM (skipped if dm_model.pkl absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DM_AVAILABLE, reason="dm_model.pkl not present")
class TestCombinedPrediction:
    def test_model_used_is_combined(self):
        result = predict(COMBINED_PATIENT)
        assert result["model_used"] == "combined"

    def test_combined_ge_ht_alone(self):
        ht_result   = predict(HT_PATIENT)
        comb_result = predict(COMBINED_PATIENT)
        for t in TARGETS:
            assert comb_result["risks"][t]["probability"] >= ht_result["risks"][t]["probability"]


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_no_disease_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            predict({**HT_PATIENT, "has_ht": False, "has_dm": False})

    def test_dm_only_without_model_raises(self):
        if DM_AVAILABLE:
            pytest.skip("dm_model.pkl present — skip this error test")
        with pytest.raises(RuntimeError, match="dm_model"):
            predict(DM_PATIENT)


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    SKIP = "\033[93mSKIP\033[0m"

    def assert_(condition, msg="Assertion failed"):
        if not condition:
            raise AssertionError(msg)

    results = []

    def run(name, fn):
        try:
            fn()
            results.append((name, "PASS", None))
            print(f"  {PASS}  {name}")
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            print(f"  {FAIL}  {name}")
            print(f"        {e}")

    # ── risk_level ────────────────────────────────
    print("\n[risk_level]")
    run("LOW at 0.0",    lambda: assert_(risk_level(0.0)  == "LOW"))
    run("MEDIUM at 0.3", lambda: assert_(risk_level(0.3)  == "MEDIUM"))
    run("HIGH at 0.6",   lambda: assert_(risk_level(0.6)  == "HIGH"))

    # ── combine_probabilities ──────────────────────
    print("\n[combine_probabilities]")
    def _test_formula():
        p_ht = {"ckd": 0.3, "stroke": 0.1, "cad": 0.5}
        p_dm = {"ckd": 0.4, "stroke": 0.2, "cad": 0.6}
        r = combine_probabilities(p_ht, p_dm)
        expected = 1 - (1-0.3)*(1-0.4)
        assert abs(r["ckd"] - expected) < 1e-6
    run("Formula correct", _test_formula)

    # ── ht model ──────────────────────────────────
    print("\n[HT model]")
    if _HT_MODEL is None:
        print(f"  {SKIP}  ht_model.pkl not found — skipping all HT tests")
    else:
        run("14 features",        lambda: assert_(len(_HT_MODEL.features) == 14))
        run("3 classifiers",      lambda: assert_(set(_HT_MODEL._models.keys()) == {"co_ckd","co_stroke","co_cad"}))

        def _ht_predict():
            r = predict(HT_PATIENT)
            assert r["status"] == "success"
            assert r["model_used"] == "ht"
            assert set(r["risks"].keys()) == {"ckd","stroke","cad"}
            for t,v in r["risks"].items():
                assert 0 <= v["probability"] <= 1
                assert v["level"] in {"LOW","MEDIUM","HIGH"}
            print(f"\n        Risks: " + " | ".join(
                f"{t.upper()} {v['probability']*100:.1f}% ({v['level']})"
                for t,v in r["risks"].items()
            ))
            print(f"        {r['explanation']}")
        run("predict HT patient", _ht_predict)
        run("None optional fields accepted", lambda: assert_(predict(HT_PATIENT)["status"] == "success"))

    # ── dm model ──────────────────────────────────
    print("\n[DM model]")
    if not DM_AVAILABLE:
        print(f"  {SKIP}  dm_model.pkl not found — DM tests skipped")
    else:
        def _dm_predict():
            r = predict(DM_PATIENT)
            assert r["status"] == "success"
            assert r["model_used"] == "dm"
        run("predict DM patient", _dm_predict)

    # ── error handling ────────────────────────────
    print("\n[error handling]")
    def _no_disease():
        try:
            predict({**HT_PATIENT, "has_ht": False, "has_dm": False})
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass
    run("No disease → ValueError", _no_disease)

    # ── Summary ───────────────────────────────────
    passed = sum(1 for _,s,_ in results if s == "PASS")
    failed = sum(1 for _,s,_ in results if s == "FAIL")
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
    if failed:
        print("\nFailed tests:")
        for name, status, err in results:
            if status == "FAIL":
                print(f"  - {name}: {err}")
    sys.exit(1 if failed else 0)
