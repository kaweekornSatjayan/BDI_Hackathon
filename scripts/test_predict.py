"""
Predict API Test Script (33-feature model)
------------------------------------------
Tests ht_model.pkl directly without needing the server running.

Run from BDI_Hackathon/ root:
    python scripts/test_predict.py
    pytest scripts/test_predict.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.predictor import _HT_MODEL, risk_level

# ---------------------------------------------------------------------------
# Test cases
# Each case has:
#   - label:          ชื่อ scenario
#   - input:          dict ส่งเข้า predict_proba()
#   - expect_ckd:     "LOW" | "MEDIUM" | "HIGH"  (ความเสี่ยง CKD ที่คาดหวัง)
#   - expect_stroke:  same
#   - expect_cad:     same
# ---------------------------------------------------------------------------

CASES = [
    {
        "label": "HIGH-RISK — ชายสูงอายุ BP สูงมาก เบาหวาน ไขมันสูง มีโรคร่วมหลายอย่าง",
        "input": {
            "sex": "MALE", "age": 72,
            "sbp_mean": 168, "sbp_std": 12, "sbp_max": 185,
            "dbp_mean": 98,  "dbp_std": 8,  "dbp_max": 112,
            "bmi_mean": 30.5, "bmi_std": 1.2, "bmi_max": 32,
            "hba1c_mean": 9.8, "hba1c_std": 0.6, "hba1c_max": 10.5,
            "fpg_mean": 190,  "fpg_std": 22, "fpg_max": 215,
            "chol_mean": 250, "chol_std": 18, "chol_max": 270,
            "ldl_mean": 165,  "ldl_std": 12, "ldl_max": 180,
            "co_dm": 1, "co_stroke": 0, "co_cad": 1,
            "co_ckd": 0, "co_arrhythmias": 1,
            "med_acei": 1, "med_arb": 0, "med_ccb": 1,
            "med_diuretics": 1, "med_beta_blocker": 0,
        },
        "expect_level": "HIGH",   # คาดว่าอย่างน้อย 1 target ควรเป็น HIGH หรือ MEDIUM
    },
    {
        "label": "LOW-RISK — หญิงอายุน้อย BP ปกติ ไม่มีโรคร่วม",
        "input": {
            "sex": "FEMALE", "age": 35,
            "sbp_mean": 118, "sbp_std": 4, "sbp_max": 124,
            "dbp_mean": 75,  "dbp_std": 3, "dbp_max": 80,
            "bmi_mean": 21.5, "bmi_std": 0.4, "bmi_max": 22,
            "hba1c_mean": 5.2, "hba1c_std": 0.1, "hba1c_max": 5.5,
            "fpg_mean": 85,  "fpg_std": 4,  "fpg_max": 90,
            "chol_mean": 170, "chol_std": 8,  "chol_max": 180,
            "ldl_mean": 95,   "ldl_std": 6,  "ldl_max": 102,
            "co_dm": 0, "co_stroke": 0, "co_cad": 0,
            "co_ckd": 0, "co_arrhythmias": 0,
            "med_acei": 0, "med_arb": 0, "med_ccb": 0,
            "med_diuretics": 0, "med_beta_blocker": 0,
        },
        "expect_level": "LOW",
    },
    {
        "label": "MEDIUM-RISK — ชายวัยกลางคน BP สูงปานกลาง เบาหวานเริ่มต้น",
        "input": {
            "sex": "MALE", "age": 55,
            "sbp_mean": 145, "sbp_std": 7, "sbp_max": 156,
            "dbp_mean": 90,  "dbp_std": 5, "dbp_max": 98,
            "bmi_mean": 27.0, "bmi_std": 0.8, "bmi_max": 28,
            "hba1c_mean": 7.0, "hba1c_std": 0.3, "hba1c_max": 7.5,
            "fpg_mean": 120,  "fpg_std": 10, "fpg_max": 135,
            "chol_mean": 215, "chol_std": 12, "chol_max": 228,
            "ldl_mean": 135,  "ldl_std": 9,  "ldl_max": 145,
            "co_dm": 1, "co_stroke": 0, "co_cad": 0,
            "co_ckd": 0, "co_arrhythmias": 0,
            "med_acei": 0, "med_arb": 1, "med_ccb": 0,
            "med_diuretics": 0, "med_beta_blocker": 1,
        },
        "expect_level": None,  # ไม่ enforce ระดับ เพียงแค่ดูผลลัพธ์
    },
    {
        "label": "EDGE — ค่า std/max ไม่ได้ใส่ (fallback to 0 / mean)",
        "input": {
            "sex": "MALE", "age": 60,
            "sbp_mean": 150, "sbp_std": 0, "sbp_max": 150,
            "dbp_mean": 92,  "dbp_std": 0, "dbp_max": 92,
            "bmi_mean": 26.0, "bmi_std": 0, "bmi_max": 26,
            "hba1c_mean": 6.5, "hba1c_std": 0, "hba1c_max": 6.5,
            "fpg_mean": 110,  "fpg_std": 0,  "fpg_max": 110,
            "chol_mean": 200, "chol_std": 0,  "chol_max": 200,
            "ldl_mean": 120,  "ldl_std": 0,  "ldl_max": 120,
            "co_dm": 0, "co_stroke": 0, "co_cad": 0,
            "co_ckd": 0, "co_arrhythmias": 0,
            "med_acei": 0, "med_arb": 0, "med_ccb": 0,
            "med_diuretics": 0, "med_beta_blocker": 0,
        },
        "expect_level": None,
    },
]


# ---------------------------------------------------------------------------
# Standalone runner (python scripts/test_predict.py)
# ---------------------------------------------------------------------------

def run_all():
    assert _HT_MODEL is not None, "ht_model.pkl ไม่ได้ถูก load — ตรวจสอบว่าไฟล์อยู่ที่ model/ht_model.pkl"

    print("=" * 60)
    print("  Predict Test Results (ht_model)")
    print("=" * 60)

    passed = 0
    failed = 0

    for case in CASES:
        result = _HT_MODEL.predict_proba(case["input"])
        levels = {k: risk_level(v) for k, v in result.items()}

        print(f"\n[{case['label']}]")
        for target, prob in result.items():
            print(f"  {target.upper():8s}: {prob*100:5.1f}%  ({levels[target]})")

        if case["expect_level"] == "HIGH":
            ok = any(v >= 0.60 for v in result.values())
            status = "PASS ✓" if ok else "FAIL ✗ (expected HIGH ≥60% for at least one target)"
        elif case["expect_level"] == "LOW":
            ok = all(v < 0.30 for v in result.values())
            status = "PASS ✓" if ok else "FAIL ✗ (expected ALL targets LOW <30%)"
        else:
            ok = True
            status = "INFO  (no expectation set)"

        print(f"  → {status}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


# ---------------------------------------------------------------------------
# pytest interface
# ---------------------------------------------------------------------------

def test_model_loads():
    assert _HT_MODEL is not None, "ht_model.pkl failed to load"


def test_output_keys():
    result = _HT_MODEL.predict_proba(CASES[0]["input"])
    assert set(result.keys()) == {"ckd", "stroke", "cad"}


def test_probabilities_in_range():
    for case in CASES:
        result = _HT_MODEL.predict_proba(case["input"])
        for target, prob in result.items():
            assert 0.0 <= prob <= 1.0, f"{target} prob {prob} out of range"


def test_high_risk_case():
    """HIGH-RISK patient should produce at least one MEDIUM or HIGH prediction."""
    result = _HT_MODEL.predict_proba(CASES[0]["input"])
    assert any(v >= 0.30 for v in result.values()), \
        f"High-risk patient produced all LOW predictions: {result}"


def test_low_risk_case():
    """LOW-RISK patient should produce all LOW predictions (<30%)."""
    result = _HT_MODEL.predict_proba(CASES[1]["input"])
    assert all(v < 0.30 for v in result.values()), \
        f"Low-risk patient produced unexpected HIGH/MEDIUM predictions: {result}"


def test_edge_zero_std():
    """Edge case: std=0, max=mean — should not crash."""
    result = _HT_MODEL.predict_proba(CASES[3]["input"])
    assert len(result) == 3


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
