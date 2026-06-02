"""
Medication Suggestion Test Script (CBR / KNN)
----------------------------------------------
Tests the /api/suggest logic directly without needing the server running.

Run from BDI_Hackathon/ root:
    python scripts/test_suggest.py
    pytest scripts/test_suggest.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from app.cbr import apply_weights, distance_weighted_rate, find_med_column, load_cleaned_model, recommendation_label
from app.config import (
    DM_DATA_PATH, DM_FEATURE_WEIGHTS, DM_MEDS,
    HT_DATA_PATH, HT_FEATURE_WEIGHTS, HT_MEDS,
    features_base, features_comorbidity,
)

# ---------------------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df_ht, X_ht, scaler_ht, knn_ht, features_ht = load_cleaned_model(
    os.path.join(_BASE, HT_DATA_PATH),
    ['vitalsign_sbp_0', 'vitalsign_dbp_0', 'co_dm',
     'vitalsign_hr_0', 'lab_chol_0', 'lab_ldl_0',
     'lab_tg_0', 'lab_hdl_0', 'lab_fpg_0'],
    HT_FEATURE_WEIGHTS
)

df_dm, X_dm, scaler_dm, knn_dm, features_dm = load_cleaned_model(
    os.path.join(_BASE, DM_DATA_PATH),
    ['lab_hba1c_0', 'lab_fpg_0', 'co_ht'],
    DM_FEATURE_WEIGHTS
)


def run_suggest(patient: dict, disease: str):
    """เรียก CBR logic เหมือนที่ main.py ทำ แล้ว return list of {med, score, label}"""
    df_query = pd.DataFrame([patient])
    df_query['sex'] = (
        df_query['sex'].astype(str).str.upper()
        .map({'MALE': 1, 'FEMALE': 0}).fillna(0)
    )

    if disease == 'hypertension':
        df_query['co_dm'] = df_query.get('co_dm', 0)
        for col in ['vitalsign_hr_0', 'lab_chol_0', 'lab_ldl_0', 'lab_tg_0', 'lab_hdl_0', 'lab_fpg_0']:
            if col not in df_query.columns:
                df_query[col] = 0
        feature_list = features_ht
        X_ref, scaler = X_ht, scaler_ht
        knn_model = knn_ht
        weight_map = HT_FEATURE_WEIGHTS
        current_df = df_ht
        target_meds = HT_MEDS
    else:  # diabetes
        df_query['co_ht'] = df_query.get('co_ht', 0)
        feature_list = features_dm
        X_ref, scaler = X_dm, scaler_dm
        knn_model = knn_dm
        weight_map = DM_FEATURE_WEIGHTS
        current_df = df_dm
        target_meds = DM_MEDS

    X_query = df_query[feature_list].copy()
    for col in X_query.columns:
        X_query[col] = pd.to_numeric(X_query[col], errors='coerce').fillna(X_ref[col].mean())

    X_query_scaled = scaler.transform(X_query)
    X_query_weighted = apply_weights(X_query_scaled, feature_list, weight_map)

    distances, indices = knn_model.kneighbors(X_query_weighted)
    distances = distances[0]
    indices = indices[0]
    neighbor_df = current_df.iloc[indices]

    results = []
    for med in target_meds:
        col = find_med_column(current_df, med)
        if col is None:
            continue
        score = distance_weighted_rate(neighbor_df[col], distances) * 100
        results.append({
            "med": med,
            "score": round(score, 1),
            "label": recommendation_label(score),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    {
        "label": "HT — ผู้ป่วย HT ที่มี CKD (ควรแนะนำ ACEi/ARB)",
        "disease": "hypertension",
        "input": {
            "age": 65, "sex": "MALE",
            "vitalsign_bmi_0": 27.5,
            "vitalsign_sbp_0": 160, "vitalsign_dbp_0": 98,
            "vitalsign_hr_0": 78,
            "lab_chol_0": 210, "lab_ldl_0": 130,
            "lab_tg_0": 150, "lab_hdl_0": 45, "lab_fpg_0": 110,
            "co_ckd": 1, "co_hf": 0, "co_cad": 0,
            "co_stroke": 0, "co_arrhythmias": 0,
            "co_atrial_fibrillation": 0, "co_dementia": 0, "co_dm": 0,
        },
        "expect_recommended": ["acei", "arb"],   # ควรติด Recommended หรือ Consider
    },
    {
        "label": "HT — ผู้ป่วย HT ที่มี CAD + Arrhythmias (ควรแนะนำ diuretics/ccb/arb)",
        "disease": "hypertension",
        "input": {
            "age": 70, "sex": "MALE",
            "vitalsign_bmi_0": 29.0,
            "vitalsign_sbp_0": 155, "vitalsign_dbp_0": 92,
            "vitalsign_hr_0": 95,
            "lab_chol_0": 230, "lab_ldl_0": 150,
            "lab_tg_0": 180, "lab_hdl_0": 40, "lab_fpg_0": 105,
            "co_ckd": 0, "co_hf": 0, "co_cad": 1,
            "co_stroke": 0, "co_arrhythmias": 1,
            "co_atrial_fibrillation": 0, "co_dementia": 0, "co_dm": 0,
        },
        "expect_recommended": ["diuretics", "ccb", "arb", "acei"],
    },
    {
        "label": "DM — ผู้ป่วย DM ที่มี HbA1c สูง (ควรแนะนำ metformin/insulin)",
        "disease": "diabetes",
        "input": {
            "age": 55, "sex": "FEMALE",
            "vitalsign_bmi_0": 30.0,
            "lab_hba1c_0": 10.5, "lab_fpg_0": 220,
            "co_ckd": 0, "co_hf": 0, "co_cad": 0,
            "co_stroke": 0, "co_arrhythmias": 0,
            "co_atrial_fibrillation": 0, "co_dementia": 0, "co_ht": 0,
        },
        "expect_recommended": ["metformin", "insulin"],
    },
    {
        "label": "DM — ผู้ป่วย DM ที่มี CKD (ควรหลีกเลี่ยง metformin บางชนิด — แค่ดูผล)",
        "disease": "diabetes",
        "input": {
            "age": 68, "sex": "MALE",
            "vitalsign_bmi_0": 26.0,
            "lab_hba1c_0": 8.5, "lab_fpg_0": 170,
            "co_ckd": 1, "co_hf": 0, "co_cad": 0,
            "co_stroke": 0, "co_arrhythmias": 0,
            "co_atrial_fibrillation": 0, "co_dementia": 0, "co_ht": 0,
        },
        "expect_recommended": None,   # แค่ดูผล ไม่ enforce
    },
]


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_all():
    print("=" * 65)
    print("  Medication Suggestion Test Results (CBR / KNN)")
    print("=" * 65)

    passed = 0
    failed = 0

    for case in CASES:
        results = run_suggest(case["input"], case["disease"])
        top5 = results[:5]

        print(f"\n┌─ {case['label']}")
        print(f"│  Disease  : {case['disease']}")
        expect = case["expect_recommended"]
        print(f"│  Expected : {', '.join(expect) + ' ควรอยู่ใน Recommended/Consider' if expect else '— (แค่ดูผล)'}")
        print(f"│  {'MEDICATION':<22}  {'SCORE':>7}   {'LABEL'}")
        print(f"│  {'----------':<22}  {'-----':>7}   {'-----'}")
        for r in top5:
            print(f"│  {r['med']:<22}  {r['score']:>6.1f}%   {r['label']}")

        if expect:
            med_label_map = {r["med"]: r["label"] for r in results}
            hits = [m for m in expect if med_label_map.get(m) in ("Recommended", "Consider")]
            ok = len(hits) > 0
            status = f"PASS ✓  ({', '.join(hits)} ติด Recommended/Consider)" if ok else \
                     f"FAIL ✗  ({', '.join(expect)} ไม่ติด Recommended/Consider เลย)"
        else:
            ok = True
            status = "INFO  (no assertion)"

        print(f"└─ {status}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 65)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 65)
    return failed == 0


# ---------------------------------------------------------------------------
# pytest interface
# ---------------------------------------------------------------------------

def test_ht_data_loads():
    assert len(df_ht) > 0, "cleaned_hypertension.xlsx โหลดไม่ได้หรือว่างเปล่า"


def test_dm_data_loads():
    assert len(df_dm) > 0, "cleaned_diabetes.xlsx โหลดไม่ได้หรือว่างเปล่า"


def test_suggest_returns_list():
    results = run_suggest(CASES[0]["input"], "hypertension")
    assert isinstance(results, list)
    assert len(results) > 0


def test_suggest_output_keys():
    results = run_suggest(CASES[0]["input"], "hypertension")
    for r in results:
        assert "med" in r and "score" in r and "label" in r


def test_scores_in_range():
    for case in CASES:
        results = run_suggest(case["input"], case["disease"])
        for r in results:
            assert 0.0 <= r["score"] <= 100.0, f"{r['med']} score {r['score']} out of range"


def test_ht_ckd_acei_arb_considered():
    """ผู้ป่วย HT+CKD ควรได้ ACEi หรือ ARB อยู่ใน Recommended/Consider"""
    results = run_suggest(CASES[0]["input"], "hypertension")
    med_map = {r["med"]: r["label"] for r in results}
    hits = [m for m in ["acei", "arb"] if med_map.get(m) in ("Recommended", "Consider")]
    assert len(hits) > 0, f"ACEi/ARB ควรติด Recommended/Consider แต่ได้: {med_map}"


def test_ht_cad_arrhythmias_considered():
    """ผู้ป่วย HT+CAD+Arrhythmias ควรได้ diuretics/ccb/arb/acei อย่างน้อย 1 ตัวใน Recommended/Consider"""
    results = run_suggest(CASES[1]["input"], "hypertension")
    med_map = {r["med"]: r["label"] for r in results}
    hits = [m for m in ["diuretics", "ccb", "arb", "acei"] if med_map.get(m) in ("Recommended", "Consider")]
    assert len(hits) > 0, f"ควรได้ diuretics/ccb/arb/acei แต่ได้: {med_map}"


def test_dm_high_hba1c_metformin_or_insulin():
    """ผู้ป่วย DM HbA1c สูง ควรได้ metformin หรือ insulin อยู่ใน Recommended/Consider"""
    results = run_suggest(CASES[2]["input"], "diabetes")
    med_map = {r["med"]: r["label"] for r in results}
    hits = [m for m in ["metformin", "insulin"] if med_map.get(m) in ("Recommended", "Consider")]
    assert len(hits) > 0, f"metformin/insulin ควรติด Recommended/Consider แต่ได้: {med_map}"


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
