"""
API Integration + Validation + Boundary Test Suite
----------------------------------------------------
Tests FastAPI endpoints via HTTP (server must be running).
Also tests input validation and boundary values directly.

Run from BDI_Hackathon/ root:
    # Start server first:
    python run.py

    # Then in another terminal:
    python scripts/test_api.py
    pytest scripts/test_api.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(path: str, body: dict):
    """HTTP POST helper — uses urllib (no extra deps required)."""
    import json
    import urllib.request
    import urllib.error

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return None, str(e)


def _server_available() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{BASE_URL}/", timeout=3)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Base valid payloads
# ---------------------------------------------------------------------------

VALID_PREDICT = {
    "has_ht": True, "has_dm": False,
    "age": 65, "sex": "MALE",
    "sbp_mean": 155, "sbp_std": 10, "sbp_max": 170,
    "dbp_mean": 95,  "dbp_std": 6,  "dbp_max": 105,
    "bmi_mean": 28.5,
    "hba1c_mean": 7.0, "fpg_mean": 120,
    "chol_mean": 210, "ldl_mean": 130,
    "co_dm": 0, "co_stroke": 0, "co_cad": 0, "co_ckd": 0, "co_arrhythmias": 0,
    "med_acei": 0, "med_arb": 0, "med_ccb": 0, "med_diuretics": 0, "med_beta_blocker": 0,
}

VALID_SUGGEST = {
    "disease_type": "hypertension",
    "age": 60, "sex": "MALE",
    "vitalsign_bmi_0": 27.5,
    "vitalsign_sbp_0": 150, "vitalsign_dbp_0": 92,
    "vitalsign_hr_0": 75,
    "lab_chol_0": 200, "lab_ldl_0": 125,
    "lab_tg_0": 140, "lab_hdl_0": 48, "lab_fpg_0": 105,
    "co_ckd": 0, "co_hf": 0, "co_cad": 0,
    "co_stroke": 0, "co_arrhythmias": 0,
    "co_atrial_fibrillation": 0, "co_dementia": 0,
}


# ---------------------------------------------------------------------------
# Section 1 — API Integration Tests (requires server running)
# ---------------------------------------------------------------------------

def run_integration_tests():
    if not _server_available():
        print("  ⚠  Server ไม่ได้รัน — ข้าม integration tests (รัน: python run.py)")
        return None  # skip

    results = []

    # Test 1: POST /predict returns correct shape
    # Response: {status, model_used, risks: {ckd, stroke, cad: {probability, level}}, explanation, timestamp}
    status, body = _post("/predict", VALID_PREDICT)
    ok = status == 200 and body.get("status") == "success" and "risks" in body
    results.append(("POST /predict — status 200 + has 'risks' key", ok,
                     f"status={status}, keys={list(body.keys()) if isinstance(body, dict) else body}"))

    # Test 2: /predict risks has ckd/stroke/cad with probability + level
    if ok:
        risks = body["risks"]
        ok2 = all(k in risks and "probability" in risks[k] and "level" in risks[k]
                  for k in ["ckd", "stroke", "cad"])
        results.append(("POST /predict — risks มี ckd, stroke, cad พร้อม probability+level", ok2,
                         f"risk keys: {list(risks.keys())}"))
    else:
        results.append(("POST /predict — risks มี ckd, stroke, cad พร้อม probability+level", False, "skipped (prev failed)"))

    # Test 3: POST /api/suggest returns dict with 'suggestions' key
    # Response: {status, disease_mode, match_details, suggestions: [...], not_relevant: [...]}
    status2, body2 = _post("/api/suggest", VALID_SUGGEST)
    ok3 = status2 == 200 and isinstance(body2, dict) and "suggestions" in body2
    results.append(("POST /api/suggest — status 200 + has 'suggestions' key", ok3,
                     f"status={status2}, keys={list(body2.keys()) if isinstance(body2, dict) else body2}"))

    # Test 4: /api/suggest suggestions is a list with items
    if ok3:
        suggestions = body2["suggestions"]
        ok4 = isinstance(suggestions, list) and len(suggestions) > 0
        results.append(("POST /api/suggest — suggestions เป็น list ที่มีรายการ", ok4,
                         f"len={len(suggestions)}, first keys={list(suggestions[0].keys()) if suggestions else 'empty'}"))
    else:
        results.append(("POST /api/suggest — suggestions เป็น list ที่มีรายการ", False, "skipped"))

    return results


# ---------------------------------------------------------------------------
# Section 2 — Input Validation Tests (no server needed)
# ---------------------------------------------------------------------------

def run_validation_tests():
    """ทดสอบ Pydantic validation โดยตรง ไม่ต้องรัน server"""
    from pydantic import ValidationError
    from app.schemas import PredictInput

    results = []

    # Test: age ต่ำกว่า 18
    try:
        PredictInput(**{**VALID_PREDICT, "age": 10})
        results.append(("Validation: age=10 (ต่ำกว่า 18) ควร error", False, "ไม่ raise error"))
    except ValidationError:
        results.append(("Validation: age=10 (ต่ำกว่า 18) ควร error", True, "ValidationError raised ✓"))

    # Test: age เกิน 120
    try:
        PredictInput(**{**VALID_PREDICT, "age": 999})
        results.append(("Validation: age=999 (เกิน 120) ควร error", False, "ไม่ raise error"))
    except ValidationError:
        results.append(("Validation: age=999 (เกิน 120) ควร error", True, "ValidationError raised ✓"))

    # Test: sbp_mean ต่ำกว่า 90
    try:
        PredictInput(**{**VALID_PREDICT, "sbp_mean": 50})
        results.append(("Validation: sbp_mean=50 (ต่ำกว่า 90) ควร error", False, "ไม่ raise error"))
    except ValidationError:
        results.append(("Validation: sbp_mean=50 (ต่ำกว่า 90) ควร error", True, "ValidationError raised ✓"))

    # Test: sbp_mean เกิน 250
    try:
        PredictInput(**{**VALID_PREDICT, "sbp_mean": 300})
        results.append(("Validation: sbp_mean=300 (เกิน 250) ควร error", False, "ไม่ raise error"))
    except ValidationError:
        results.append(("Validation: sbp_mean=300 (เกิน 250) ควร error", True, "ValidationError raised ✓"))

    # Test: input ถูกต้อง ควร pass
    try:
        PredictInput(**VALID_PREDICT)
        results.append(("Validation: valid input ควร pass", True, "สร้าง object ได้ ✓"))
    except ValidationError as e:
        results.append(("Validation: valid input ควร pass", False, str(e)))

    return results


# ---------------------------------------------------------------------------
# Section 3 — Boundary Value Tests
# ---------------------------------------------------------------------------

def run_boundary_tests():
    from app.predictor import risk_level

    results = []

    # Boundary: ขอบ LOW/MEDIUM (29.9% → LOW, 30.0% → MEDIUM)
    cases = [
        (0.299, "LOW",    "29.9% ควรเป็น LOW"),
        (0.300, "MEDIUM", "30.0% ควรเป็น MEDIUM"),
        (0.599, "MEDIUM", "59.9% ควรเป็น MEDIUM"),
        (0.600, "HIGH",   "60.0% ควรเป็น HIGH"),
        (0.0,   "LOW",    "0% ควรเป็น LOW"),
        (1.0,   "HIGH",   "100% ควรเป็น HIGH"),
    ]
    for prob, expected, label in cases:
        actual = risk_level(prob)
        ok = actual == expected
        results.append((f"Boundary: {label}", ok,
                         f"expected={expected}, actual={actual}"))

    return results


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_all():
    total_passed = 0
    total_failed = 0

    def print_section(title, results):
        nonlocal total_passed, total_failed
        print(f"\n{'=' * 65}")
        print(f"  {title}")
        print(f"{'=' * 65}")
        if results is None:
            print("  ⚠  Skipped")
            return
        for label, ok, detail in results:
            icon = "PASS ✓" if ok else "FAIL ✗"
            print(f"  [{icon}] {label}")
            if not ok or "✓" in detail:
                print(f"          → {detail}")
            if ok:
                total_passed += 1
            else:
                total_failed += 1

    print_section("1. API Integration Tests (HTTP)", run_integration_tests())
    print_section("2. Input Validation Tests (Pydantic)", run_validation_tests())
    print_section("3. Boundary Value Tests (risk_level)", run_boundary_tests())

    print(f"\n{'=' * 65}")
    print(f"  Total: {total_passed} passed, {total_failed} failed")
    print(f"{'=' * 65}")
    return total_failed == 0


# ---------------------------------------------------------------------------
# pytest interface
# ---------------------------------------------------------------------------

# -- Validation --
def test_validation_age_too_low():
    from pydantic import ValidationError
    from app.schemas import PredictInput
    with pytest.raises(ValidationError):
        PredictInput(**{**VALID_PREDICT, "age": 5})


def test_validation_age_too_high():
    from pydantic import ValidationError
    from app.schemas import PredictInput
    with pytest.raises(ValidationError):
        PredictInput(**{**VALID_PREDICT, "age": 200})


def test_validation_sbp_too_low():
    from pydantic import ValidationError
    from app.schemas import PredictInput
    with pytest.raises(ValidationError):
        PredictInput(**{**VALID_PREDICT, "sbp_mean": 50})


def test_validation_sbp_too_high():
    from pydantic import ValidationError
    from app.schemas import PredictInput
    with pytest.raises(ValidationError):
        PredictInput(**{**VALID_PREDICT, "sbp_mean": 300})


def test_validation_valid_input_passes():
    from app.schemas import PredictInput
    obj = PredictInput(**VALID_PREDICT)
    assert obj.age == 65


# -- Boundary --
def test_boundary_low_medium():
    from app.predictor import risk_level
    assert risk_level(0.299) == "LOW"
    assert risk_level(0.300) == "MEDIUM"


def test_boundary_medium_high():
    from app.predictor import risk_level
    assert risk_level(0.599) == "MEDIUM"
    assert risk_level(0.600) == "HIGH"


def test_boundary_extremes():
    from app.predictor import risk_level
    assert risk_level(0.0) == "LOW"
    assert risk_level(1.0) == "HIGH"


# -- Integration (skip if server not running) --
@pytest.mark.skipif(not _server_available(), reason="Server not running")
def test_api_predict_status_200():
    status, body = _post("/predict", VALID_PREDICT)
    assert status == 200, f"Expected 200 got {status}: {body}"


@pytest.mark.skipif(not _server_available(), reason="Server not running")
def test_api_predict_has_risks():
    _, body = _post("/predict", VALID_PREDICT)
    assert "risks" in body
    for target in ["ckd", "stroke", "cad"]:
        assert target in body["risks"]
        assert "probability" in body["risks"][target]
        assert "level" in body["risks"][target]


@pytest.mark.skipif(not _server_available(), reason="Server not running")
def test_api_suggest_has_suggestions():
    status, body = _post("/api/suggest", VALID_SUGGEST)
    assert status == 200
    assert isinstance(body, dict) and "suggestions" in body
    assert isinstance(body["suggestions"], list) and len(body["suggestions"]) > 0


@pytest.mark.skipif(not _server_available(), reason="Server not running")
def test_api_predict_status_field():
    _, body = _post("/predict", VALID_PREDICT)
    assert body.get("status") == "success"

@pytest.mark.skipif(not _server_available(), reason="Server not running")
def test_api_predict_status_field():
    _, body = _post("/predict", VALID_PREDICT)
    assert body.get("status") == "success"


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
