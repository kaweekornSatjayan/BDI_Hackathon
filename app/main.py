import logging

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.cbr import (
    apply_weights,
    distance_weighted_rate,
    find_med_column,
    load_cleaned_model,
    recommendation_label,
)
from app.config import (
    DM_DATA_PATH,
    DM_FEATURE_WEIGHTS,
    DM_MEDS,
    HT_DATA_PATH,
    HT_FEATURE_WEIGHTS,
    HT_MEDS,
    features_comorbidity,
    features_base,
)
from app.schemas import PatientInput, PredictInput
from app import predictor

app = FastAPI(
    title="BDI Hackathon CDSS API",
    description="Clinical Decision Support System — Medication Suggestion (CBR) + Complication Risk Prediction (ML).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# Startup — load datasets into KNN models
# -------------------------------------------------------------
print("🔄 Loading professionally cleaned datasets into KNN models...")
try:
    df_dm, X_dm, scaler_dm, knn_dm, features_dm = load_cleaned_model(
        DM_DATA_PATH, ['lab_hba1c_0', 'lab_fpg_0', 'co_ht'], DM_FEATURE_WEIGHTS
    )
    df_ht, X_ht, scaler_ht, knn_ht, features_ht = load_cleaned_model(
        HT_DATA_PATH, ['vitalsign_sbp_0', 'vitalsign_dbp_0', 'co_dm'], HT_FEATURE_WEIGHTS
    )
    print("🚀 KNN Backend models ready!")
except Exception as e:
    print(f"❌ Error during model startup: {e}")


# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Medication Suggestion CDSS API is running successfully on Cleaned Datasets. Go to /docs",
    }


@app.post("/api/suggest", summary="แนะนำยาผ่าน Weighted KNN + Distance-Weighted Voting")
def suggest_medication(patient: PatientInput):
    data = patient.dict()
    disease = data['disease_type'].lower()

    df_query = pd.DataFrame([data])
    df_query['sex'] = (
        df_query['sex'].astype(str).str.upper()
        .map({'MALE': 1, 'FEMALE': 0}).fillna(0)
    )

    # เลือก config ตาม disease
    if disease == 'diabetes':
        df_query['co_ht'] = 0
        feature_list  = features_dm
        X_ref, scaler = X_dm, scaler_dm
        knn_model     = knn_dm
        weight_map    = DM_FEATURE_WEIGHTS
        current_df    = df_dm
        target_meds   = DM_MEDS

    elif disease == 'hypertension':
        df_query['co_dm'] = 0
        feature_list  = features_ht
        X_ref, scaler = X_ht, scaler_ht
        knn_model     = knn_ht
        weight_map    = HT_FEATURE_WEIGHTS
        current_df    = df_ht
        target_meds   = HT_MEDS

    else:  # both
        df_query['co_ht'] = 1
        feature_list  = features_dm
        X_ref, scaler = X_dm, scaler_dm
        knn_model     = knn_dm
        weight_map    = DM_FEATURE_WEIGHTS
        current_df    = df_dm
        target_meds   = DM_MEDS + ['acei', 'arb', 'ccb', 'beta_blocker', 'diuretics']

    # เตรียม query vector
    X_query = df_query[feature_list].copy()
    for col in X_query.columns:
        X_query[col] = pd.to_numeric(X_query[col], errors='coerce').fillna(X_ref[col].mean())

    X_query_scaled   = scaler.transform(X_query)
    X_query_weighted = apply_weights(X_query_scaled, feature_list, weight_map)

    distances, indices = knn_model.kneighbors(X_query_weighted)
    distances    = distances[0]
    indices      = indices[0]
    similar_cases = current_df.iloc[indices]

    # คำนวณ suggestion ด้วย distance-weighted voting
    suggestions = []
    for med in target_meds:
        excel_col = find_med_column(current_df, med)

        if excel_col and excel_col in similar_cases.columns:
            med_values = pd.to_numeric(similar_cases[excel_col], errors='coerce').fillna(0)
            weighted_rate = distance_weighted_rate(med_values, distances)
            rate_pct = round(weighted_rate * 100, 1)
        else:
            rate_pct = 0.0

        is_current = med.lower() in [m.lower() for m in data['current_medications']]

        suggestions.append({
            "medication_name":       med.upper(),
            "score":                 rate_pct,
            "recommendation":        recommendation_label(rate_pct),
            "is_current_medication": is_current,
        })

    suggestions  = sorted(suggestions, key=lambda x: x['score'], reverse=True)
    recommended  = [s for s in suggestions if s['score'] > 0]
    not_relevant = [s for s in suggestions if s['score'] == 0]

    return {
        "status":       "success",
        "disease_mode": disease,
        "match_details": {
            "neighbors_used":             len(indices),
            "matched_historical_indices": indices.tolist(),
            "calculated_distances":       [round(float(d), 4) for d in distances],
        },
        "suggestions":  recommended,
        "not_relevant": [s["medication_name"] for s in not_relevant],
    }


# -------------------------------------------------------------
# /predict — Complication Risk Prediction
# -------------------------------------------------------------
@app.post("/predict", summary="ทำนายความเสี่ยงภาวะแทรกซ้อน (CKD, Stroke, CAD)")
def predict_complications(patient: PredictInput):
    try:
        result = predictor.predict(patient.dict())
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logging.exception("Unexpected error in /predict")
        raise HTTPException(status_code=500, detail="Internal server error")


# -------------------------------------------------------------
# Serve frontend (Vite build output: frontend/dist)
# -------------------------------------------------------------
import os

_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")

if os.path.isdir(_FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")), name="assets")

    @app.get("/app", include_in_schema=False)
    @app.get("/app/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str = ""):
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))
