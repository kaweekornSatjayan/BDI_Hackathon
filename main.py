import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

app = FastAPI(
    title="Medication Suggestion CDSS API (Production Ready)",
    description="Medical Decision Support System using Case-Based Reasoning (CBR) with Cleaned Datasets."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Medication Suggestion CDSS API is running successfully on Cleaned Datasets. Go to /docs"
    }

# -------------------------------------------------------------
# [1] Feature lists & Medication targets
# -------------------------------------------------------------
features_base        = ['age', 'sex', 'vitalsign_bmi_0']
features_comorbidity = ['co_ckd', 'co_hf', 'co_cad', 'co_stroke',
                        'co_arrhythmias', 'co_atrial_fibrillation', 'co_dementia']

DM_MEDS = [
    'metformin', 'insulin', 'gliclazide', 'glipizide', 'glimepiride', 'glibenclamide',
    'empagliflozin', 'dapagliflozin', 'sitagliptin', 'linagliptin', 'gemigliptin',
    'trelagliptin', 'pioglitazone', 'acarbose', 'dulaglutide', 'liraglutide', 'semaglutide'
]
HT_MEDS = [
    'acei', 'arb', 'ccb', 'beta_blocker', 'diuretics', 'hydralazine',
    'neprilysin_inhibitor', 'alpha_blocker', 'alpha2_agonist', 'alpha_beta_blocker'
]

# -------------------------------------------------------------
# [2] Feature weights — lab values & comorbidities สำคัญกว่า age/BMI
# -------------------------------------------------------------
DM_FEATURE_WEIGHTS = {
    'age':                    0.5,
    'sex':                    0.3,
    'vitalsign_bmi_0':        0.7,
    'co_ckd':                 1.8,
    'co_hf':                  1.8,
    'co_cad':                 1.5,
    'co_stroke':              1.2,
    'co_arrhythmias':         1.0,
    'co_atrial_fibrillation': 1.0,
    'co_dementia':            0.8,
    'lab_hba1c_0':            2.5,   # สำคัญที่สุดสำหรับ DM
    'lab_fpg_0':              2.0,
    'co_ht':                  1.2,
}

HT_FEATURE_WEIGHTS = {
    'age':                    0.5,
    'sex':                    0.3,
    'vitalsign_bmi_0':        0.7,
    'co_ckd':                 1.8,
    'co_hf':                  1.8,
    'co_cad':                 1.5,
    'co_stroke':              1.2,
    'co_arrhythmias':         1.0,
    'co_atrial_fibrillation': 1.0,
    'co_dementia':            0.8,
    'vitalsign_sbp_0':        2.5,   # สำคัญที่สุดสำหรับ HT
    'vitalsign_dbp_0':        2.0,
    'co_dm':                  1.2,
}

def apply_weights(X_scaled: np.ndarray, feature_list: list, weight_map: dict) -> np.ndarray:
    """คูณ weight กับ scaled features ก่อนส่งเข้า KNN"""
    weights = np.array([weight_map.get(f, 1.0) for f in feature_list])
    return X_scaled * weights

# -------------------------------------------------------------
# [3] Load & fit model
# -------------------------------------------------------------
def load_cleaned_model(file_path: str, specific_features: list, weight_map: dict):
    df = pd.read_excel(file_path)
    all_features = features_base + features_comorbidity + specific_features

    X = df[all_features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_weighted = apply_weights(X_scaled, all_features, weight_map)

    # ใช้ n_neighbors = min(10, len(df)) เพื่อรองรับ dataset เล็ก
    k = min(10, len(df))
    knn = NearestNeighbors(n_neighbors=k, metric='manhattan')
    knn.fit(X_weighted)

    return df, X, scaler, knn, all_features

print("🔄 Loading professionally cleaned datasets into KNN models...")
try:
    df_dm, X_dm, scaler_dm, knn_dm, features_dm = load_cleaned_model(
        "cleaned_diabetes.xlsx", ['lab_hba1c_0', 'lab_fpg_0', 'co_ht'], DM_FEATURE_WEIGHTS
    )
    df_ht, X_ht, scaler_ht, knn_ht, features_ht = load_cleaned_model(
        "cleaned_hypertension.xlsx", ['vitalsign_sbp_0', 'vitalsign_dbp_0', 'co_dm'], HT_FEATURE_WEIGHTS
    )
    print("🚀 KNN Backend models ready!")
except Exception as e:
    print(f"❌ Error during model startup: {e}")


# -------------------------------------------------------------
# [4] Patient input schema
# -------------------------------------------------------------
class PatientInput(BaseModel):
    disease_type: str
    age: float
    sex: str
    vitalsign_bmi_0: float
    lab_hba1c_0: float = 0.0
    lab_fpg_0: float = 0.0
    vitalsign_sbp_0: float = 0.0
    vitalsign_dbp_0: float = 0.0
    co_ckd: int = 0
    co_hf: int = 0
    co_cad: int = 0
    co_stroke: int = 0
    co_arrhythmias: int = 0
    co_atrial_fibrillation: int = 0
    co_dementia: int = 0
    current_medications: list = []


# -------------------------------------------------------------
# [5] Helper: find medication column
# -------------------------------------------------------------
def find_med_column(df: pd.DataFrame, med_name: str) -> str | None:
    exact = f"med_{med_name}"
    if exact in df.columns:
        return exact
    for col in df.columns:
        if med_name.lower() in col.lower():
            return col
    return None


# -------------------------------------------------------------
# [6] Distance-weighted vote — neighbor ใกล้กว่า = น้ำหนักมากกว่า
# -------------------------------------------------------------
def distance_weighted_rate(med_values: pd.Series, distances: np.ndarray) -> float:
    """
    คำนวณ weighted mean โดยใช้ inverse-distance เป็น weight
    ถ้า distance = 0 (exact match) ให้ weight สูงสุด
    """
    eps = 1e-6  # ป้องกัน division by zero
    weights = 1.0 / (distances + eps)
    weights = weights / weights.sum()  # normalize
    values = med_values.values.astype(float)
    return float(np.dot(weights, values))


# -------------------------------------------------------------
# [7] Core suggest endpoint
# -------------------------------------------------------------
@app.post("/api/suggest", summary="แนะนำยาผ่าน Weighted KNN + Distance-Weighted Voting")
def suggest_medication(patient: PatientInput):
    data = patient.dict()
    disease = data['disease_type'].lower()

    df_query = pd.DataFrame([data])
    df_query['sex'] = (
        df_query['sex'].astype(str).str.upper()
        .map({'MALE': 1, 'FEMALE': 0}).fillna(0)
    )

    # --- เลือก config ตาม disease ---
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

    # --- เตรียม query vector ---
    X_query = df_query[feature_list].copy()
    for col in X_query.columns:
        X_query[col] = pd.to_numeric(X_query[col], errors='coerce').fillna(X_ref[col].mean())

    X_query_scaled   = scaler.transform(X_query)
    X_query_weighted = apply_weights(X_query_scaled, feature_list, weight_map)

    distances, indices = knn_model.kneighbors(X_query_weighted)
    distances = distances[0]
    indices   = indices[0]
    similar_cases = current_df.iloc[indices]

    # --- คำนวณ suggestion ด้วย distance-weighted voting ---
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
            "medication_name":      med.upper(),
            "score":                rate_pct,           # weighted score 0–100
            "recommendation":       _label(rate_pct),  # Recommended / Consider / Not Recommended
            "is_current_medication": is_current,
        })

    # เรียงจาก score สูงสุด และตัดยาที่ score = 0 ออก (ไม่ recommend เลย)
    suggestions = sorted(suggestions, key=lambda x: x['score'], reverse=True)
    recommended  = [s for s in suggestions if s['score'] > 0]
    not_relevant = [s for s in suggestions if s['score'] == 0]

    return {
        "status":       "success",
        "disease_mode": disease,
        "match_details": {
            "neighbors_used":            len(indices),
            "matched_historical_indices": indices.tolist(),
            "calculated_distances":       [round(float(d), 4) for d in distances],
        },
        "suggestions":   recommended,
        "not_relevant":  [s["medication_name"] for s in not_relevant],
    }


def _label(score: float) -> str:
    if score >= 60:  return "Recommended"
    if score >= 25:  return "Consider"
    return "Not Recommended"