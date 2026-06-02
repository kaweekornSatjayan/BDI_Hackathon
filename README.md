# BDI Hackathon — Complication Risk Prediction CDSS

Clinical Decision Support System (CDSS) สำหรับทำนายความเสี่ยงภาวะแทรกซ้อน (CKD, Stroke, CAD) และแนะนำยาสำหรับผู้ป่วย Hypertension / Diabetes โดยใช้ XGBoost ML model และ KNN Case-Based Reasoning

---

## Project Structure

```
BDI_Hackathon/
├── app/
│   ├── main.py          # FastAPI routes: POST /predict, POST /api/suggest
│   ├── predictor.py     # โหลด ht_model.pkl / dm_model.pkl + predict 33 features
│   ├── schemas.py       # Pydantic input/output validation
│   ├── cbr.py           # KNN Case-Based Reasoning (medication suggestion)
│   └── config.py        # Feature lists, weights, data paths, med lists
├── frontend/
│   ├── src/
│   │   ├── pages/       # PredictPage.jsx, SuggestPage.jsx
│   │   ├── components/  # RiskCard, RiskResults, MedTable, FieldInput, etc.
│   │   └── api/         # predict.js, suggest.js (fetch wrappers)
│   └── dist/            # Production build (served at /app by FastAPI)
├── model/
│   ├── ht_model.pkl     # XGBoost bundle — Hypertension risk model
│   └── dm_model.pkl     # XGBoost bundle — Diabetes risk model (optional)
├── data/
│   └── processed/
│       ├── cleaned_hypertension.xlsx   # CBR reference cases (HT)
│       └── cleaned_diabetes.xlsx       # CBR reference cases (DM)
├── scripts/
│   ├── test_predict.py  # Black-box test: ML model prediction (4 cases)
│   ├── test_suggest.py  # Black-box test: CBR medication suggestion (4 cases)
│   └── test_api.py      # Integration + Validation + Boundary tests
├── requirements.txt
└── run.py               # Entry point: python run.py
```

---

## Web App Flow

```
User opens http://localhost:8000/app
         │
         ├──[Tab 1] 📊 Risk Prediction
         │         │
         │         ▼
         │   กรอกข้อมูลผู้ป่วย (33 features)
         │   ┌─────────────────────────────────┐
         │   │ Disease Status: HT / DM / Both  │
         │   │ Demographics: Age, Sex          │
         │   │ Blood Pressure: SBP, DBP        │
         │   │ Labs: BMI, HbA1c, FPG,          │
         │   │       Cholesterol, LDL          │
         │   │ Comorbidities: DM, Stroke, CAD, │
         │   │                CKD, Arrhythmias │
         │   │ Medications: ACEi, ARB, CCB,    │
         │   │              Diuretics, BB      │
         │   └─────────────────────────────────┘
         │         │
         │         ▼ POST /predict
         │   FastAPI → predictor.py
         │         │
         │         ├─ has_ht=True  → ht_model.pkl (XGBoost, 33 features)
         │         ├─ has_dm=True  → dm_model.pkl (XGBoost)
         │         └─ both=True    → P_combined = 1-(1-P_HT)(1-P_DM)
         │         │
         │         ▼ Response
         │   { risks: { ckd, stroke, cad: { probability, level } },
         │     explanation, model_used, timestamp }
         │         │
         │         ▼
         │   แสดงผล Risk Cards (LOW / MEDIUM / HIGH)
         │   + explanation text
         │
         └──[Tab 2] 💊 Medication Suggestion
                   │
                   ▼
             กรอกข้อมูลผู้ป่วย (demographics + labs + comorbidities)
                   │
                   ▼ POST /api/suggest
             FastAPI → cbr.py (Weighted KNN)
                   │
                   ├─ โหลด cleaned_hypertension.xlsx หรือ cleaned_diabetes.xlsx
                   ├─ MinMaxScaler + apply feature weights
                   ├─ KNN (k=10, manhattan distance) หา neighbor ที่ใกล้ที่สุด
                   └─ Distance-Weighted Voting คำนวณ score ต่อยาแต่ละตัว
                   │
                   ▼ Response
             { suggestions: [{ medication, score, recommendation }],
               disease_mode, match_details }
                   │
                   ▼
             แสดงตาราง Medication + Score bar
             (Recommended ≥60% / Consider 25–60% / Not Recommended <25%)
```

---

## Risk Thresholds

| Level | Probability |
|-------|------------|
| 🟢 LOW | < 30% |
| 🟡 MEDIUM | 30% – 59% |
| 🔴 HIGH | ≥ 60% |

---

## ML Model (ht_model.pkl) — 33 Input Features

| กลุ่ม | Features |
|-------|---------|
| Demographics | `age`, `sex` |
| Systolic BP | `vitalsign_sbp_mean`, `_std`, `_max` |
| Diastolic BP | `vitalsign_dbp_mean`, `_std`, `_max` |
| BMI | `vitalsign_bmi_mean`, `_std`, `_max` |
| HbA1c | `lab_hba1c_mean`, `_std`, `_max` |
| Fasting Glucose | `lab_fpg_mean`, `_std`, `_max` |
| Cholesterol | `lab_chol_mean`, `_std`, `_max` |
| LDL | `lab_ldl_mean`, `_std`, `_max` |
| Comorbidities | `co_dm`, `co_stroke`, `co_cad`, `co_ckd`, `co_arrhythmias` |
| Medications | `med_acei`, `med_arb`, `med_ccb`, `med_diuretics`, `med_beta_blocker` |

**Output targets:** CKD, Stroke, CAD

**Combining (HT + DM):**
$$P_{combined} = 1 - (1 - P_{HT}) \times (1 - P_{DM})$$

---

## API FOUND Endpoints

### POST /predict
**Request:**
```json
{
  "has_ht": true,
  "has_dm": false,
  "age": 65,
  "sex": "MALE",
  "sbp_mean": 155, "sbp_std": 10, "sbp_max": 170,
  "dbp_mean": 95,  "dbp_std": 6,  "dbp_max": 105,
  "bmi_mean": 28.5,
  "hba1c_mean": 7.0, "fpg_mean": 120,
  "chol_mean": 210,  "ldl_mean": 130,
  "co_dm": 0, "co_stroke": 0, "co_cad": 0, "co_ckd": 0, "co_arrhythmias": 0,
  "med_acei": 0, "med_arb": 0, "med_ccb": 0, "med_diuretics": 0, "med_beta_blocker": 0
}
```
**Response:**
```json
{
  "status": "success",
  "model_used": "ht",
  "risks": {
    "ckd":    { "probability": 0.015, "level": "LOW" },
    "stroke": { "probability": 0.012, "level": "LOW" },
    "cad":    { "probability": 0.011, "level": "LOW" }
  },
  "explanation": "Overall risk appears LOW...",
  "timestamp": "2026-06-02T09:58:05Z"
}
```

### POST /api/suggest
**Request:**
```json
{
  "disease_type": "hypertension",
  "age": 60, "sex": "MALE",
  "vitalsign_bmi_0": 27.5,
  "vitalsign_sbp_0": 150, "vitalsign_dbp_0": 92,
  "co_ckd": 1, "co_cad": 0
}
```
**Response:**
```json
{
  "status": "success",
  "disease_mode": "hypertension",
  "suggestions": [
    { "medication": "arb",       "score": 49.7, "recommendation": "Consider" },
    { "medication": "diuretics", "score": 48.9, "recommendation": "Consider" }
  ],
  "not_relevant": [...]
}
```

---

## Deployment FOUND

### Production — Cloud (Hugging Face Spaces)

| รายการ | รายละเอียด |
|--------|-----------|
| **ประเภท** | Cloud Deployment (Public) |
| **Platform** | [Hugging Face Spaces](https://huggingface.co/spaces) |
| **Runtime** | Docker container (python:3.11-slim) |
| **URL** | https://kavkorn-bdi-hackathon.hf.space |
| **Web UI** | https://kavkorn-bdi-hackathon.hf.space/app |
| **API Docs** | https://kavkorn-bdi-hackathon.hf.space/docs |
| **Hardware** | CPU Basic (ฟรี) |

**สิ่งที่ deploy บน cloud:**
- FastAPI backend (Python 3.11)
- React + Vite frontend (pre-built แล้ว serve เป็น static files)
- XGBoost model (`ht_model.pkl`) อยู่ใน container
- CBR reference data (`cleaned_hypertension.xlsx`, `cleaned_diabetes.xlsx`) อยู่ใน container

**หมายเหตุ:** ไม่ใช่ On-Premise — ไม่ต้องมี server ของตัวเอง ทุกอย่างรันบน infrastructure ของ Hugging Face

---

### Local (On-Premise / Development)

รันในเครื่องเองได้เช่นกัน เหมาะสำหรับพัฒนาหรือใช้ภายในองค์กร:

---

## Setup & Run

### Requirements
- Python 3.10+
- Node.js 18+

### 1. Clone & Install
```bash
git clone https://github.com/kaweekornSatjayan/BDI_Hackathon.git
cd BDI_Hackathon/BDI_Hackathon
pip install -r requirements.txt
```

### 2. Build Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Run Server
```bash
python run.py
```
Server starts at **http://localhost:8000**

| URL | Description |
|-----|-------------|
| http://localhost:8000/app | Web UI |
| http://localhost:8000/docs | Swagger API docs |

### Frontend Dev Mode (hot-reload)
```bash
cd frontend
npm run dev   # http://localhost:5173
```
ต้องรัน backend ที่ port 8000 ก่อน (Vite proxy `/predict` และ `/api` อัตโนมัติ)

---

## Why These Models Were Chosen

### XGBoost — Risk Prediction

XGBoost was selected as the core prediction engine for the following reasons:

| Reason | Detail |
|--------|--------|
| **Best-in-class on tabular data** | XGBoost consistently outperforms other algorithms on structured/tabular medical records — the exact format used in this system |
| **Handles mixed feature types** | Naturally handles both continuous vitals (SBP, BMI) and binary flags (comorbidities, medications) without requiring separate preprocessing |
| **Robust to small datasets** | Built-in L1/L2 regularization reduces overfitting, which is critical when clinical datasets are limited in size |
| **Interpretable feature importance** | Generates feature importance scores, helping clinicians understand *why* a patient is classified as HIGH risk — essential for building trust in medical AI |
| **Multi-output support** | Predicts HT risk, DM risk, and CAD risk simultaneously through separate trained estimators without needing separate pipelines |
| **Proven in clinical ML** | XGBoost is widely validated in healthcare prediction tasks (readmission, mortality, disease progression) and is the most cited baseline in clinical ML literature |

### Case-Based Reasoning (KNN) — Medication Suggestion

CBR/KNN was chosen for medication suggestion because:

| Reason | Detail |
|--------|--------|
| **Mirrors clinical reasoning** | Physicians naturally reason by analogy — "this patient resembles past case X, so a similar treatment may work" — CBR formalizes this approach |
| **Evidence-based and transparent** | Suggestions come directly from real historical patient outcomes, not from an opaque black-box model |
| **Auditable** | The system can show *which* past cases were referenced, making the suggestion explainable and acceptable to clinicians |
| **No retraining required** | Adding new patient cases immediately improves future suggestions without retraining |
| **Performs well with limited data** | KNN works effectively even with small training sets — an advantage over neural approaches that require large datasets |

---

## Why These Features Were Chosen

All numeric features are derived from **longitudinal clinical records** using three aggregations — **mean, std, and max** — per measurement. This captures trends across multiple visits rather than relying on a single snapshot, which is more representative of chronic disease progression.

### Demographics

| Feature | Rationale |
|---------|-----------|
| `age` | The strongest non-modifiable risk factor for HT, DM, and CAD. Risk rises sharply after age 55. |
| `sex` | Males have higher baseline cardiovascular risk; females face elevated risk post-menopause. Sex affects treatment thresholds in clinical guidelines. |

### Blood Pressure (SBP / DBP)

| Feature | Rationale |
|---------|-----------|
| `sbp_mean`, `sbp_std`, `sbp_max` | Systolic BP is the primary diagnostic criterion for hypertension. Mean reflects sustained vascular load; std detects variability (an independent risk factor); max captures dangerous peaks. |
| `dbp_mean`, `dbp_std`, `dbp_max` | Diastolic BP contributes to isolated or combined hypertension classification and is strongly linked to CAD risk in younger patients. |

### Metabolic Markers

| Feature | Rationale |
|---------|-----------|
| `bmi_mean`, `bmi_std`, `bmi_max` | Obesity (BMI ≥ 30) is a root cause of insulin resistance, hypertension, and dyslipidemia — affecting all three target diseases. |
| `hba1c_mean`, `hba1c_std`, `hba1c_max` | HbA1c reflects average blood glucose over ~3 months. It is the gold standard for DM diagnosis (≥ 6.5%) and long-term glycemic control monitoring. |
| `fpg_mean`, `fpg_std`, `fpg_max` | Fasting Plasma Glucose is the primary DM screening test. Elevated FPG signals impaired glucose metabolism even before clinical DM onset. |

### Lipid Panel

| Feature | Rationale |
|---------|-----------|
| `chol_mean`, `chol_std`, `chol_max` | Total cholesterol is a key cardiovascular risk marker. Sustained high levels drive atherosclerosis and CAD. |
| `ldl_mean`, `ldl_std`, `ldl_max` | LDL ("bad" cholesterol) directly causes plaque formation in coronary arteries. Lowering LDL is the primary CAD prevention target in all major clinical guidelines. |

### Comorbidities

| Feature | Rationale |
|---------|-----------|
| `co_dm` | Diabetes doubles cardiovascular risk and causes vessel and kidney damage, directly elevating HT and CAD risk. |
| `co_stroke` | History of stroke signals existing cerebrovascular disease — strongly correlated with future cardiac events. |
| `co_cad` | Pre-existing coronary artery disease is the strongest predictor of future CAD-related complications. |
| `co_ckd` | Chronic kidney disease elevates blood pressure through fluid/electrolyte dysregulation and accelerates cardiovascular disease progression. |
| `co_arrhythmias` | Arrhythmias (e.g., atrial fibrillation) increase clot risk and are independently associated with both HT and CAD. |

### Current Medications

Including current medications allows the model to distinguish between *controlled* and *uncontrolled* risk — a patient on 4 antihypertensives with still-high BP represents very different clinical risk than an untreated patient with the same BP values.

| Feature | Rationale |
|---------|-----------|
| `med_acei` | ACE inhibitors lower BP and protect kidneys — their presence signals how aggressively HT is being managed. |
| `med_arb` | ARBs serve the same role as ACEi and are used when ACEi cause side effects (e.g., dry cough). |
| `med_ccb` | Calcium channel blockers reduce arterial stiffness — key in HT and CAD management. |
| `med_diuretics` | Diuretics reduce fluid volume and BP — first-line HT treatment per JNC/ESC guidelines. |
| `med_beta_blocker` | Beta-blockers slow heart rate and reduce cardiac workload — standard treatment in both CAD and HT. |

---

## Testing

```bash
# ML model prediction (4 cases — no server needed)
python scripts/test_predict.py

# Medication suggestion CBR (4 cases — no server needed)
python scripts/test_suggest.py

# Integration + Validation + Boundary (server optional)
python scripts/test_api.py

# Run all with pytest
pytest scripts/ -v
```

**Test coverage:**

| Script | Type | Cases |
|--------|------|-------|
| `test_predict.py` | Black-box (ML output) | HIGH-risk, LOW-risk, MEDIUM-risk, Edge (no std/max) |
| `test_suggest.py` | Black-box (CBR output) | HT+CKD, HT+CAD, DM high HbA1c, DM+CKD |
| `test_api.py` | Integration + Validation + Boundary | HTTP response shape, Pydantic errors, risk_level thresholds |

---

## Notes

- `dm_model.pkl` ยังไม่ได้รวมใน repo — server รันได้แต่ DM prediction ไม่พร้อมใช้
- หากไม่ใส่ `sbp_std`/`sbp_max` ระบบจะ fallback เป็น 0 / mean อัตโนมัติ
- `sex` รับค่า `"MALE"` หรือ `"FEMALE"` (case-insensitive)

