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

## API Endpoints

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

## Deployment

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

