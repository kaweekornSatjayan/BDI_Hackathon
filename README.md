# BDI Hackathon — Complication Risk Prediction CDSS

Clinical Decision Support System ที่ทำนายความเสี่ยงภาวะแทรกซ้อน (CKD, Stroke, CAD) ในผู้ป่วย Hypertension และ/หรือ Diabetes โดยใช้ XGBoost ML models และแนะนำยาผ่าน KNN Case-Based Reasoning

---

## Project Structure

```
BDI_Hackathon/
├── app/                    # FastAPI backend
│   ├── main.py             # Routes: /predict, /api/suggest
│   ├── predictor.py        # ML model loading + prediction logic
│   ├── schemas.py          # Pydantic input/output models
│   ├── cbr.py              # KNN Case-Based Reasoning (medication suggestion)
│   └── config.py           # Constants, feature lists, data paths
├── frontend/               # React + Vite frontend
│   ├── src/
│   │   ├── pages/          # PredictPage.jsx
│   │   ├── components/     # RiskCard, RiskResults, FieldInput, DiseaseToggle, Card
│   │   └── api/            # predict.js (fetch wrapper)
│   ├── dist/               # Production build (served by FastAPI at /app)
│   └── package.json
├── model/
│   ├── ht_model.pkl        # XGBoost bundle for Hypertension patients
│   └── dm_model.pkl        # XGBoost bundle for Diabetes patients (required)
├── data/
│   ├── raw/                # Original Excel files
│   └── processed/          # Cleaned data (used by CBR)
├── scripts/
│   └── clean_data.py       # Data cleaning script
├── requirements.txt
└── run.py                  # Server entry point
```

---

## Features

| Feature | Description |
|---------|-------------|
| `POST /predict` | ทำนายความเสี่ยง CKD, Stroke, CAD จาก 14 clinical features |
| `POST /api/suggest` | แนะนำยาผ่าน Weighted KNN + Distance-Weighted Voting |
| `GET /app` | React web UI |
| `GET /docs` | FastAPI Swagger UI |

---

## ML Model Details

**Input features (14):**

| Feature | Description |
|---------|-------------|
| `age` | อายุ (ปี) |
| `sex` | เพศ (MALE / FEMALE) |
| `sbp_mean`, `sbp_std`, `sbp_max` | Systolic BP aggregations (mmHg) |
| `dbp_mean`, `dbp_std`, `dbp_max` | Diastolic BP aggregations (mmHg) |
| `hba1c_mean`, `hba1c_std`, `hba1c_max` | HbA1c aggregations (%) |
| `bmi_mean` | BMI (kg/m²) |
| `fpg_mean` | Fasting Plasma Glucose (mg/dL) |
| `hemoglobin_mean` | Hemoglobin (g/dL) |

**Output (3 targets):**

| Target | Threshold |
|--------|-----------|
| CKD (Chronic Kidney Disease) | LOW <30% / MEDIUM 30–60% / HIGH ≥60% |
| Stroke | LOW <30% / MEDIUM 30–60% / HIGH ≥60% |
| CAD (Coronary Artery Disease) | LOW <30% / MEDIUM 30–60% / HIGH ≥60% |

**Combining formula (HT + DM):**
```
P_combined = 1 - (1 - P_HT) × (1 - P_DM)
```

---

## Setup

### Requirements
- Python 3.10+
- Node.js 18+

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install & build frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Run server
```bash
python run.py
```

Server starts at **http://localhost:8000**

---

## Usage

### Web UI
Open **http://localhost:8000/app** in browser

### API (example)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "has_ht": true,
    "has_dm": false,
    "age": 65,
    "sex": "MALE",
    "sbp_mean": 155,
    "sbp_std": 10,
    "sbp_max": 170,
    "dbp_mean": 95,
    "dbp_std": 6,
    "dbp_max": 105,
    "bmi_mean": 28.5
  }'
```

### Frontend dev mode (hot-reload)
```bash
cd frontend
npm run dev   # http://localhost:5173
```
Requires backend running on port 8000 (Vite proxies `/predict` and `/api` automatically).

---

## Testing

```bash
# Run ML model tests
python scripts/test_ml.py

# Run all tests (if pytest installed)
pytest scripts/test_ml.py -v
```

---

## Notes

- `dm_model.pkl` ต้องวางในโฟลเดอร์ `model/` ก่อนใช้งาน DM prediction
- หากไม่มี `dm_model.pkl` server จะยังรันได้ แต่จะ error เมื่อเลือก DM หรือ HT+DM
- ข้อมูล CBR อยู่ที่ `data/processed/cleaned_diabetes.xlsx` และ `cleaned_hypertension.xlsx`
