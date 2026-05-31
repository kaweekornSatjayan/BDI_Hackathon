import { useState } from 'react'
import { callPredict } from '../api/predict'
import Card from '../components/Card'
import DiseaseToggle from '../components/DiseaseToggle'
import FieldInput from '../components/FieldInput'
import RiskResults from '../components/RiskResults'
import styles from './PredictPage.module.css'

const INITIAL_FORM = {
  // disease flags
  diseases: { ht: false, dm: false },

  // demographics
  age: '',
  sex: '',

  // blood pressure
  sbp_mean: '', sbp_std: '', sbp_max: '',
  dbp_mean: '', dbp_std: '', dbp_max: '',

  // labs
  hba1c_mean: '', hba1c_std: '', hba1c_max: '',
  bmi_mean:   '',
  fpg_mean:   '',
  hemoglobin_mean: '',
}

function buildPayload(form) {
  const numOrNull = (v) => (v === '' || v === undefined ? null : Number(v))

  // If only mean is provided, fill std=0 and max=mean for BP/HbA1c
  const sbpStd = form.sbp_std !== '' ? Number(form.sbp_std) : 0
  const sbpMax = form.sbp_max !== '' ? Number(form.sbp_max) : Number(form.sbp_mean)
  const dbpStd = form.dbp_std !== '' ? Number(form.dbp_std) : 0
  const dbpMax = form.dbp_max !== '' ? Number(form.dbp_max) : Number(form.dbp_mean)

  return {
    has_ht: form.diseases.ht,
    has_dm: form.diseases.dm,
    age:    Number(form.age),
    sex:    form.sex,

    sbp_mean: Number(form.sbp_mean),
    sbp_std:  sbpStd,
    sbp_max:  sbpMax,

    dbp_mean: Number(form.dbp_mean),
    dbp_std:  dbpStd,
    dbp_max:  dbpMax,

    hba1c_mean: numOrNull(form.hba1c_mean),
    hba1c_std:  numOrNull(form.hba1c_std),
    hba1c_max:  numOrNull(form.hba1c_max),

    bmi_mean:        Number(form.bmi_mean),
    fpg_mean:        numOrNull(form.fpg_mean),
    hemoglobin_mean: numOrNull(form.hemoglobin_mean),
  }
}

function validate(form) {
  const errs = {}
  if (!form.diseases.ht && !form.diseases.dm) errs.diseases = true
  if (!form.age)      errs.age      = true
  if (!form.sex)      errs.sex      = true
  if (!form.sbp_mean) errs.sbp_mean = true
  if (!form.dbp_mean) errs.dbp_mean = true
  if (!form.bmi_mean) errs.bmi_mean = true
  return errs
}

export default function PredictPage() {
  const [form,    setForm]    = useState(INITIAL_FORM)
  const [errors,  setErrors]  = useState({})
  const [loading, setLoading] = useState(false)
  const [apiErr,  setApiErr]  = useState(null)
  const [result,  setResult]  = useState(null)

  function setField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }))
    setErrors((prev) => ({ ...prev, [key]: false }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setApiErr(null)

    const errs = validate(form)
    if (Object.keys(errs).length) {
      setErrors(errs)
      return
    }

    setLoading(true)
    try {
      const data = await callPredict(buildPayload(form))
      setResult(data)
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
    } catch (err) {
      setApiErr(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>🏥 Complication Risk Predictor</h1>
        <p className={styles.subtitle}>
          Predicts CKD, Stroke &amp; CAD risk using clinical ML models (XGBoost)
        </p>
      </header>

      <form onSubmit={handleSubmit} noValidate>

        {/* ── Disease Status ───────────────────────────── */}
        <Card title="Disease Status">
          <DiseaseToggle
            selected={form.diseases}
            onChange={(d) => setField('diseases', d)}
            error={errors.diseases}
          />
        </Card>

        {/* ── Demographics ─────────────────────────────── */}
        <Card title="Demographics">
          <div className={styles.grid2}>
            <FieldInput
              id="age" label="Age" unit="years"
              placeholder="e.g. 65" min={18} max={120}
              value={form.age} onChange={(v) => setField('age', v)}
              error={errors.age}
            />
            <FieldInput
              id="sex" label="Sex"
              type="select"
              options={[{ value: 'MALE', label: 'Male' }, { value: 'FEMALE', label: 'Female' }]}
              value={form.sex} onChange={(v) => setField('sex', v)}
              error={errors.sex}
            />
          </div>
        </Card>

        {/* ── Blood Pressure ───────────────────────────── */}
        <Card title="Blood Pressure">
          <p className={styles.sectionHint}>
            If you have only one reading, enter it in Mean — Std and Max will be auto-filled.
          </p>

          <p className={styles.subLabel}>Systolic BP (mmHg)</p>
          <div className={`${styles.grid3} ${styles.mb}`}>
            <FieldInput id="sbp_mean" label="Mean *" placeholder="e.g. 140"
              min={90} max={250} value={form.sbp_mean}
              onChange={(v) => setField('sbp_mean', v)} error={errors.sbp_mean} />
            <FieldInput id="sbp_std"  label="Std"    placeholder="e.g. 10"
              min={0} value={form.sbp_std}
              onChange={(v) => setField('sbp_std', v)} />
            <FieldInput id="sbp_max"  label="Max"    placeholder="auto"
              min={0} value={form.sbp_max}
              onChange={(v) => setField('sbp_max', v)} />
          </div>

          <p className={styles.subLabel}>Diastolic BP (mmHg)</p>
          <div className={styles.grid3}>
            <FieldInput id="dbp_mean" label="Mean *" placeholder="e.g. 90"
              min={40} max={150} value={form.dbp_mean}
              onChange={(v) => setField('dbp_mean', v)} error={errors.dbp_mean} />
            <FieldInput id="dbp_std"  label="Std"    placeholder="e.g. 6"
              min={0} value={form.dbp_std}
              onChange={(v) => setField('dbp_std', v)} />
            <FieldInput id="dbp_max"  label="Max"    placeholder="auto"
              min={0} value={form.dbp_max}
              onChange={(v) => setField('dbp_max', v)} />
          </div>
        </Card>

        {/* ── Laboratory Results ───────────────────────── */}
        <Card title="Laboratory Results">
          <p className={styles.sectionHint}>
            If you have only one reading, enter it in Mean.
          </p>

          <p className={styles.subLabel}>HbA1c (%)</p>
          <div className={`${styles.grid3} ${styles.mb}`}>
            <FieldInput id="hba1c_mean" label="Mean" placeholder="e.g. 7.2"
              min={0} step={0.1} value={form.hba1c_mean}
              onChange={(v) => setField('hba1c_mean', v)} />
            <FieldInput id="hba1c_std"  label="Std"  placeholder="e.g. 0.5"
              min={0} step={0.1} value={form.hba1c_std}
              onChange={(v) => setField('hba1c_std', v)} />
            <FieldInput id="hba1c_max"  label="Max"  placeholder="auto"
              min={0} step={0.1} value={form.hba1c_max}
              onChange={(v) => setField('hba1c_max', v)} />
          </div>

          <div className={styles.grid3}>
            <FieldInput id="bmi_mean" label="BMI *" unit="kg/m²"
              placeholder="e.g. 27.5" min={10} max={70} step={0.1}
              value={form.bmi_mean} onChange={(v) => setField('bmi_mean', v)}
              error={errors.bmi_mean} />
            <FieldInput id="fpg_mean" label="Fasting Plasma Glucose" unit="mg/dL"
              placeholder="e.g. 110" min={0}
              value={form.fpg_mean} onChange={(v) => setField('fpg_mean', v)} />
            <FieldInput id="hemoglobin_mean" label="Hemoglobin" unit="g/dL"
              placeholder="e.g. 13.5" min={0} step={0.1}
              value={form.hemoglobin_mean} onChange={(v) => setField('hemoglobin_mean', v)} />
          </div>
        </Card>

        {/* ── Submit ───────────────────────────────────── */}
        <button type="submit" className={styles.submitBtn} disabled={loading}>
          {loading ? 'Analyzing…' : 'Predict Complication Risk'}
        </button>

        {loading && (
          <div className={styles.spinner}>
            <div className={styles.spinnerDot} />
            Analyzing patient data…
          </div>
        )}

        {apiErr && (
          <div className={styles.errorBanner}>{apiErr}</div>
        )}
      </form>

      {/* ── Results ──────────────────────────────────── */}
      {result && (
        <Card title="Prediction Results">
          <RiskResults result={result} />
        </Card>
      )}
    </div>
  )
}
