import { useState } from 'react'
import { callPredict } from '../api/predict'
import Card from '../components/Card'
import DiseaseToggle from '../components/DiseaseToggle'
import FieldInput from '../components/FieldInput'
import RiskResults from '../components/RiskResults'
import styles from './PredictPage.module.css'

const COMORBIDITIES = [
  { key: 'co_dm',          label: 'Diabetes (DM)' },
  { key: 'co_stroke',      label: 'Stroke' },
  { key: 'co_cad',         label: 'CAD' },
  { key: 'co_ckd',         label: 'CKD' },
  { key: 'co_arrhythmias', label: 'Arrhythmias' },
]

const HT_MEDS = [
  { key: 'med_acei',         label: 'ACEi' },
  { key: 'med_arb',          label: 'ARB' },
  { key: 'med_ccb',          label: 'CCB' },
  { key: 'med_diuretics',    label: 'Diuretics' },
  { key: 'med_beta_blocker', label: 'Beta-blocker' },
]

const INITIAL_FORM = {
  diseases: { ht: false, dm: false },
  age: '', sex: '',
  sbp_mean: '', sbp_std: '', sbp_max: '',
  dbp_mean: '', dbp_std: '', dbp_max: '',
  bmi_mean: '', bmi_std: '', bmi_max: '',
  hba1c_mean: '', hba1c_std: '', hba1c_max: '',
  fpg_mean: '', fpg_std: '', fpg_max: '',
  chol_mean: '', chol_std: '', chol_max: '',
  ldl_mean: '', ldl_std: '', ldl_max: '',
  co_dm: false, co_stroke: false, co_cad: false, co_ckd: false, co_arrhythmias: false,
  med_acei: false, med_arb: false, med_ccb: false, med_diuretics: false, med_beta_blocker: false,
}

function buildPayload(form) {
  const n    = (v) => (v === '' || v == null ? null : Number(v))
  const nReq = (v) => Number(v) || 0
  const tri  = (mean, std, max) => ({
    mean: nReq(mean),
    std:  std !== '' ? Number(std) : 0,
    max:  max !== '' ? Number(max) : nReq(mean),
  })

  const sbp  = tri(form.sbp_mean,  form.sbp_std,  form.sbp_max)
  const dbp  = tri(form.dbp_mean,  form.dbp_std,  form.dbp_max)
  const bmi  = tri(form.bmi_mean,  form.bmi_std,  form.bmi_max)
  const hba1c = { mean: n(form.hba1c_mean), std: n(form.hba1c_std), max: n(form.hba1c_max) }
  const fpg   = { mean: n(form.fpg_mean),   std: n(form.fpg_std),   max: n(form.fpg_max)   }
  const chol  = { mean: n(form.chol_mean),  std: n(form.chol_std),  max: n(form.chol_max)  }
  const ldl   = { mean: n(form.ldl_mean),   std: n(form.ldl_std),   max: n(form.ldl_max)   }

  return {
    has_ht: form.diseases.ht,
    has_dm: form.diseases.dm,
    age: nReq(form.age),
    sex: form.sex,
    sbp_mean: sbp.mean, sbp_std: sbp.std, sbp_max: sbp.max,
    dbp_mean: dbp.mean, dbp_std: dbp.std, dbp_max: dbp.max,
    bmi_mean: bmi.mean, bmi_std: bmi.std, bmi_max: bmi.max,
    hba1c_mean: hba1c.mean, hba1c_std: hba1c.std, hba1c_max: hba1c.max,
    fpg_mean:   fpg.mean,   fpg_std:   fpg.std,   fpg_max:   fpg.max,
    chol_mean:  chol.mean,  chol_std:  chol.std,  chol_max:  chol.max,
    ldl_mean:   ldl.mean,   ldl_std:   ldl.std,   ldl_max:   ldl.max,
    co_dm:          form.co_dm          ? 1 : 0,
    co_stroke:      form.co_stroke      ? 1 : 0,
    co_cad:         form.co_cad         ? 1 : 0,
    co_ckd:         form.co_ckd         ? 1 : 0,
    co_arrhythmias: form.co_arrhythmias ? 1 : 0,
    med_acei:         form.med_acei         ? 1 : 0,
    med_arb:          form.med_arb          ? 1 : 0,
    med_ccb:          form.med_ccb          ? 1 : 0,
    med_diuretics:    form.med_diuretics    ? 1 : 0,
    med_beta_blocker: form.med_beta_blocker ? 1 : 0,
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
    if (Object.keys(errs).length) { setErrors(errs); return }
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
          Predicts CKD, Stroke &amp; CAD risk using XGBoost (33 clinical features)
        </p>
      </header>

      <form onSubmit={handleSubmit} noValidate>

        {/* ── Disease Status ─────────────────────────── */}
        <Card title="Disease Status">
          <DiseaseToggle
            selected={form.diseases}
            onChange={(d) => setField('diseases', d)}
            error={errors.diseases}
          />
          {errors.diseases && (
            <p className={styles.fieldErr}>Please select at least one disease</p>
          )}
        </Card>

        {/* ── Demographics ───────────────────────────── */}
        <Card title="Demographics">
          <div className={styles.grid2}>
            <FieldInput id="age" label="Age *" unit="years" placeholder="e.g. 65"
              min={18} max={120} value={form.age}
              onChange={(v) => setField('age', v)} error={errors.age} />
            <FieldInput id="sex" label="Sex *" type="select"
              options={[{ value: 'MALE', label: 'Male' }, { value: 'FEMALE', label: 'Female' }]}
              value={form.sex} onChange={(v) => setField('sex', v)} error={errors.sex} />
          </div>
        </Card>

        {/* ── Blood Pressure ─────────────────────────── */}
        <Card title="Blood Pressure">
          <p className={styles.subLabel}>Systolic BP — SBP (mmHg) *</p>
          <div className={`${styles.grid3} ${styles.mb}`}>
            <FieldInput id="sbp_mean" label="Mean *" placeholder="e.g. 140"
              value={form.sbp_mean} onChange={(v) => setField('sbp_mean', v)} error={errors.sbp_mean} />
            <FieldInput id="sbp_std" label="Std" placeholder="e.g. 10 (optional)"
              min={0} value={form.sbp_std} onChange={(v) => setField('sbp_std', v)} />
            <FieldInput id="sbp_max" label="Max" placeholder="auto = mean"
              min={0} value={form.sbp_max} onChange={(v) => setField('sbp_max', v)} />
          </div>

          <p className={styles.subLabel}>Diastolic BP — DBP (mmHg) *</p>
          <div className={styles.grid3}>
            <FieldInput id="dbp_mean" label="Mean *" placeholder="e.g. 85"
              value={form.dbp_mean} onChange={(v) => setField('dbp_mean', v)} error={errors.dbp_mean} />
            <FieldInput id="dbp_std" label="Std" placeholder="e.g. 6 (optional)"
              min={0} value={form.dbp_std} onChange={(v) => setField('dbp_std', v)} />
            <FieldInput id="dbp_max" label="Max" placeholder="auto = mean"
              min={0} value={form.dbp_max} onChange={(v) => setField('dbp_max', v)} />
          </div>
        </Card>

        {/* ── Labs & Body ────────────────────────────── */}
        <Card title="Labs & Body Measurements">
          <p className={styles.subLabel}>BMI (kg/m²) *</p>
          <div className={`${styles.grid3} ${styles.mb}`}>
            <FieldInput id="bmi_mean" label="Mean *" placeholder="e.g. 27.5"
              min={10} max={70} step={0.1} value={form.bmi_mean}
              onChange={(v) => setField('bmi_mean', v)} error={errors.bmi_mean} />
            <FieldInput id="bmi_std" label="Std" placeholder="optional"
              min={0} step={0.1} value={form.bmi_std} onChange={(v) => setField('bmi_std', v)} />
            <FieldInput id="bmi_max" label="Max" placeholder="auto = mean"
              min={0} step={0.1} value={form.bmi_max} onChange={(v) => setField('bmi_max', v)} />
          </div>

          <p className={styles.subLabel}>HbA1c (%)</p>
          <div className={`${styles.grid3} ${styles.mb}`}>
            <FieldInput id="hba1c_mean" label="Mean" placeholder="e.g. 7.2"
              min={0} step={0.1} value={form.hba1c_mean} onChange={(v) => setField('hba1c_mean', v)} />
            <FieldInput id="hba1c_std" label="Std" placeholder="optional"
              min={0} step={0.1} value={form.hba1c_std} onChange={(v) => setField('hba1c_std', v)} />
            <FieldInput id="hba1c_max" label="Max" placeholder="auto = mean"
              min={0} step={0.1} value={form.hba1c_max} onChange={(v) => setField('hba1c_max', v)} />
          </div>

          <p className={styles.subLabel}>Fasting Plasma Glucose — FPG (mg/dL)</p>
          <div className={`${styles.grid3} ${styles.mb}`}>
            <FieldInput id="fpg_mean" label="Mean" placeholder="e.g. 110"
              min={0} value={form.fpg_mean} onChange={(v) => setField('fpg_mean', v)} />
            <FieldInput id="fpg_std" label="Std" placeholder="optional"
              min={0} value={form.fpg_std} onChange={(v) => setField('fpg_std', v)} />
            <FieldInput id="fpg_max" label="Max" placeholder="auto = mean"
              min={0} value={form.fpg_max} onChange={(v) => setField('fpg_max', v)} />
          </div>

          <p className={styles.subLabel}>Total Cholesterol (mg/dL)</p>
          <div className={`${styles.grid3} ${styles.mb}`}>
            <FieldInput id="chol_mean" label="Mean" placeholder="e.g. 200"
              min={0} value={form.chol_mean} onChange={(v) => setField('chol_mean', v)} />
            <FieldInput id="chol_std" label="Std" placeholder="optional"
              min={0} value={form.chol_std} onChange={(v) => setField('chol_std', v)} />
            <FieldInput id="chol_max" label="Max" placeholder="auto = mean"
              min={0} value={form.chol_max} onChange={(v) => setField('chol_max', v)} />
          </div>

          <p className={styles.subLabel}>LDL Cholesterol (mg/dL)</p>
          <div className={styles.grid3}>
            <FieldInput id="ldl_mean" label="Mean" placeholder="e.g. 130"
              min={0} value={form.ldl_mean} onChange={(v) => setField('ldl_mean', v)} />
            <FieldInput id="ldl_std" label="Std" placeholder="optional"
              min={0} value={form.ldl_std} onChange={(v) => setField('ldl_std', v)} />
            <FieldInput id="ldl_max" label="Max" placeholder="auto = mean"
              min={0} value={form.ldl_max} onChange={(v) => setField('ldl_max', v)} />
          </div>
        </Card>

        {/* ── Comorbidities ──────────────────────────── */}
        <Card title="Comorbidities">
          <div className={styles.checkGrid}>
            {COMORBIDITIES.map(({ key, label }) => (
              <label key={key} className={`${styles.checkItem} ${form[key] ? styles.checkActive : ''}`}>
                <input type="checkbox" checked={form[key]}
                  onChange={(e) => setField(key, e.target.checked)} />
                {label}
              </label>
            ))}
          </div>
        </Card>

        {/* ── Current Medications ────────────────────── */}
        <Card title="Current Medications (HT)">
          <p className={styles.hint}>ยาที่ผู้ป่วยใช้อยู่ปัจจุบัน — ส่งผลต่อการ predict</p>
          <div className={styles.checkGrid}>
            {HT_MEDS.map(({ key, label }) => (
              <label key={key} className={`${styles.checkItem} ${form[key] ? styles.checkActive : ''}`}>
                <input type="checkbox" checked={form[key]}
                  onChange={(e) => setField(key, e.target.checked)} />
                {label}
              </label>
            ))}
          </div>
        </Card>

        <button type="submit" className={styles.submitBtn} disabled={loading}>
          {loading ? 'Predicting…' : 'Predict Complication Risk'}
        </button>

        {loading && (
          <div className={styles.spinner}>
            <div className={styles.spinnerDot} />
            Running XGBoost models…
          </div>
        )}

        {apiErr && <div className={styles.errorBanner}>{apiErr}</div>}
      </form>

      {result && <RiskResults result={result} />}
    </div>
  )
}
