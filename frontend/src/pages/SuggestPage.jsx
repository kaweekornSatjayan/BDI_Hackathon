import { useState } from 'react'
import { callSuggest } from '../api/suggest'
import Card from '../components/Card'
import FieldInput from '../components/FieldInput'
import MedTable from '../components/MedTable'
import styles from './SuggestPage.module.css'

const COMORBIDITIES = [
  { key: 'co_ckd',                label: 'CKD' },
  { key: 'co_hf',                 label: 'Heart Failure' },
  { key: 'co_cad',                label: 'CAD' },
  { key: 'co_stroke',             label: 'Stroke' },
  { key: 'co_arrhythmias',        label: 'Arrhythmias' },
  { key: 'co_atrial_fibrillation',label: 'Atrial Fibrillation' },
  { key: 'co_dementia',           label: 'Dementia' },
]

const INITIAL_FORM = {
  disease_type: 'hypertension',
  age: '',
  sex: '',
  vitalsign_bmi_0: '',
  lab_hba1c_0: '',
  lab_fpg_0: '',
  vitalsign_sbp_0: '',
  vitalsign_dbp_0: '',
  vitalsign_hr_0: '',
  lab_chol_0: '',
  lab_ldl_0: '',
  lab_tg_0: '',
  lab_hdl_0: '',
  co_ckd: false,
  co_hf: false,
  co_cad: false,
  co_stroke: false,
  co_arrhythmias: false,
  co_atrial_fibrillation: false,
  co_dementia: false,
  current_medications: '',
}

function buildPayload(form) {
  const numOrZero = (v) => (v === '' || v == null ? 0 : Number(v))
  return {
    disease_type: form.disease_type,
    age:               Number(form.age),
    sex:               form.sex,
    vitalsign_bmi_0:   numOrZero(form.vitalsign_bmi_0),
    lab_hba1c_0:       numOrZero(form.lab_hba1c_0),
    lab_fpg_0:         numOrZero(form.lab_fpg_0),
    vitalsign_sbp_0:   numOrZero(form.vitalsign_sbp_0),
    vitalsign_dbp_0:   numOrZero(form.vitalsign_dbp_0),
    vitalsign_hr_0:    numOrZero(form.vitalsign_hr_0),
    lab_chol_0:        numOrZero(form.lab_chol_0),
    lab_ldl_0:         numOrZero(form.lab_ldl_0),
    lab_tg_0:          numOrZero(form.lab_tg_0),
    lab_hdl_0:         numOrZero(form.lab_hdl_0),
    co_ckd:            form.co_ckd ? 1 : 0,
    co_hf:             form.co_hf  ? 1 : 0,
    co_cad:            form.co_cad ? 1 : 0,
    co_stroke:         form.co_stroke ? 1 : 0,
    co_arrhythmias:    form.co_arrhythmias ? 1 : 0,
    co_atrial_fibrillation: form.co_atrial_fibrillation ? 1 : 0,
    co_dementia:       form.co_dementia ? 1 : 0,
    current_medications: form.current_medications
      .split(',')
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean),
  }
}

function validate(form) {
  const errs = {}
  if (!form.age)             errs.age             = true
  if (!form.sex)             errs.sex             = true
  if (!form.vitalsign_bmi_0) errs.vitalsign_bmi_0 = true
  return errs
}

export default function SuggestPage() {
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
      const data = await callSuggest(buildPayload(form))
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
        <h1 className={styles.title}>💊 Medication Suggestion</h1>
        <p className={styles.subtitle}>
          Weighted KNN Case-Based Reasoning — แนะนำยาจากผู้ป่วยที่คล้ายกัน
        </p>
      </header>

      <form onSubmit={handleSubmit} noValidate>

        {/* ── Disease Type ─────────────────────────── */}
        <Card title="Disease Type">
          <div className={styles.diseaseRow}>
            {['hypertension', 'diabetes', 'both'].map((d) => (
              <button
                key={d}
                type="button"
                className={`${styles.diseaseBtn} ${form.disease_type === d ? styles.diseaseBtnActive : ''}`}
                onClick={() => setField('disease_type', d)}
              >
                {d === 'hypertension' ? '🫀 Hypertension' : d === 'diabetes' ? '🩸 Diabetes' : '🫀🩸 Both'}
              </button>
            ))}
          </div>
        </Card>

        {/* ── Demographics ─────────────────────────── */}
        <Card title="Demographics">
          <div className={styles.grid2}>
            <FieldInput id="age" label="Age" unit="years" placeholder="e.g. 60"
              min={18} max={120} value={form.age}
              onChange={(v) => setField('age', v)} error={errors.age} />
            <FieldInput id="sex" label="Sex" type="select"
              options={[{ value: 'MALE', label: 'Male' }, { value: 'FEMALE', label: 'Female' }]}
              value={form.sex} onChange={(v) => setField('sex', v)} error={errors.sex} />
          </div>
        </Card>

        {/* ── Vitals & Labs ────────────────────────── */}
        <Card title="Vitals & Labs">
          <div className={styles.grid3}>
            <FieldInput id="vitalsign_bmi_0" label="BMI *" unit="kg/m²"
              placeholder="e.g. 27" min={10} max={70} step={0.1}
              value={form.vitalsign_bmi_0}
              onChange={(v) => setField('vitalsign_bmi_0', v)} error={errors.vitalsign_bmi_0} />
            <FieldInput id="vitalsign_sbp_0" label="SBP" unit="mmHg"
              placeholder="e.g. 150" value={form.vitalsign_sbp_0}
              onChange={(v) => setField('vitalsign_sbp_0', v)} />
            <FieldInput id="vitalsign_dbp_0" label="DBP" unit="mmHg"
              placeholder="e.g. 90" value={form.vitalsign_dbp_0}
              onChange={(v) => setField('vitalsign_dbp_0', v)} />
            <FieldInput id="vitalsign_hr_0" label="Heart Rate" unit="bpm"
              placeholder="e.g. 75" value={form.vitalsign_hr_0}
              onChange={(v) => setField('vitalsign_hr_0', v)} />
            <FieldInput id="lab_hba1c_0" label="HbA1c" unit="%"
              placeholder="e.g. 7.5" step={0.1} value={form.lab_hba1c_0}
              onChange={(v) => setField('lab_hba1c_0', v)} />
            <FieldInput id="lab_fpg_0" label="FPG" unit="mg/dL"
              placeholder="e.g. 130" value={form.lab_fpg_0}
              onChange={(v) => setField('lab_fpg_0', v)} />
            <FieldInput id="lab_chol_0" label="Total Cholesterol" unit="mg/dL"
              placeholder="e.g. 210" value={form.lab_chol_0}
              onChange={(v) => setField('lab_chol_0', v)} />
            <FieldInput id="lab_ldl_0" label="LDL" unit="mg/dL"
              placeholder="e.g. 130" value={form.lab_ldl_0}
              onChange={(v) => setField('lab_ldl_0', v)} />
            <FieldInput id="lab_tg_0" label="Triglycerides" unit="mg/dL"
              placeholder="e.g. 150" value={form.lab_tg_0}
              onChange={(v) => setField('lab_tg_0', v)} />
            <FieldInput id="lab_hdl_0" label="HDL" unit="mg/dL"
              placeholder="e.g. 50" value={form.lab_hdl_0}
              onChange={(v) => setField('lab_hdl_0', v)} />
          </div>
        </Card>

        {/* ── Comorbidities ────────────────────────── */}
        <Card title="Comorbidities">
          <div className={styles.comorbGrid}>
            {COMORBIDITIES.map(({ key, label }) => (
              <label key={key} className={`${styles.comorbItem} ${form[key] ? styles.comorbActive : ''}`}>
                <input
                  type="checkbox"
                  checked={form[key]}
                  onChange={(e) => setField(key, e.target.checked)}
                />
                {label}
              </label>
            ))}
          </div>
        </Card>

        {/* ── Current Medications ──────────────────── */}
        <Card title="Current Medications (optional)">
          <FieldInput
            id="current_medications"
            label="Current medications"
            type="text"
            placeholder="e.g. metformin, acei (comma-separated)"
            value={form.current_medications}
            onChange={(v) => setField('current_medications', v)}
            hint="ยาที่ผู้ป่วยใช้อยู่ปัจจุบัน — จะถูก highlight ในผลลัพธ์"
          />
        </Card>

        <button type="submit" className={styles.submitBtn} disabled={loading}>
          {loading ? 'Searching similar cases…' : 'Suggest Medications'}
        </button>

        {loading && (
          <div className={styles.spinner}>
            <div className={styles.spinnerDot} />
            Searching similar patient records…
          </div>
        )}

        {apiErr && <div className={styles.errorBanner}>{apiErr}</div>}
      </form>

      {/* ── Results ──────────────────────────────── */}
      {result && (
        <Card title="Medication Suggestions">
          <div className={styles.meta}>
            <span className={styles.tag}>{result.disease_mode?.toUpperCase()} mode</span>
            <span className={styles.metaText}>
              Based on {result.match_details?.neighbors_used} similar cases
            </span>
          </div>
          <MedTable suggestions={result.suggestions} />
          {result.not_relevant?.length > 0 && (
            <p className={styles.notRelevant}>
              Not relevant: {result.not_relevant.join(', ')}
            </p>
          )}
        </Card>
      )}
    </div>
  )
}
