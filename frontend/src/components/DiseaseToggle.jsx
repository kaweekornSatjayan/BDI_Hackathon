import styles from './DiseaseToggle.module.css'

const DISEASES = [
  { key: 'ht', label: 'Hypertension (HT)', icon: '🫀' },
  { key: 'dm', label: 'Diabetes (DM)',      icon: '🩸' },
]

/**
 * @param {{ ht: boolean, dm: boolean }} selected
 * @param {function} onChange  — called with updated { ht, dm }
 * @param {boolean}  error
 */
export default function DiseaseToggle({ selected, onChange, error }) {
  function toggle(key) {
    onChange({ ...selected, [key]: !selected[key] })
  }

  return (
    <div>
      <div className={styles.row}>
        {DISEASES.map(({ key, label, icon }) => (
          <button
            key={key}
            type="button"
            className={`${styles.btn} ${selected[key] ? styles.btnActive : ''}`}
            onClick={() => toggle(key)}
            aria-pressed={selected[key]}
          >
            <span>{icon}</span>
            {label}
          </button>
        ))}
      </div>
      {error && (
        <p className={styles.error}>Please select at least one disease.</p>
      )}
    </div>
  )
}
