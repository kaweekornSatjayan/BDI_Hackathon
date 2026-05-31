import styles from './MedTable.module.css'

const SCORE_COLOR = (score) => {
  if (score >= 60) return 'high'
  if (score >= 30) return 'med'
  return 'low'
}

/**
 * @param {Array} suggestions  — [{ medication_name, score, recommendation, is_current_medication }]
 */
export default function MedTable({ suggestions }) {
  if (!suggestions?.length) return <p className={styles.empty}>No suggestions available.</p>

  return (
    <div className={styles.wrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Medication</th>
            <th>Score</th>
            <th>Recommendation</th>
            <th>Current</th>
          </tr>
        </thead>
        <tbody>
          {suggestions.map((s) => (
            <tr key={s.medication_name} className={s.is_current_medication ? styles.currentRow : ''}>
              <td className={styles.medName}>{s.medication_name}</td>
              <td>
                <div className={styles.scoreBar}>
                  <div
                    className={`${styles.bar} ${styles[SCORE_COLOR(s.score)]}`}
                    style={{ width: `${Math.min(s.score, 100)}%` }}
                  />
                  <span className={styles.scoreLabel}>{s.score}%</span>
                </div>
              </td>
              <td>
                <span className={`${styles.badge} ${styles[SCORE_COLOR(s.score)]}`}>
                  {s.recommendation}
                </span>
              </td>
              <td className={styles.center}>
                {s.is_current_medication ? '✅' : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
