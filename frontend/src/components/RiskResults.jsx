import RiskCard from './RiskCard'
import styles from './RiskResults.module.css'

const DISEASE_LABELS = {
  ht:       'Hypertension (HT)',
  dm:       'Diabetes (DM)',
  combined: 'HT + DM (Combined)',
}

/**
 * @param {object} result  — PredictOutput from backend
 *   { model_used, risks: { ckd, stroke, cad }, explanation }
 */
export default function RiskResults({ result }) {
  const { model_used, risks, explanation } = result

  return (
    <div className={styles.wrapper}>
      <span className={styles.tag}>{DISEASE_LABELS[model_used] ?? model_used}</span>

      <div className={styles.grid}>
        {Object.entries(risks).map(([comp, { probability, level }]) => (
          <RiskCard
            key={comp}
            complication={comp}
            probability={probability}
            level={level}
          />
        ))}
      </div>

      {explanation && (
        <div className={styles.explanation}>{explanation}</div>
      )}

      <p className={styles.disclaimer}>
        ⚠️ This tool is for clinical decision support only. Results are probabilistic
        and must be reviewed by a qualified healthcare professional.
      </p>
    </div>
  )
}
