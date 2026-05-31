import styles from './RiskCard.module.css'

const META = {
  LOW:    { icon: '🟢', label: 'LOW RISK'    },
  MEDIUM: { icon: '🟡', label: 'MEDIUM RISK' },
  HIGH:   { icon: '🔴', label: 'HIGH RISK'   },
}

const COMPLICATION_NAMES = {
  ckd:    'Chronic Kidney Disease',
  stroke: 'Stroke',
  cad:    'Coronary Artery Disease',
}

/**
 * @param {string} complication  — 'ckd' | 'stroke' | 'cad'
 * @param {number} probability   — 0–1
 * @param {'LOW'|'MEDIUM'|'HIGH'} level
 */
export default function RiskCard({ complication, probability, level }) {
  const { icon, label } = META[level] ?? META.LOW
  const pct = Math.round(probability * 100)

  return (
    <div className={`${styles.card} ${styles[level.toLowerCase()]}`}>
      <div className={styles.icon}>{icon}</div>
      <div className={styles.name}>{COMPLICATION_NAMES[complication] ?? complication}</div>
      <div className={styles.pct}>{pct}%</div>
      <span className={styles.badge}>{label}</span>
    </div>
  )
}
