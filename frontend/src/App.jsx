import { useState } from 'react'
import PredictPage  from './pages/PredictPage'
import SuggestPage  from './pages/SuggestPage'
import styles from './App.module.css'

const TABS = [
  { id: 'predict', label: '📊 Risk Prediction' },
  { id: 'suggest', label: '💊 Medication Suggestion' },
]

export default function App() {
  const [tab, setTab] = useState('predict')

  return (
    <>
      <nav className={styles.nav}>
        <div className={styles.navInner}>
          <span className={styles.logo}>BDI Hack</span>
          <div className={styles.tabs}>
            {TABS.map((t) => (
              <button
                key={t.id}
                className={`${styles.tab} ${tab === t.id ? styles.tabActive : ''}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main>
        {tab === 'predict' ? <PredictPage /> : <SuggestPage />}
      </main>
    </>
  )
}
