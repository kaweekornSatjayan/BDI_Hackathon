const API_BASE = import.meta.env.VITE_API_URL ?? ''

/**
 * @param {object} payload  — matches PredictInput schema
 * @returns {Promise<object>} — PredictOutput from backend
 */
export async function callPredict(payload) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err?.detail ?? `Server error ${res.status}`)
  }

  return res.json()
}
