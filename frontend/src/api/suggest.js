const API_BASE = import.meta.env.VITE_API_URL ?? ''

/**
 * @param {object} payload  — matches PatientInput schema
 * @returns {Promise<object>} — suggestions from backend
 */
export async function callSuggest(payload) {
  const res = await fetch(`${API_BASE}/api/suggest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    const detail = err?.detail
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg ?? JSON.stringify(d)).join(', ')
      : (typeof detail === 'string' ? detail : `Server error ${res.status}`)
    throw new Error(message)
  }

  return res.json()
}
