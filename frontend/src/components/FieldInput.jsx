import styles from './FieldInput.module.css'

/**
 * Labelled number/text/select input.
 *
 * @param {string}   id
 * @param {string}   label
 * @param {string}   [unit]
 * @param {string}   [placeholder]
 * @param {string}   [hint]
 * @param {boolean}  [error]
 * @param {*}        value
 * @param {function} onChange
 * @param {'number'|'text'|'select'} [type='number']
 * @param {Array<{value,label}>} [options]   — only for type='select'
 * @param {object}  [rest]  — forwarded to <input>
 */
export default function FieldInput({
  id,
  label,
  unit,
  placeholder,
  hint,
  error,
  value,
  onChange,
  type = 'number',
  options,
  ...rest
}) {
  return (
    <div className={styles.field}>
      <label htmlFor={id} className={styles.label}>
        {label}
        {unit && <span className={styles.unit}> ({unit})</span>}
      </label>

      {type === 'select' ? (
        <select
          id={id}
          className={`${styles.input} ${error ? styles.inputError : ''}`}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="">— Select —</option>
          {options?.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          type={type}
          className={`${styles.input} ${error ? styles.inputError : ''}`}
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          {...rest}
        />
      )}

      {hint && <p className={styles.hint}>{hint}</p>}
    </div>
  )
}
