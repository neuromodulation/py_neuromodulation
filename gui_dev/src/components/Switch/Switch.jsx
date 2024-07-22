import styles from "./Switch.module.css";

export const Switch = ({ isEnabled, onChange, label }) => {
  return (
    <label className={styles.label}>
      <span className={styles.switch}>
        <input
          type="checkbox"
          checked={isEnabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className={styles.slider}></span>
      </span>
      <span className={styles.featureName}>{label}</span>
    </label>
  );
};
