import styles from "./Welcome.module.css";
import { useAppInfoStore } from "@/stores";

export const Welcome = () => (
  <div className={styles.welcomeContainer}>
    <h2>PyNeuromodulation</h2>
    <p className={styles.loading}>Loading application...</p>
    <div className={styles.info}>
      <p>Authors: John Doe, Jane Smith</p>
      <p>License: MIT</p>
    </div>
  </div>
);
