import styles from "./TitleBar.module.css";
import { WindowButtons } from "./WindowButtons";
import { TiHome } from "react-icons/ti";

export const TitleBar = () => {
  return (
    <div className={styles.titleBar}>
      <a href="https://neuromodulation.github.io/py_neuromodulation/">
        <h1>PyNeuromodulation</h1>
      </a>
      <a className={styles.homeButton} href="welcome">
        <TiHome className="homeIcon" />
      </a>
      <div className={`${styles.dragArea} pywebview-drag-region`}></div>
      <WindowButtons />
    </div>
  );
};
