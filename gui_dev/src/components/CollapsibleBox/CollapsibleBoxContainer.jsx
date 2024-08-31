import styles from "./CollapsibleBox.module.css";

export const CollapsibleBoxContainer = ({ children }) => {
  return <div className={styles.collapsibleBoxContainer}>{children}</div>;
};
