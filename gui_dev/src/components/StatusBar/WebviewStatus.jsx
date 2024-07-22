import { useWebviewStore } from "@/stores";
import styles from "./StatusBar.module.css";

export const WebviewStatus = () => {
  const webviewStatus = useWebviewStore((state) => state.statusMessage);

  return <span className={styles.webviewStatus}>{webviewStatus}</span>;
};
