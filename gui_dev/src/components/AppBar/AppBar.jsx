import styles from "./AppBar.module.css";
import { WindowButtons } from "./WindowButtons";
import { Toolbar } from "./Toolbar";
import { useWebviewStore } from "@/stores";

export const AppBar = () => {
  const { isWebView } = useWebviewStore((state) => ({
    isWebView: state.isWebView,
  }));

  return (
    <div className={`${styles.appBar} pywebview-drag-region`}>
      <h1>PyNeuromodulation</h1>
      <Toolbar />
      {isWebView && <WindowButtons />}
    </div>
  );
};
