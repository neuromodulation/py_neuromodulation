import { useState } from "react";
import { WindowButtons } from "./WindowButtons";
import { Toolbar } from "./Toolbar";
import { useWebviewStore } from "@/stores";
import { AppInfoModal } from "@/components";
import styles from "./AppBar.module.css";

export const AppBar = () => {
  const { isWebView } = useWebviewStore((state) => ({
    isWebView: state.isWebView,
  }));

  const [showModal, setShowModal] = useState(false);

  const handleTitleClick = () => {
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
  };

  return (
    <div className={`${styles.appBar} pywebview-drag-region`}>
      <h1 className={styles.appTitle} onClick={handleTitleClick}>
        PyNeuromodulation
      </h1>
      <Toolbar />
      {isWebView && <WindowButtons />}
      {showModal && <AppInfoModal onClose={handleCloseModal} />}
    </div>
  );
};
