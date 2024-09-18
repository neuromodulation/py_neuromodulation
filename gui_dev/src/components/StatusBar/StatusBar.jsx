import { ResizeHandle } from "./ResizeHandle";
import { SocketStatus } from "./SocketStatus";
import { WebviewStatus } from "./WebviewStatus";

import { useWebviewStore } from "@/stores";

import styles from "./StatusBar.module.css";

export const StatusBar = () => {
  const { isWebView } = useWebviewStore((state) => state.isWebView);

  return (
    <div className={styles.statusBar}>
      <div className={styles.spacerLeft}></div>

      <div className={styles.statusBarLeft}>
        <WebviewStatus />
        {/* Current experiment */}
        {/* Current stream */}
        {/* Current activity */}
      </div>
      <div className={styles.spacerMiddle}></div>
      <div className={styles.statusBarRight}>
        <SocketStatus />
      </div>
      <div className={styles.spacerRight}></div>
      {isWebView && <ResizeHandle />}
    </div>
  );
};
