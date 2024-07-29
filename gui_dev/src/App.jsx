import { useEffect } from "react";
import { useSettingsStore, useSocketStore, useWebviewStore } from "@/stores";
import {
  Settings,
  TitleBar,
  Graph,
  StatusBar,
  Sidebar,
  Drawer,
  CollapsibleBox,
} from "@/components";
import styles from "./App.module.css";

export default function App() {
  // Get settings from backend on start-up
  const fetchSettingsWithDelay = useSettingsStore(
    (state) => state.fetchSettingsWithDelay
  );
  useEffect(() => {
    fetchSettingsWithDelay();
  }, [fetchSettingsWithDelay]);

  // Connect to web-socket
  const { connectSocket, disconnectSocket } = useSocketStore((state) => ({
    connectSocket: state.connectSocket,
    disconnectSocket: state.disconnectSocket,
  }));

  useEffect(() => {
    console.log("Connecting socket from App component...");
    connectSocket();
    return () => {
      console.log("Disconnecting socket from App component...");
      disconnectSocket();
    };
  }, [connectSocket, disconnectSocket]);

  // Check PyWebView status
  const checkWebviewReady = useWebviewStore((state) => state.checkWebviewReady);

  useEffect(() => {
    checkWebviewReady();
  }, [checkWebviewReady]);

  return (
    <div className={styles.appContainer}>
      <TitleBar />
      <div className={styles.appContent}>
        <Sidebar>
          <Drawer name="settings">
            <CollapsibleBox title = "Features"> 
              <Settings />
            </CollapsibleBox>
          </Drawer>
        </Sidebar>
        <div className={styles.appMainContent}>
          <Graph />
        </div>
      </div>
      <StatusBar />
    </div>
  );
}
