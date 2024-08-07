import { useEffect } from "react";
import { useSettingsStore, useSocketStore, useWebviewStore } from "@/stores";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

import { TitleBar, StatusBar } from "@/components";
import { Dashboard, SourceSelection, Decoding, Channels, Settings } from "@/pages";
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
    <Router>
      <div className={styles.appContainer}>
        <TitleBar />
        <div className={styles.appContent}>
          <Routes>
            <Route exact path="/" element={<SourceSelection />} />
            <Route exact path="/channels" element={<Channels />} />
            <Route exact path="/dashboard" element={<Dashboard />} />
            <Route exact path="/decoding" element={<Dashboard />} />
            <Route exact path="/settings" element={<Settings />} />
          </Routes>
        </div>
        <StatusBar />
      </div>
    </Router>
  );
}
