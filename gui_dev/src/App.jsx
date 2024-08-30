import { useEffect } from "react";
import {
  useSocketStore,
  useFetchSettings,
  useInitPyWebView,
  useFetchAppInfo,
} from "@/stores";
import {
  BrowserRouter as Router,
  Route,
  Routes,
  Navigate,
} from "react-router-dom";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { AppBar, StatusBar } from "@/components";
import { Dashboard, SourceSelection, Channels, Settings } from "@/pages";
import styles from "./App.module.css";
import theme from "./theme";

/**
 *
 * @returns {JSX.Element} The rendered App component
 */
export const App = () => {
  // Get settings from backend on start-up and check for PyWebView
  const initPyWebView = useInitPyWebView();
  const fetchSettingsWithDelay = useFetchSettings();
  const fetchAppInfo = useFetchAppInfo();

  useEffect(() => {
    fetchSettingsWithDelay();
    initPyWebView();
    fetchAppInfo();
  }, [fetchSettingsWithDelay, initPyWebView, fetchAppInfo]);

  // Connect to web-socket
  const connectSocket = useSocketStore((state) => state.connectSocket);
  const disconnectSocket = useSocketStore((state) => state.disconnectSocket);

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
    <ThemeProvider theme={theme}>
      <CssBaseline />

      <Router>
        <div className={styles.appContainer}>
          <AppBar />
          <div className={styles.appContent}>
            <Routes>
              <Route path="/" element={<Navigate to="/source" replace />} />
              <Route
                path="/source/"
                element={<Navigate to="/source/file" replace />}
              />
              <Route exact path="/source/*" element={<SourceSelection />} />
              <Route exact path="/channels" element={<Channels />} />
              <Route exact path="/settings" element={<Settings />} />
              <Route exact path="/dashboard" element={<Dashboard />} />
              <Route exact path="/decoding" element={<Dashboard />} />
            </Routes>
          </div>
          <StatusBar />
        </div>
      </Router>
    </ThemeProvider>
  );
};
