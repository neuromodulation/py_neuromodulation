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
import { createTheme, ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { AppBar, StatusBar } from "@/components";
import { Dashboard, SourceSelection, Channels, Settings } from "@/pages";
import styles from "./App.module.css";

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
    console.log(window?.pywebview);
    console.log(window.pywebview?.api);

    console.log("Connecting socket from App component...");
    connectSocket();
    return () => {
      console.log("Disconnecting socket from App component...");
      disconnectSocket();
    };
  }, [connectSocket, disconnectSocket]);

  const theme = createTheme({
    palette: {
      mode: "dark", // This sets the overall theme to dark mode
      primary: {
        main: "#1a73e8", // Change this to your preferred primary color
      },
      secondary: {
        main: "#f4f4f4", // Light color for secondary elements
      },
      background: {
        default: "#333", // Background color
        paper: "#424242", // Background color for Paper components
      },
      text: {
        primary: "#f4f4f4", // Text color
        secondary: "#cccccc", // Slightly lighter text color
      },
    },
    typography: {
      fontFamily: '"Figtree", sans-serif', // Use the Figtree font globally
    },
  });

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
