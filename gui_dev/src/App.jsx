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
import { Box, Stack, ThemeProvider } from "@mui/material";
import CssBaseline from "@mui/material/CssBaseline";
import { AppBar, StatusBar } from "@/components";
import {
  Dashboard,
  SourceSelection,
  Channels,
  Settings,
  Decoding,
} from "@/pages";
import { theme } from "./theme";
import { StreamSelector } from "@/pages/SourceSelection/StreamSelector";
import { FileSelector } from "@/pages/SourceSelection/FileSelector";

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

  connectSocket();

  useEffect(() => {
    console.log("Connecting socket from App component...");
    connectSocket();
    return () => {
      console.log("Disconnecting socket from App component...");
      disconnectSocket();
    };
  }, [connectSocket, disconnectSocket]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Stack height="100vh" width="100vw" overflow="hidden">
          <AppBar />
          <Box
            sx={{
              height: "100%",
              overflow: "auto",
              width: "100%",
              p: 0,
              m: 0,
            }}
          >
            <Routes>
              <Route index element={<Navigate to="/source" replace />} />
              <Route path="source" element={<SourceSelection />}>
                <Route index element={<Navigate to="/source/file" replace />} />
                <Route path="file" element={<FileSelector />} />
                <Route path="lsl" element={<StreamSelector />} />
              </Route>
              <Route path="channels" element={<Channels />} />
              <Route path="settings" element={<Settings />} />
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="decoding" element={<Decoding />} />
            </Routes>
          </Box>
          <StatusBar />
        </Stack>
      </Router>
    </ThemeProvider>
  );
};
