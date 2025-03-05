import { createStore } from "./createStore";

const PYWEBVIEW_CHECK_INTERVAL = 100; // ms
const PYWEBVIEW_TIMEOUT = 15000; // ms, increased to 15 seconds

export const useWebviewStore = createStore("webview", (set) => ({
  isWebviewReady: false,
  isWebView: false,
  statusMessage: "Checking for PyWebview...",
  isMaximized: false,
  setIsWebviewReady: (isReady) => set({ isWebviewReady: isReady }),
  setStatusMessage: (message) => set({ statusMessage: message }),
  setIsMaximized: (maximized) => set({ isMaximized: maximized }),

  initPyWebView: () => {
    if (navigator.userAgent.includes("PyNmWebView")) {
      set({
        isWebView: true,
        statusMessage: "Detected PyWebView, waiting for API...",
      });

      const startTime = Date.now();
      const checkPywebview = () => {
        if (window.pywebview?.api) {
          set({
            isWebviewReady: true,
            statusMessage: "Found PyWebView API",
          });
        } else if (Date.now() - startTime > PYWEBVIEW_TIMEOUT) {
          set({
            statusMessage: "PyWebView initialization timed out",
            isWebviewReady: false,
          });
        } else {
          setTimeout(checkPywebview, PYWEBVIEW_CHECK_INTERVAL);
        }
      };

      checkPywebview();
    } else {
      set({
        isWebView: false,
        statusMessage: "Running in a regular browser",
        isWebviewReady: false,
      });
    }
  },
}));

export const useInitPyWebView = () =>
  useWebviewStore((state) => state.initPyWebView);
