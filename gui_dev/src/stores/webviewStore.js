import { create } from "zustand";

export const useWebviewStore = create((set, get) => ({
  isWebviewReady: false,
  isWebView: false,
  statusMessage: "Waiting for PyWebView...",
  isMaximized: false,
  setIsWebviewReady: (isReady) => set({ isWebviewReady: isReady }),
  setStatusMessage: (message) => set({ statusMessage: message }),
  setIsMaximized: (maximized) => set({ isMaximized: maximized }),

  initializePyWebView: () => {
    set({ statusMessage: "Checking for PyWebview..." });

    if (window.pywebview) {
      set({
        isWebView: true,
        statusMessage: "Detected PyWebView, waiting for API...",
      });

      const startTime = Date.now();
      while (!window.pywebview.api) {
        if (Date.now() - startTime > PYWEBVIEW_TIMEOUT) {
          set({
            statusMessage: "PyWebView initialization timed out",
            isWebviewReady: false,
          });
          return;
        }

        setTimeout(resolve, PYWEBVIEW_CHECK_INTERVAL);
      }

      set({
        isWebviewReady: true,
        statusMessage: "PyWebView is ready",
      });
    } else {
      set({
        isWebView: false,
        statusMessage: "Running in a regular browser",
        isWebviewReady: false,
      });
    }
  },
}));
