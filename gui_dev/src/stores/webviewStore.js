import { create } from "zustand";

export const useWebviewStore = create((set, get) => ({
  isWebviewReady: false,
  statusMessage: "Waiting for PyWebView...",
  isMaximized: false,
  setIsWebviewReady: (isReady) => set({ isWebviewReady: isReady }),
  setStatusMessage: (message) => set({ statusMessage: message }),
  setIsMaximized: (maximized) => set({ isMaximized: maximized }),

  checkWebviewReady: () => {
    if (window.pywebview) {
      set({
        isWebviewReady: true,
        statusMessage: "Connected to PyWebView",
      });
    } else {
      if (!get().startTime) {
        get().startTime = Date.now();
      }
      set({ statusMessage: "Waiting for PyWebView..." });
      setTimeout(() => get().checkWebviewReady(), 100);
    }
  },
}));
