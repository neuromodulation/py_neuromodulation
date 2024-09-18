import { createStore } from "./createStore";

export const useAppInfoStore = createStore("appInfo", (set) => ({
  version: "",
  website: "",
  authors: [],
  maintainers: [],
  repository: "",
  documentation: "",
  license: "",
  launchMode: "",
  fetchAppInfo: async () => {
    try {
      const response = await fetch("/api/app-info");
      const data = await response.json();
      set(data);
    } catch (error) {
      console.error("Failed to fetch app info:", error);
    }
  },
}));

export const useFetchAppInfo = () =>
  useAppInfoStore((state) => state.fetchAppInfo);
