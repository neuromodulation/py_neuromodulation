import { create } from "zustand";

export const useAppInfoStore = create((set) => ({
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

export const useAppInfo = () =>
  useAppInfoStore((state) => ({
    version: state.version,
    website: state.website,
    authors: state.authors,
    maintainers: state.maintainers,
    repository: state.repository,
    documentation: state.documentation,
    license: state.license,
    launchMode: state.launchMode,
  }));
