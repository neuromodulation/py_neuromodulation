import { getBackendURL } from "@/utils";
import { createStore } from "./createStore";

const INITIAL_DELAY = 3000; // wait for Flask
const RETRY_DELAY = 1000; // ms
const MAX_RETRIES = 100;

export const useSettingsStore = createStore("settings", (set, get) => ({
  settings: null,
  lastValidSettings: null,
  frequencyRangeOrder: [],
  isLoading: false,
  error: null,
  validationErrors: null,
  retryCount: 0,

  updateLocalSettings: (updater) => {
    set((state) => updater(state.settings));
  },

  fetchSettingsWithDelay: () => {
    set({ isLoading: true, error: null });
    setTimeout(() => {
      get().fetchSettings();
    }, INITIAL_DELAY);
  },

  fetchSettings: async () => {
    try {
      console.log("Fetching settings...");
      const response = await fetch(getBackendURL("/api/settings"));
      if (!response.ok) {
        throw new Error("Failed to fetch settings");
      }

      const data = await response.json();

      set({
        settings: data,
        lastValidSettings: data,
        frequencyRangeOrder: Object.keys(data.frequency_ranges_hz || {}),
        retryCount: 0,
      });
    } catch (error) {
      console.log("Error fetching settings:", error);
      set((state) => ({
        error: error.message,
        retryCount: state.retryCount + 1,
      }));

      console.log(get().retryCount);

      if (get().retryCount < MAX_RETRIES) {
        await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY));
        return get().fetchSettings();
      } else {
        console.error("Error fetching settings after all retries:", error);
      }
    }
  },

  resetRetryCount: () => set({ retryCount: 0 }),

  resetSettings: async () => {
    await get().fetchSettings(true);
  },

  updateFrequencyRangeOrder: (newOrder) => {
    set({ frequencyRangeOrder: newOrder });
  },

  uploadSettings: async (updater, validateOnly = false) => {
    if (updater) {
      set((state) => {
        updater(state.settings);
      });
    }

    const currentSettings = get().settings;

    try {
      const response = await fetch(
        getBackendURL(
          `/api/settings${validateOnly ? "?validate_only=true" : ""}`
        ),
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(currentSettings),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error("Failed to upload settings to backend");
      }

      if (data.valid) {
        // Settings are valid
        set({
          lastValidSettings: currentSettings,
          validationErrors: null,
        });
        return true;
      } else {
        // Settings are invalid
        set({
          validationErrors: data.errors,
        });
        // Note: We don't revert the settings here, keeping the potentially invalid state
        return false;
      }
    } catch (error) {
      console.error(
        `Error ${validateOnly ? "validating" : "updating"} settings:`,
        error
      );
      return false;
    }
  },
}));

export const useSettings = () => useSettingsStore((state) => state.settings);
export const useFetchSettings = () =>
  useSettingsStore((state) => state.fetchSettingsWithDelay);
