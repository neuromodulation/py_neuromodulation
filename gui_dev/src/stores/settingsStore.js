import { createStore } from "./createStore";

const INITIAL_DELAY = 3000; // wait for Flask
const RETRY_DELAY = 1000; // ms
const MAX_RETRIES = 100;

const uploadSettingsToServer = async (settings) => {
  try {
    const response = await fetch("/api/settings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(settings),
    });
    if (!response.ok) {
      throw new Error("Failed to update settings");
    }
    return { success: true };
  } catch (error) {
    console.error("Error updating settings:", error);
    return { success: false, error };
  }
};

export const useSettingsStore = createStore("settings", (set, get) => ({
  settings: null,
  isLoading: false,
  error: null,
  retryCount: 0,

  setSettings: (settings) => set({ settings }),

  fetchSettingsWithDelay: () => {
    set({ isLoading: true, error: null });
    setTimeout(() => {
      get().fetchSettings();
    }, INITIAL_DELAY);
  },

  fetchSettings: async () => {
    try {
      console.log("Fetching settings...");
      const response = await fetch("/api/settings");
      if (!response.ok) {
        throw new Error("Failed to fetch settings");
      }
      const data = await response.json();
      set({ settings: data, retryCount: 0 });
    } catch (error) {
      console.log("Error fetching settings:", error);
      set((state) => ({
        error: error.message,
        retryCount: state.retryCount + 1,
      }));

      if (get().retryCount < MAX_RETRIES) {
        await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY));
        return get().fetchSettings();
      } else {
        console.error("Error fetching settings after all retries:", error);
      }
    }
  },

  resetRetryCount: () => set({ retryCount: 0 }),

  updateSettings: async (newSettings) => {
    //Update settings optimistically
    const currentSettings = get().settings;

    set({ settings: newSettings });

    const result = await uploadSettingsToServer(newSettings);

    await get().uploadSettings;

    if (!result.success) {
      // Revert the local state if the server update failed
      set({ settings: currentSettings });
    }

    return result;
  },
}));

export const useFetchSettings = () =>
  useSettingsStore((state) => state.fetchSettingsWithDelay);
