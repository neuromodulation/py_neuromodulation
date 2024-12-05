import { debounce } from "@/utils";
import { getBackendURL } from "@/utils";

const DEBOUNCE_MS = 500; // Adjust as needed


const syncWithBackend = async (state) => {
  try {
    const response = await fetch(getBackendURL("/api/experiment-session"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state),
    });
    if (!response.ok) throw new Error("Failed to sync with backend");
    return await response.json();
  } catch (error) {
    console.error("Sync failed:", error);
    throw error;
  }
};

const debouncedSync = debounce(syncWithBackend, DEBOUNCE_MS);


/*****************************/
/******** BACKEND SYNC *******/
/*****************************/

// Wrap state updates with sync logic
setState: async (newState) => {
  set((state) => ({ ...state, ...newState, syncStatus: "syncing" }));
  try {
    await debouncedSync(get());
    set({ syncStatus: "synced", syncError: null });
  } catch (error) {
    set({ syncStatus: "error", syncError: error.message });
  }
},

// // Use this for actions that need immediate sync
// setStateAndSync: async (newState) => {
//   set((state) => ({ ...state, ...newState, syncStatus: "syncing" }));
//   try {
//     const syncedState = await syncWithBackend(get());
//     set({ ...syncedState, syncStatus: "synced", syncError: null });
//   } catch (error) {
//     set({ syncStatus: "error", syncError: error.message });
//   }
// }