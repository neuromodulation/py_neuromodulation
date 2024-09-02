// This store is used to store the current experimental
// session information, such as the experiment name,
// the data source, stream paramerters, the output files paths, etc

import { create } from "zustand";
import { debounce } from "@/utils";

const DEBOUNCE_MS = 500; // Adjust as needed

// Workflow stages enum-like object
const WorkflowStage = Object.freeze({
  SOURCE_SELECTION: Symbol("SOURCE_SELECTION"),
  CHANNEL_SELECTION: Symbol("CHANNEL_SELECTION"),
  SETTINGS_CONFIGURATION: Symbol("SETTINGS_CONFIGURATION"),
  VISUALIZATION: Symbol("VISUALIZATION"),
});

const syncWithBackend = async (state) => {
  try {
    const response = await fetch("/api/experiment-session", {
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

export const useSessionStore = create((set, get) => ({
  // Sync status
  syncStatus: "synced", // 'synced', 'syncing', 'error'
  syncError: null,

  // Workflow stage
  currentStage: WorkflowStage.SOURCE_SELECTION,
  streamSetupMessage: null,
  isStreamSetupCorrect: false,

  // Source selection
  sourceType: null, // 'file' or 'lsl'
  isSourceValid: false,
  fileSource: {
    filePath: "",
    fileFormat: "",
  },
  lslSource: {
    streamName: "",
  },
  samplingRateValue: null,
  lineNoiseValue: null,
  samplingRateFeaturesValue: null,
  areParametersValid: false,

  // Channel selection
  channels: null,
  selectedChannels: [],

  // Actions
  setSourceType: (type) => set({ sourceType: type }),
  setIsSourceValid: (isValid) => set({ isSourceValid: isValid }),
  setFileSource: (fileParams) =>
    set({ fileSource: { ...get().fileSource, ...fileParams } }),
  setLslSource: (lslParams) =>
    set({ lslSource: { ...get().lslSource, ...lslParams } }),
  setSelectedChannels: (channels) => set({ selectedChannels: channels }),
  setAnalysisParams: (params) =>
    set({ analysisParams: { ...get().analysisParams, ...params } }),
  setResults: (results) => set({ results }),
  setIsSessionActive: (isActive) => set({ isSessionActive: isActive }),

  setWorkflowStage: (stage) => {
    if (Object.values(WorkflowStage).includes(stage)) {
      set({ currentStage: stage });
    } else {
      console.error(`Invalid workflow stage: ${stage}`);
    }
  },

  // Get the current workflow stage
  getWorkflowStage: () => get().currentStage,

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

  // Use this for actions that need immediate sync
  setStateAndSync: async (newState) => {
    set((state) => ({ ...state, ...newState, syncStatus: "syncing" }));
    try {
      const syncedState = await syncWithBackend(get());
      set({ ...syncedState, syncStatus: "synced", syncError: null });
    } catch (error) {
      set({ syncStatus: "error", syncError: error.message });
    }
  },

  // Initial load from backend
  loadFromBackend: async () => {
    try {
      const response = await fetch("/api/experiment-session");
      if (!response.ok) throw new Error("Failed to load from backend");
      const backendState = await response.json();
      set({ ...backendState, syncStatus: "synced", syncError: null });
    } catch (error) {
      set({ syncStatus: "error", syncError: error.message });
    }
  },

  // Check that all stream parameters are valid
  checkStreamParameters: () => {
    const { samplingRateValue, lineNoiseValue, samplingRateFeaturesValue } =
      get();
    set({
      areParametersValid:
        samplingRateValue && lineNoiseValue && samplingRateFeaturesValue,
    });
  },

  // Stream initialization in the backend
  initializeStream: async () => {
    const { fileSource, sourceSelectionSettings } = get();
    try {
      const response = await fetch("/api/setup-Offline-stream", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          file_path: fileSource.filePath,
          sampling_rate_features: sourceSelectionSettings.samplingRateFeatures,
          line_noise: sourceSelectionSettings.lineNoise,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const content = await response.json();
      console.log(content);

      set({
        streamSetupMessage: content.message,
        isStreamSetupCorrect: true,
      });

      return content;
    } catch (error) {
      console.error("Error initializing stream:", error);
      set({
        streamSetupMessage: `Error: ${error.message}`,
        isStreamSetupCorrect: false,
      });
      throw error;
    }
  },

  // Computed properties
  canProceedToChannelSelection: () => {
    const { sourceType, isSourceValid } = get();
    return sourceType && isSourceValid;
  },

  canStartAnalysis: () => {
    const { isSourceValid, selectedChannels, analysisParams } = get();
    return (
      isSourceValid &&
      selectedChannels.length > 0 &&
      Object.keys(analysisParams).length > 0
    );
  },

  resetSession: () =>
    get().setStateAndSync({
      sourceType: null,
      isSourceValid: false,
      fileSource: { filePath: "" },
      lslSource: { streamName: "" },
      selectedChannels: [],
      analysisParams: {},
      results: null,
      isSessionActive: false,
    }),
}));

export { WorkflowStage };
