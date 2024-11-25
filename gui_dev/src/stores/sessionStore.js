// This store is used to store the current experimental
// session information, such as the experiment name,
// the data source, stream paramerters, the output files paths, etc

import { createStore } from "@/stores/createStore";
import { getBackendURL } from "@/utils/getBackendURL";

// Workflow stages enum-like object
export const WorkflowStage = Object.freeze({
  SOURCE_SELECTION: Symbol("SOURCE_SELECTION"),
  CHANNEL_SELECTION: Symbol("CHANNEL_SELECTION"),
  SETTINGS_CONFIGURATION: Symbol("SETTINGS_CONFIGURATION"),
  VISUALIZATION: Symbol("VISUALIZATION"),
});

export const useSessionStore = createStore("session", (set, get) => ({
  // Sync status
  syncStatus: "synced", // 'synced', 'syncing', 'error'
  syncError: null,

  // Workflow stage
  currentStage: WorkflowStage.SOURCE_SELECTION,
  streamSetupMessage: "Stream not setup",
  isStreamSetupCorrect: false,

  // Get the current workflow stage
  getWorkflowStage: () => get().currentStage,
  setWorkflowStage: (stage) => {
    if (Object.values(WorkflowStage).includes(stage)) {
      set({ currentStage: stage });
    } else {
      console.error(`Invalid workflow stage: ${stage}`);
    }
  },

  /*****************************/
  /***** SOURCE SELECTION ******/
  /*****************************/

  // Source selection
  sourceType: null, // 'file' or 'lsl'
  isSourceValid: false,
  fileSource: {}, // FileInfo object
  lslSource: {
    selectedStream: null,
    availableStreams: [],
  },
  streamParameters: {
    samplingRate: 1000,
    lineNoise: 50,
    samplingRateFeatures: 11,
    allValid: false,
    experimentName: "sub",
    outputDirectory: "default",
  },

  setSourceType: (type) => set({ sourceType: type }),
  updateStreamParameter: (field, value) =>
    set((state) => {
      state.streamParameters[field] = value;
    }),

  // Actions
  /**
   *
   * @param {import("@/utils/FileManager").FileInfo} fileObj
   */
  setFileSource: (fileObj) => {
    set({
      fileSource: fileObj,
      isSourceValid: true,
    });
  },

  /*****************************/
  /******** LSL STREAMS ********/
  /*****************************/

  // Search for LSL streams

  fetchLSLStreams: async () => {
    const response = await fetch(getBackendURL("/api/LSL-streams"));

    if (!response.ok) {
      set({ lslSource: { selectedStream: null, availableStreams: [] } });
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    const streams = data.message;

    set({
      lslSource: { availableStreams: streams },
    });
  },

  selectLSLStream: async (streamIndex) => {
    set((state) => {
      state.lslSource.selectedStream =
        state.lslSource.availableStreams[streamIndex];
      state.streamParameters.samplingRate =
        state.lslSource.selectedStream.sfreq;
    });

    get().checkStreamParameters();
  },

  // Check that all stream parameters are valid
  checkStreamParameters: () => {
    // const { samplingRate, lineNoise, samplingRateFeatures } = get();
    set({
      areParametersValid:
        get().streamParameters.samplingRate &&
        get().streamParameters.lineNoise &&
        get().streamParameters.samplingRateFeatures,
    });
  },

  /*****************************/
  /******** STREAM SETUP *******/
  /*****************************/

  // Stream initialization in the backend
  initializeOfflineStream: async () => {
    try {
      const response = await fetch(getBackendURL("/api/setup-Offline-stream"), {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          file_path: get().fileSource.path,
          sampling_rate_features: get().streamParameters.samplingRateFeatures,
          line_noise: get().streamParameters.lineNoise,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const content = await response.json();

      set({
        streamSetupMessage: content.message,
        isStreamSetupCorrect: true,
      });

      get().fetchChannels();
    } catch (error) {
      console.error("Error initializing stream:", error);
      set({
        streamSetupMessage: `Error: ${error.message}`,
        isStreamSetupCorrect: false,
      });
      throw error;
    }
  },

  initializeLSLStream: async () => {
    const lslSource = get().lslSource;
    const streamParameters = get().streamParameters;

    try {
      const response = await fetch(getBackendURL("/api/setup-LSL-stream"), {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          stream_name: lslSource.availableStreams[0].name,
          sampling_rate_features: streamParameters.samplingRateFeatures,
          line_noise: streamParameters.lineNoise,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const content = await response.json();

      set({
        sourceType: "lsl",
        isSourceValid: true,
      });

      set({
        streamSetupMessage: content.message,
        isStreamSetupCorrect: true,
      });

      get().fetchChannels();
    } catch (error) {
      console.error("Error initializing stream:", error);
      set({
        streamSetupMessage: `Error: ${error.message}`,
        isStreamSetupCorrect: false,
      });
      throw error;
    }
  },

  sendStreamParametersToBackend: async () => {
    const streamParameters = get().streamParameters;

    try {
      const response = await fetch(getBackendURL("/api/set-stream-params"), {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sampling_rate: streamParameters.samplingRate,
          sampling_rate_features: streamParameters.samplingRateFeatures,
          line_noise: streamParameters.lineNoise,
          experiment_name: streamParameters.experimentName,
          out_dir: streamParameters.outputDirectory,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    } catch (error) {
      console.error("Error sendin stream params:", error);
      set({
        streamSetupMessage: `Error: ${error.message}`,
        isStreamSetupCorrect: false,
      });
      throw error;
    }
  },

  /*****************************/
  /***** CHANNEL SELECTION *****/
  /*****************************/
  channels: [],

  updateChannel: (index, field, value) =>
    set((state) => {
      state.channels[index][field] = value;
    }),

  fetchChannels: async () => {
    try {
      const response = await fetch(getBackendURL("/api/channels"));
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      const data = await response.json();

      if ("channels" in data) {
        set({ channels: data.channels });
      } else {
        throw new Error("Invalid channels response");
      }
    } catch (error) {
      console.error("Error fetching channels:", error);
    }
  },

  uploadChannels: async () => {
    try {
      console.log(
        "Data being sent to the backend:",
        JSON.stringify({ channels: get().channels })
      );

      const response = await fetch(getBackendURL("/api/channels"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ channels: get().channels }),
      });

      if (!response.ok) {
        throw new Error(`Failed to update channels: ${await response.text()}`);
      }

      const result = await response.json();
      console.log("Update successful:", result);
    } catch (error) {
      console.error("Error updating channels:", error);
    }
  },

  // Computed properties
  canStartAnalysis: () => {
    const { isSourceValid, selectedChannels, analysisParams } = get();
    return (
      isSourceValid &&
      selectedChannels.length > 0 &&
      Object.keys(analysisParams).length > 0
    );
  },

  startStream: async () => {
    try {
      console.log("Start Stream");

      const response = await fetch(getBackendURL("/api/stream-control"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // This needs to be adapted depending on the backend changes
        body: JSON.stringify({ action: "start" }),
      });

      if (!response.ok) {
        throw new Error(`Failed start stream: ${await response.text()}`);
      }

      const result = await response.json();
      console.log("Stream started:", result);
    } catch (error) {
      console.error("Failed to start stream:", error);
    }
  },

  stopStream: async () => {
    try {
      console.log("Stop Stream");

      const response = await fetch(getBackendURL("/api/stream-control"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // This needs to be adapted depending on the backend changes
        body: JSON.stringify({ action: "stop" }),
      });

      if (!response.ok) {
        throw new Error(`Failed stopping stream: ${await response.text()}`);
      }

      const result = await response.json();
      console.log("Stream Stopping:", result);
    } catch (error) {
      console.error("Failed to stop stream:", error);
    }
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
