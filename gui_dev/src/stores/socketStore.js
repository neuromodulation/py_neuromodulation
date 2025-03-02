import { createStore } from "./createStore";
import { getBackendURL } from "@/utils";
import CBOR from "cbor-js";

const WEBSOCKET_URL = getBackendURL("/ws");
const RECONNECT_INTERVAL = 500; // ms

const getChannelAndFeature = (availableChannels, keystr) => {
  const channelName = availableChannels.find((channel) =>
    keystr.startsWith(channel + "_")
  );

  if (!channelName) return {};

  const featureName = keystr.slice(channelName.length + 1);

  return { channelName, featureName };
};

export const useSocketStore = createStore("socket", (set, get) => ({
  socket: null,
  status: "disconnected", // 'disconnected', 'connecting', 'connected'
  error: null,
  graphData: [],
  graphRawData: [],
  graphDecodingData: [],
  availableDecodingOutputs: [],
  infoMessages: [],
  reconnectTimer: null,
  intentionalDisconnect: false,
  messageCount: 0,

  setSocket: (socket) => set({ socket }),

  connectSocket: () => {
    // Get current socket status and cancel if connecting or already connected
    const { socket, status, reconnectTimer } = get();

    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
    }

    if (socket || status === "connecting" || status === "connected") return;

    // Set socket status to connecting
    set({ status: "connecting", error: null, intentionalDisconnect: false });

    // Create new socket connection
    const newSocket = new WebSocket(WEBSOCKET_URL);
    newSocket.binaryType = "arraybuffer"; // Default is "blob"

    newSocket.onopen = () => {
      console.log("WebSocket connected");
      set({ socket: newSocket, status: "connected", error: null });
    };

    // Error event fires when the connection is closed due to error
    newSocket.onerror = (event) => {
      if (!get().intentionalDisconnect) {
        console.error("WebSocket error:", event);
        set({
          status: "disconnected",
          error: "Connection error",
          socket: null,
        });

        // Attempt to reconnect after a delay
        get().setReconnectTimer(RECONNECT_INTERVAL);
      }
    };

    newSocket.onclose = (event) => {
      if (!get().intentionalDisconnect) {
        console.log("WebSocket closed unexpectedly:", event.reason);
        set({ status: "disconnected", error: null, socket: null });
        get().setReconnectTimer(RECONNECT_INTERVAL);
      } else {
        console.log("WebSocket closed intentionally");
      }
    };

    newSocket.onmessage = (event) => {
      try {
        const decodedData = CBOR.decode(event.data);
        if (Object.keys(decodedData)[0] == "raw_data") {
          set({ graphRawData: decodedData.raw_data });
        } else {
          // check here if there are values in decodedData that start with "decoding"
          // if so, set graphDecodingData to the value of those keys
          // else, set graphData to decodedData
          let decodingData = {};
          let dataNonDecodingFeatures = {};

          // check if this is the same:
          Object.entries(decodedData).forEach(([key, value]) => {
            (key.startsWith("decode") ? decodingData : dataNonDecodingFeatures)[
              key
            ] = value;
          });

          set({
            availableDecodingOutputs: Object.keys(decodingData),
            graphDecodingData: decodingData,
            graphData: dataNonDecodingFeatures,
          });
        }
        set({
          messageCount: get().messageCount + 1,
        });
      } catch (error) {
        console.error("Failed to decode CBOR message:", error);
      }
    };

    set({ socket: newSocket });
  },

  disconnectSocket: () => {
    const { socket, reconnectTimer } = get();

    set({ intentionalDisconnect: true });

    if (socket) {
      try {
        socket.close();
      } catch (error) {
        console.warn("Error closing socket:", error);
      }
    }

    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
    }

    set({
      socket: null,
      status: "disconnected",
      error: null,
      reconnectTimer: null,
      intentionalDisconnect: false,
    });
  },

  setReconnectTimer: (delay) => {
    const timer = setTimeout(() => {
      set({ intentionalDisconnect: false });
      get().connectSocket();
    }, delay);
    set({ reconnectTimer: timer });
  },

  // Method to send messages
  sendMessage: (message) => {
    const { socket, status } = get();
    if (socket && status === "connected") {
      socket.send(message);
    } else {
      console.error("Cannot send message: WebSocket not connected");
    }
  },

  // Clear messages
  clearMessages: set({ messages: [] }),

  getData: (selectedChannel, usedChannels) => {
    const fftFeatures = [
      "fft_theta_mean",
      "fft_alpha_mean",
      "fft_low_beta_mean",
      "fft_high_beta_mean",
      "fft_low_gamma_mean",
      "fft_high_gamma_mean",
    ];
    const dataByChannel = {};

    let graphData = get().graphData;
    for (const key in graphData) {
      const { channelName = "", featureName = "" } = getChannelAndFeature(
        usedChannels,
        key
      );
      if (!channelName) continue;
      if (!fftFeatures.includes(featureName)) continue;

      if (!dataByChannel[channelName]) {
        dataByChannel[channelName] = {
          channelName,
          features: [],
          values: [],
        };
      }

      dataByChannel[channelName].features.push(featureName);
      dataByChannel[channelName].values.push(graphData[key]);
    }

    const channelData = dataByChannel[selectedChannel];
    if (channelData) {
      const sortedValues = fftFeatures.map((feature) => {
        const index = channelData.features.indexOf(feature);
        return index !== -1 ? channelData.values[index] : null;
      });
      return {
        channelName: selectedChannel,
        features: fftFeatures.map((f) =>
          f.replace("_mean", "").replace("fft_", "")
        ),
        values: sortedValues,
      };
    } else {
      return {
        channelName: selectedChannel,
        features: fftFeatures.map((f) =>
          f.replace("_mean", "").replace("fft_", "")
        ),
        values: fftFeatures.map(() => null),
      };
    }
  },
}));
