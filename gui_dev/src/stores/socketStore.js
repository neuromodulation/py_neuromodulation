import { createStore } from "./createStore";
import { getBackendURL } from "@/utils";
import CBOR from "cbor-js";

const WEBSOCKET_URL = getBackendURL("/ws");
const RECONNECT_INTERVAL = 500; // ms

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
        const arrayBuffer = event.data;
        const decodedData = CBOR.decode(arrayBuffer);
        // console.log("Decoded message from server:", decodedData);
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

          set({ availableDecodingOutputs: Object.keys(decodingData) });

          set({ graphDecodingData: decodingData });
          set({ graphData: dataNonDecodingFeatures });
        }
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
}));
