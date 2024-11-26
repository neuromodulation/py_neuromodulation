import { createStore } from "./createStore";
import { getBackendURL } from "@/utils/getBackendURL";
import init, { process_cbor_data } from '../../data_processor/pkg/cbor_decoder.js';

const WEBSOCKET_URL = getBackendURL("/ws");
const RECONNECT_INTERVAL = 500; // ms

let wasmInitialized = false;

async function initWasm() {
  if (!wasmInitialized) {
    await init();
    wasmInitialized = true;
  }
}

export const useSocketStore = createStore("socket", (set, get) => ({
  socket: null,
  status: "disconnected", // 'disconnected', 'connecting', 'connected'
  error: null,
  psdProcessedData: null,
  infoMessages: [],
  reconnectTimer: null,
  intentionalDisconnect: false,

  setSocket: (socket) => set({ socket }),

  connectSocket: () => {
    const { socket, status, reconnectTimer } = get();

    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
    }

    if (socket || status === "connecting" || status === "connected") return;

    set({ status: "connecting", error: null, intentionalDisconnect: false });

    const newSocket = new WebSocket(WEBSOCKET_URL);
    newSocket.binaryType = "arraybuffer"; // Default is "blob"

    newSocket.onopen = () => {
      console.log("WebSocket connected");
      set({ socket: newSocket, status: "connected", error: null });
    };

    newSocket.onerror = (event) => {
      if (!get().intentionalDisconnect) {
        console.error("WebSocket error:", event);
        set({
          status: "disconnected",
          error: "Connection error",
          socket: null,
        });

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

    newSocket.onmessage = async (event) => {
      try {
        const arrayBuffer = event.data;
        const uint8Array = new Uint8Array(arrayBuffer);

        // Ensure the WASM module is initialized
        await initWasm();

        // Process CBOR data using Rust module
        const processedData = process_cbor_data(uint8Array);

        // Set processed data in store
        set({ psdProcessedData: processedData });
        console.log("PSD processed data:", processedData);
      } catch (error) {
        console.error("Failed to process CBOR message:", error);
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
