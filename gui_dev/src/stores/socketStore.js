import { createStore } from "./createStore";

const WEBSOCKET_URL = "ws";
const RECONNECT_INTERVAL = 500; // ms
const WEBSOCKET_URL = "ws://localhost:50001/ws";
const MAGIC_BYTE = 98; // binary messages start with an ASCII `b`

export const useSocketStore = createStore("socket", (set, get) => ({
  socket: null,
  status: "disconnected", // 'disconnected', 'connecting', 'connected'
  error: null,
  graphData: [],
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
      console.log("Received message from server:", event.data);
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
        console.warning("Error closing socket:", error);
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
