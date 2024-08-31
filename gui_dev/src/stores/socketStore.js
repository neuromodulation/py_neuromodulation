import { create } from "zustand";

const WEBSOCKET_URL = "ws";
const MAGIC_BYTE = 98; // binary messages start with an ASCII `b`
const RECONNECT_INTERVAL = 500; // ms

export const useSocketStore = create((set, get) => ({
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
      // console.log("Received message from server:", event.data);
      if (event.data instanceof ArrayBuffer) {
        const view = new DataView(event.data);

        const firstByte = view.getUint8(0);

        switch (firstByte) {
          case MAGIC_BYTE:
            // Get header length
            const headerLength = view.getUint32(1);
            // Parse JSON header
            const headerJson = new TextDecoder().decode(
              event.data.slice(5, 5 + headerLength)
            );
            const header = JSON.parse(headerJson);
            // console.log(header);

            if (header.payload) {
              // Extract payload
              const payload = event.data.slice(5 + headerLength);

              if (header.type === "new_batch") {
                console.log("Received new batch of data:");
                if (header.data_type === "float64") {
                  const data = new Float64Array(payload);
                  // console.log("Received new batch of data:", Array.from(data));
                  set({ graphData: Array.from(data) });
                }
                // Handle other data types as needed
              }
              // Handle other payload types as needed
            } else {
              // Handle non-payload messages
              if (header.type === "info") {
                set((state) => ({
                  infoMessages: [...state.infoMessages, header.message],
                }));
              }
              // Handle other message types as needed
            }
            break;

          // Handle JSON messages
          case "{":
            const message = JSON.parse(event.data);
            console.log("Received JSON message:", message);
            break;

          // Handle other message types
          default:
            console.error("Unrecognized message format:", event.data);
            break;
        }
      } else {
        console.error("Unexpected non-binary message:", event.data);
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
  clearMessages: () => set({ messages: [] }),
}));
