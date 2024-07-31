import { create } from "zustand";

const WEBSOCKET_URL = "ws://localhost:50000/ws";
const MAGIC_BYTE = 98; // binary messages start with an ASCII `b`

export const useSocketStore = create((set, get) => ({
  socket: null,
  status: "disconnected", // 'disconnected', 'connecting', 'connected'
  error: null,
  graphData: null,
  infoMessages: [],

  setSocket: (socket) => set({ socket }),

  connectSocket: () => {
    // Get current socket status and cancel if connecting or already connected
    const { socket, status } = get();
    if (socket || status === "connecting" || status === "connected") return;

    // Set socket status to connecting
    set({ status: "connecting", error: null });

    // Create new socket connection
    const newSocket = new WebSocket(WEBSOCKET_URL);
    newSocket.binaryType = "arraybuffer";

    newSocket.onopen = () => {
      console.log("WebSocket connected");
      set({ socket: newSocket, status: "connected", error: null });
    };

    newSocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      set({ status: "disconnected", error: error });
    };

    newSocket.onclose = (event) => {
      console.log("WebSocket closed:", event.reason);
      set({ status: "disconnected", error: null });
      // Attempt to reconnect after a delay
      setTimeout(() => get().connectSocket(), 500);
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
    const { socket } = get();
    if (socket) {
      console.log(socket);
      socket.close();
    }
    set({ socket: null, status: "disconnected", error: null });
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
