import { create } from "zustand";
import io from "socket.io-client";

const SOCKET_SERVER_URL = "http://localhost:5000";
const SOCKET_CLIENT_URL = "http://localhost:5173";

export const useSocketStore = create((set, get) => ({
  socket: null,
  status: "disconnected", // 'disconnected', 'connecting', 'connected'
  error: null,

  setSocket: (socket) => set({ socket }),

  connectSocket: () => {
    const { socket, status } = get();
    if (socket || status === "connecting") return;

    set({ status: "connecting", error: null });

    const newSocket = io(SOCKET_SERVER_URL, {
      transports: ["websocket"],
      cors: {
        origin: SOCKET_CLIENT_URL,
        methods: ["GET", "POST"],
      },
    });

    newSocket.on("connect", () => {
      console.log("Socket connected with ID:", newSocket.id);
      set({ socket: newSocket, status: "connected", error: null });
    });

    newSocket.on("connect_error", (error) => {
      console.error("Socket connection error:", error);
      set({ status: "disconnected", error: error });
    });

    newSocket.on("disconnect", (reason) => {
      console.log("Socket disconnected:", reason);
      set({ status: "disconnected", error: null });
    });

    newSocket.on("message", (message) => {
      console.log("Received message from server:", message);
    });

    // Attempt to reconnect on error
    newSocket.io.on("error", (error) => {
      console.error("Socket error:", error);
      set({ status: "disconnected", error: error });
      get().connectSocket(); // Attempt to reconnect
    });
  },

  disconnectSocket: () => {
    const { socket } = get();
    if (socket) {
      socket.removeAllListeners();
      socket.disconnect();
    }
    set({ socket: null, status: "disconnected", error: null });
  },

  // Add a method to send messages
  sendMessage: (event, data) => {
    const { socket, status } = get();
    if (socket && status === "connected") {
      socket.emit(event, data);
    } else {
      console.error("Cannot send message: socket not connected");
    }
  },
}));
