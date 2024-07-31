from fastapi import WebSocket
import logging
import struct
import json

MAGIC_BYTE = b"b"  # Magic byte to identify non-textmessages


class WebSocketManager:
    """
    Manages WebSocket connections and messages.
    Perhaps in the future it will handle multiple connections.
    """

    def __init__(self):
        # self.active_connections: List[WebSocket] = []
        self.connection: WebSocket | None = None
        self.logger = logging.getLogger("PyNM")

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        # self.active_connections.append(websocket)
        self.connection = websocket
        self.logger.info(f"Client connected with client ID: {websocket.client}")

    def disconnect(self):
        # self.active_connections.remove(websocket)
        self.connection = None
        self.logger.info("Client disconnected.")

    async def send_bytes(self, header: dict, payload: bytes | None = None):
        if not self.connection:
            self.logger.warning("No active connection to send message.")
            return

        if payload:
            header["payload_length"] = len(payload)
            header["payload"] = True

        # Convert the header to JSON and encode it as UTF-8
        header_bytes = json.dumps(header).encode("utf-8")
        header_length = len(header_bytes)

        # Construct the message
        message = bytearray(MAGIC_BYTE)  # start with magic byte
        message.extend(struct.pack(">I", header_length))  # add header length
        message.extend(header_bytes)  # add header string in JSON format
        if payload:
            message.extend(payload)

        await self.connection.send_bytes(message)

    async def send_message(self, message: str):
        if self.connection:
            await self.connection.send_text(message)
            self.logger.info(f"Message sent: {message}")
        else:
            self.logger.warning("No active connection to send message.")

    @property
    def is_connected(self):
        return self.connection is not None
