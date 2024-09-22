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
        self.active_connections: list[WebSocket] = []
        self.logger = logging.getLogger("PyNM")

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        client_address = websocket.client
        if client_address:
            self.logger.info(
                f"Client connected with client ID: {client_address.port}:{client_address.port}"
            )

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        client_address = websocket.client
        if client_address:
            self.logger.info(
                f"Client {client_address.port}:{client_address.port} disconnected."
            )

    # Combine IP and port to create a unique client ID
    async def send_bytes(self, header: dict, payload: bytes | None = None):
        if not self.active_connections:
            self.logger.warning("No active connectios to send message.")
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

        for connection in self.active_connections:
            await connection.send_bytes(message)

    async def send_message(self, message: str | dict):
        self.logger.info(f"Sending message within app_socket: {message.keys()}")
        if self.active_connections:
            for connection in self.active_connections:
                if type(message) is dict:
                    await connection.send_json(json.dump(message))
                elif type(message) is str:
                    await connection.send_text(message)
            self.logger.info(f"Message sent")
        else:
            self.logger.warning("No active connection to send message.")

    @property
    def is_connected(self):
        return self.active_connections is not None
