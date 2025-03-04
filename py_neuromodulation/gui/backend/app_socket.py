from fastapi import WebSocket
import logging
import cbor2
import time


class WebsocketManager:
    """
    Manages WebSocket connections and messages.
    Perhaps in the future it will handle multiple connections.
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.logger = logging.getLogger("PyNM")
        self.disconnected = []
        self._queue_task = None
        self._stop_event = None
        self.loop = None
        self.messages_sent = 0
        self._last_diagnostic_time = time.time()

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
    async def send_cbor(self, object: dict):
        if not self.active_connections:
            self.logger.warning("No active connection to send message.")
            return

        start_time = time.time()
        cbor_data = cbor2.dumps(object)
        serialize_time = time.time() - start_time

        if serialize_time > 0.1:  # Log slow serializations
            self.logger.warning(f"CBOR serialization took {serialize_time:.3f}s")

        send_start = time.time()
        for connection in self.active_connections:
            try:
                await connection.send_bytes(cbor_data)
            except RuntimeError as e:
                self.logger.error(f"Error sending CBOR message: {e}")
                self.disconnected.append(connection)

        send_time = time.time() - send_start
        if send_time > 0.1:  # Log slow sends
            self.logger.warning(f"WebSocket send took {send_time:.3f}s")

        self.messages_sent += 1

        # Log diagnostics every 5 seconds
        current_time = time.time()
        if current_time - self._last_diagnostic_time > 5:
            self.logger.info(f"Messages sent: {self.messages_sent}")
            self._last_diagnostic_time = current_time

    async def send_message(self, message: str | dict):
        if not self.active_connections:
            self.logger.warning("No active connection to send message.")
            return

        self.logger.info(
            f"Sending message within app_socket: {message.keys() if type(message) is dict else message}"
        )
        for connection in self.active_connections:
            try:
                if type(message) is dict:
                    await connection.send_json(message)
                elif type(message) is str:
                    await connection.send_text(message)
                self.logger.info(f"Message sent to {connection.client}")
            except RuntimeError as e:
                self.logger.error(f"Error sending message: {e}.")
                self.active_connections.remove(connection)
                await connection.close()

    @property
    def is_connected(self):
        return self.active_connections is not None
