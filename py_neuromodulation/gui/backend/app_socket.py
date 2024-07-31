from flask_socketio import SocketIO
from flask import request


class PyNMSocket(SocketIO):
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)

        @self.on("connect")
        def handle_connect():
            print(f"Client connected. Session ID: {self.get_session_id()}")

        @self.on("disconnect")
        def handle_disconnect():
            print(f"Client disconnected. Session ID: {self.get_session_id()}")

        @self.on("message")
        def handle_message(data):
            print(f"Received message from {self.get_session_id()}:", data)
            self.emit("response", {"message": f"Server received: {data}"})

    def get_session_id(self):
        return request.sid  # type: ignore
