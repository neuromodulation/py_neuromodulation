import asyncio
import numpy as np
from flask_socketio import SocketIO
import threading
import datetime


class RandomGenerator:
    def __init__(self, socket: SocketIO):
        self.socket = socket

        self.start()

    def start(self):
        processing_thread = threading.Thread(
            target=asyncio.run, args=(self.generate(),)
        )
        processing_thread.daemon = True
        processing_thread.start()

    async def generate(self):
        BATCH_TIME = 0.032  # 30 FPS

        while True:
            # Simulate processing a batch of data
            data = np.random.random(1000).astype(np.float64)
            self.socket.emit("new_batch", data.tobytes())
            print(
                f"Emitting new batch of data at {datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}"
            )
            self.socket.sleep(BATCH_TIME)  # Simulate processing time # type: ignore
