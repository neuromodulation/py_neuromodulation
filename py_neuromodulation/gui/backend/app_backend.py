from pathlib import Path
from collections import defaultdict
import tomllib
import numpy as np
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from .app_pynm import PyNMState
from .app_socket import WebSocketManager

import pandas as pd

import py_neuromodulation as nm


class PyNMBackend(FastAPI):
    def __init__(self, pynm_state: PyNMState, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use the FastAPI logger for the backend
        self.logger = logging.getLogger("uvicorn.error")

        # Configure CORS
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:54321"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Has to be before mounting static files
        self.setup_routes()

        # Serve static files
        self.mount(
            "",
            StaticFiles(directory="py_neuromodulation/gui/frontend/", html=True),
            name="static",
        )

        self.pynm_state = pynm_state
        self.websocket_manager = WebSocketManager()

    def setup_routes(self):
        @self.get("/api/health")
        async def healthcheck():
            return {"message": "API is working"}

        @self.get("/api/settings")
        async def get_settings():
            return self.pynm_state.settings.model_dump()

        @self.post("/api/settings")
        async def update_settings(data: dict):
            try:
                self.pynm_state.settings = nm.NMSettings.model_validate(data)
                self.logger.info(self.pynm_state.settings.features)
                return self.pynm_state.settings.model_dump()
            except ValueError as e:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "Validation failed", "details": str(e)},
                )

        @self.get("/api/channels")
        async def get_channels():
            channels_html = self.pynm_state.stream.nm_channels.to_html(
                index=False, classes="table table-bordered"
            )
            return {"html": channels_html}

        @self.post("/api/channels")
        async def update_channels(data: dict):
            self.pynm_state.stream.nm_channels = pd.DataFrame(data)
            return {"message": "Channels updated successfully"}

        @self.post("/api/stream-control")
        async def handle_stream_control(data: dict):
            action = data["action"]
            if action == "start":
                self.pynm_state.stream.run()
            # Add other actions as needed
            return {"message": f"Stream action '{action}' executed"}

        @self.get("/api/app-info")
        # TODO: fix this function
        async def get_app_info():
            pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)

            project_info = defaultdict(lambda: "", pyproject_data.get("project", {}))
            urls = defaultdict(str, project_info.get("urls", {}))

            return {
                "version": project_info["version"],
                "website": urls["documentation"],
                "authors": [
                    author.get("name", "") for author in project_info["authors"]
                ],
                "maintainers": [
                    maintainer.get("name", "")
                    for maintainer in project_info["maintainers"]
                ],
                "repository": urls["repository"],
                "documentation": urls["documentation"],
                "license": next(
                    (
                        classifier.split(" :: ")[-1]
                        for classifier in project_info["classifiers"]
                        if classifier.startswith("License")
                    ),
                    "",
                ),
                "launchMode": "debug" if self.debug else "release",
            }

        @self.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            if self.websocket_manager.is_connected:
                self.logger.info(
                    "WebSocket connection attempted while already connected"
                )
                await websocket.close(
                    code=1008, reason="Another client is already connected"
                )
                return

            await self.websocket_manager.connect(websocket)

            periodic_task: asyncio.Task | None = None
            try:
                # Start the periodic task
                periodic_task = asyncio.create_task(self.send_periodic_data())

                # Handle incoming messages
                while True:
                    data = await websocket.receive_text()
                    await self.websocket_manager.send_message(
                        f"Message received: {data}"
                    )
            except WebSocketDisconnect:
                self.websocket_manager.disconnect()
                self.logger.info("Client disconnected")
            finally:
                # Ensure the periodic task is cancelled when the WebSocket disconnects
                if periodic_task:
                    periodic_task.cancel()
                    try:
                        await periodic_task
                    except asyncio.CancelledError:
                        pass

    async def send_periodic_data(self):
        while True:
            try:
                if self.websocket_manager.is_connected:
                    # Send binary data
                    data = np.random.random(1000).astype(np.float64)
                    header = {
                        "type": "new_batch",
                        "data_type": "float64",
                        "length": len(data),
                        "payload": True,
                    }
                    await self.websocket_manager.send_bytes(header, data.tobytes())

                    # Send JSON-only data
                    header = {
                        "type": "info",
                        "message": "This is an info message",
                        "payload": False,
                    }
                    await self.websocket_manager.send_bytes(header)

                await asyncio.sleep(0.016)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic task: {e}")
                break

        @self.get("/{full_path:path}")
        async def serve_spa(request, full_path: str):
            # Serve the index.html for any path that doesn't match an API route
            return FileResponse("frontend/index.html")
