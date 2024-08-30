import tomllib
import numpy as np
import logging
import importlib.metadata

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from .app_pynm import PyNMState
from .app_socket import WebSocketManager

import pandas as pd

from py_neuromodulation import PYNM_DIR, NMSettings


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
                self.pynm_state.settings = NMSettings.model_validate(data)
                self.logger.info(self.pynm_state.settings.features)
                return self.pynm_state.settings.model_dump()
            except ValueError as e:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "Validation failed", "details": str(e)},
                )

        @self.get("/api/channels")
        async def get_channels():
            channels = self.pynm_state.stream.nm_channels.to_dict(orient='records')
            return channels

        @self.post("/api/channels")
        async def update_channels(data: dict):
            try:
                self.logger.info(self.pynm_state.settings.features)
                self.pynm_state.stream.nm_channels = pd.DataFrame(data)
                return {"message": "Channels updated successfully"}
            except ValueError as e:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "Validation failed", "details": str(e)},
                )

        @self.post("/api/stream-control")
        async def handle_stream_control(data: dict):
            action = data["action"]
            if action == "start":
                self.pynm_state.stream.run()
            # Add other actions as needed
            return {"message": f"Stream action '{action}' executed"}

        @self.get("/api/app-info")
        async def get_app_info():
            # TODO: make this function not depend on pyproject.toml, since it's not shipped
            pyproject_path = PYNM_DIR.parent / "pyproject.toml"

            try:
                with open(pyproject_path, "rb") as f:
                    pyproject_data = tomllib.load(f)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="pyproject.toml not found")
            except tomllib.TOMLDecodeError:
                raise HTTPException(
                    status_code=500, detail="Error parsing pyproject.toml"
                )

            metadata = importlib.metadata.metadata("py_neuromodulation")
            url_list = metadata.get_all("Project-URL")
            urls = (
                {url.split(",")[0]: url.split(",")[1] for url in url_list}
                if url_list
                else {}
            )

            classifier_list = metadata.get_all("Classifier")
            classifiers = (
                {
                    item[: item.find("::") - 1]: item[item.find("::") + 3 :]
                    for item in classifier_list
                }
                if classifier_list
                else {}
            )
            if "License" in classifiers:
                classifiers["License"] = classifiers["License"].split("::")[1]

            return {
                "version": metadata.get("Version", ""),
                "website": urls["Homepage"],
                "authors": [metadata.get("Author-email", "")],
                "maintainers": [metadata.get("Maintainer", "")],
                "repository": urls.get("Repository", ""),
                "documentation": urls.get("Documentation", ""),
                "license": classifiers["License"],
                # "launchMode": "debug" if app.debug else "release",
            }

        @self.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            # if self.websocket_manager.is_connected:
            #     self.logger.info(
            #         "WebSocket connection attempted while already connected"
            #     )
            #     await websocket.close(
            #         code=1008, reason="Another client is already connected"
            #     )
            #     return

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
                self.websocket_manager.disconnect(websocket)
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
            except Exception as e:
                self.logger.error(f"Error in periodic task: {e}")
                await asyncio.sleep(0.5)

        @self.get("/{full_path:path}")
        async def serve_spa(request, full_path: str):
            # Serve the index.html for any path that doesn't match an API route
            return FileResponse("frontend/index.html")
