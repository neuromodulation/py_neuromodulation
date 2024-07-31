from pathlib import Path
from collections import defaultdict
import tomllib

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app_pynm import PyNMState

import pandas as pd

import py_neuromodulation as nm


class PyNMBackend(FastAPI):
    def __init__(self, pynm_state: PyNMState, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Configure CORS
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:54321"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Serve static files
        self.mount(
            "/static",
            StaticFiles(directory="py_neuromodulation/gui/frontend/"),
            name="static",
        )

        self.pynm_state = pynm_state
        self.setup_routes()

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
                print(self.pynm_state.settings.features)
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
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    await websocket.send_text(f"Message text was: {data}")
            except WebSocketDisconnect:
                print("Client disconnected")

        @self.get("/{full_path:path}")
        async def serve_spa(request, full_path: str):
            # Serve the index.html for any path that doesn't match an API route
            return FileResponse("frontend/index.html")
