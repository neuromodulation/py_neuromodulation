from email.policy import HTTP
import tomllib
import logging
import importlib.metadata
from datetime import datetime
from pathlib import Path
import os
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from . import app_pynm
from .app_socket import WebSocketManager
from .app_utils import is_hidden, get_quick_access
import pandas as pd

from py_neuromodulation import PYNM_DIR, NMSettings
from py_neuromodulation.nm_types import FileInfo

# TODO: maybe pull this list from the MNE package?
ALLOWED_EXTENSIONS = [".npy", ".vhdr", ".fif", ".edf", ".bdf"]


class PyNMBackend(FastAPI):
    def __init__(
        self, pynm_state: app_pynm.PyNMState, debug=False, fastapi_kwargs: dict = {}
    ) -> None:
        super().__init__(debug=debug, **fastapi_kwargs)

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

        ####################
        ##### SETTINGS #####
        ####################

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

        ########################
        ##### PYNM CONTROL #####
        ########################

        @self.post("/api/stream-control")
        async def handle_stream_control(data: dict):
            action = data["action"]
            if action == "start":
                self.pynm_state.stream.run()
            # Add other actions as needed
            return {"message": f"Stream action '{action}' executed"}

        ####################
        ##### CHANNELS #####
        ####################

        @self.get("/api/channels")
        async def get_channels():
            channels = self.pynm_state.stream.nm_channels
            self.logger.info(f"Sending channels: {channels}")
            if isinstance(channels, pd.DataFrame):
                return {"channels": channels.to_dict(orient="records")}
            else:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "Channels is not a DataFrame"},
                )

        @self.post("/api/channels")
        async def update_channels(data: dict):
            try:
                new_channels = pd.DataFrame(data["channels"])
                self.logger.info(f"Received channels:\n {new_channels}")
                self.pynm_state.stream.nm_channels = new_channels
                return {
                    "channels": self.pynm_state.stream.nm_channels.to_dict(
                        orient="records"
                    )
                }
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "Error updating channels", "details": str(e)},
                )

        ###################
        ### LSL STREAMS ###
        ###################

        @self.get("/api/LSL-streams")
        async def get_lsl_streams():
            from mne_lsl.lsl import resolve_streams

            return {
                "message": [
                    {
                        "dtype":  # MNE-LSL might return a class, so we get the name
                        getattr(stream.dtype, "__name__", str(stream.dtype)),
                        "name": stream.name,
                        "n_channels": stream.n_channels,
                        "sfreq": stream.sfreq,
                        "source_id": stream.source_id,
                        "stype": stream.stype,  # Stream type (e.g. EEG)
                        "created_at": stream.created_at,
                        "hostname": stream.hostname,
                        "session_id": stream.session_id,
                        "uid": stream.uid,
                        "protocol_version": stream.protocol_version,
                    }
                    for stream in resolve_streams()
                ]
            }

        @self.post("/api/setup-LSL-stream")
        async def setup_lsl_stream(data: dict):
            try:
                stream_name = data["stream_name"]
                self.logger.info(f"Attempting to setup LSL stream: '{stream_name}'")
                self.pynm_state.setup_lsl_stream(
                    lsl_stream_name=stream_name,
                    sampling_rate_features=data["sampling_rate_features"],
                    line_noise=data["line_noise"],
                )
                return {"message": f"LSL stream '{stream_name}' setup successfully"}
            except Exception as e:
                return {
                    "message": "LSL stream could not be setup",
                    "error": str(e),
                }

        @self.post("/api/setup-Offline-stream")
        async def setup_offline_stream(data: dict):
            self.logger.info("Data received to setup offline stream:")
            self.logger.info(data)
            try:
                self.pynm_state.setup_offline_stream(
                    file_path=data["file_path"],
                    line_noise=data["line_noise"],
                    sampling_rate_features=data["sampling_rate_features"],
                )
                return {"message": f"Offline stream setup successfully"}
            except ValueError as e:
                return {"message": f"Offline stream could not be setup"}

        #######################
        ### PYNM ABOUT INFO ###
        #######################

        @self.get("/api/app-info")
        async def get_app_info():
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
                "website": urls.get("Homepage", ""),
                "authors": [metadata.get("Author-email", "")],
                "maintainers": [metadata.get("Maintainer", "")],
                "repository": urls.get("Repository", ""),
                "documentation": urls.get("Documentation", ""),
                "license": classifiers["License"],
                # "launchMode": "debug" if app.debug else "release",
            }

        ##############################
        ### FILE BROWSER ENDPOINTS ###
        ##############################
        # Get home directory for the current user
        @self.get("/api/home_directory")
        async def home_directory():
            try:
                home_dir = str(Path.home())
                return {"home_directory": home_dir}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Get list of available drives in Windows systems
        @self.get("/api/drives")
        async def list_drives():
            if os.name == "nt":
                import string

                drives = []
                for letter in string.ascii_uppercase:
                    if Path(f"{letter}:").exists():
                        drives.append(f"{letter}:")

                return {"drives": drives}
            else:
                return {"drives": ["/"]}  # Unix-like systems have a single root

        # Get list of files and directories in a directory
        @self.get("/api/files")
        async def list_files(
            path: str = Query(default="", description="Directory path to list"),
            allowed_extensions: str = Query(
                default=",".join(ALLOWED_EXTENSIONS),
                description="Comma-separated list of allowed file extensions",
            ),
            show_hidden: bool = Query(
                default=False,
                description="Whether to show hidden files and directories",
            ),
        ) -> list[FileInfo]:
            try:
                if not path:
                    path = str(Path.home())

                if not Path(path).is_dir():
                    raise FileNotFoundError("The specified path is not a directory")

                allowed_ext = allowed_extensions.split(",")

                files = []
                for entry in Path(path).iterdir():
                    # Skip hidden files/directories if show_hidden is False
                    if not show_hidden and is_hidden(entry):
                        continue

                    if entry.is_file() and not any(
                        entry.name.lower().endswith(ext.lower()) for ext in allowed_ext
                    ):
                        continue

                    stats = entry.stat()
                    files.append(
                        FileInfo(
                            name=entry.name,
                            path=str(entry),
                            dir=str(entry.parent),
                            is_directory=entry.is_dir(),
                            size=stats.st_size if not entry.is_dir() else 0,
                            created_at=datetime.fromtimestamp(stats.st_birthtime),
                            modified_at=datetime.fromtimestamp(stats.st_mtime),
                        )
                    )
                return files
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Directory not found")
            except PermissionError:
                raise HTTPException(status_code=403, detail="Permission denied")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.get("/api/quick-access")
        def quick_access():
            return get_quick_access()

        ###########################
        ### WEBSOCKET ENDPOINTS ###
        ###########################
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

        # #######################
        # ### SPA ENTRY POINT ###
        # #######################
        @self.get("/{full_path:path}")
        async def serve_spa(request, full_path: str):
            # Serve the index.html for any path that doesn't match an API route
            return FileResponse("frontend/index.html")
