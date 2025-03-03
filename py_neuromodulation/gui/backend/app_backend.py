import logging
import importlib.metadata
from datetime import datetime
from pathlib import Path
import os

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from . import app_pynm
from .app_socket import WebsocketManager
from .app_utils import is_hidden, get_quick_access
import pandas as pd

from py_neuromodulation import PYNM_DIR, NMSettings
from py_neuromodulation.utils.types import FileInfo

# TODO: maybe pull this list from the MNE package?
ALLOWED_EXTENSIONS = [".npy", ".vhdr", ".fif", ".edf", ".bdf"]

DEV_SERVER_PORT = 54321


class PyNMBackend(FastAPI):
    def __init__(
        self,
        debug: bool | None = None,
        dev: bool | None = None,
        dev_port: int | None = None,
        fastapi_kwargs: dict = {},
    ) -> None:
        if debug is None:
            self.debug = os.environ.get("PYNM_DEBUG", "False").lower() == "true"
        if dev is None:
            self.dev = os.environ.get("PYNM_DEV", "False").lower() == "true"
        if dev_port is None:
            self.dev_port = os.environ.get("PYNM_DEV_PORT", str(DEV_SERVER_PORT))

        super().__init__(
            title="PyNeuromodulation",
            description="PyNeuromodulation FastAPI backend",
            version=importlib.metadata.version("py_neuromodulation"),
            debug=self.debug,
            **fastapi_kwargs,
        )

        # Use the FastAPI logger for the backend
        self.logger = logging.getLogger("uvicorn.error")
        self.logger.warning(PYNM_DIR)

        if self.dev:
            cors_origins = (
                ["http://localhost:" + str(self.dev_port)] if self.dev else []
            )
            # Configure CORS
            self.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Has to be before mounting static files
        self.setup_routes()

        # Serve static files
        if not self.dev:
            self.mount(
                "/",
                StaticFiles(directory=PYNM_DIR / "gui" / "frontend", html=True),
                name="static",
            )

        self.websocket_manager = WebsocketManager()
        self.pynm_state = app_pynm.PyNMState()

    def setup_routes(self):
        @self.get("/api/health")
        async def healthcheck():
            return {"message": "API is working"}

        ####################
        ##### SETTINGS #####
        ####################
        @self.get("/api/settings")
        async def get_settings(
            reset: bool = Query(False, description="Reset settings to default"),
        ):
            if reset:
                settings = NMSettings.get_default()
            else:
                settings = self.pynm_state.settings

            return settings.serialize_with_metadata()

        @self.post("/api/settings")
        async def update_settings(data: dict, validate_only: bool = Query(False)):
            try:
                # First, validate with Pydantic
                try:
                    # TODO: check if this works properly or needs model_validate_strings
                    validated_settings = NMSettings.model_validate(data)
                except ValidationError as e:
                    self.logger.error(f"Error validating settings: {e}")
                    if not validate_only:
                        # If validation failed but we wanted to upload, return error
                        raise HTTPException(
                            status_code=422,
                            detail={
                                "error": "Error validating settings",
                                "details": str(e),
                            },
                        )
                    # Else return list of errors
                    return {
                        "valid": False,
                        "errors": [err for err in e.errors()],
                        "details": str(e),
                    }

                # If validation succesful, return or update settings
                if validate_only:
                    return {
                        "valid": True,
                        "settings": validated_settings.serialize_with_metadata(),
                    }

                self.pynm_state.settings = validated_settings
                self.logger.info("Settings successfully updated")

                return {
                    "valid": True,
                    "settings": self.pynm_state.settings.serialize_with_metadata(),
                }

            # If something else than validation went wrong, return error
            except Exception as e:
                self.logger.error(f"Error validating/updating settings: {e}")
                raise HTTPException(
                    status_code=422,
                    detail={"error": "Error uploading settings", "details": str(e)},
                )

        ########################
        ##### PYNM CONTROL #####
        ########################

        @self.post("/api/stream-control")
        async def handle_stream_control(data: dict):
            action = data["action"]
            if action == "start":
                # TODO: create out_dir and experiment_name text filds in frontend
                self.logger.info("websocket:")
                self.logger.info(self.websocket_manager)
                self.logger.info("Starting stream")

                self.pynm_state.start_run_function(
                    websocket_manager=self.websocket_manager,
                )

            if action == "stop":
                self.logger.info("Stopping stream")
                self.pynm_state.stop_run_function()

            return {"message": f"Stream action '{action}' executed"}

        ####################
        ##### CHANNELS #####
        ####################

        @self.get("/api/channels")
        async def get_channels():
            channels = self.pynm_state.stream.channels
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
                self.pynm_state.stream.channels = new_channels
                return {
                    "channels": self.pynm_state.stream.channels.to_dict(
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
                    line_noise=float(data["line_noise"]),
                )
                return {"message": "Offline stream setup successfully"}
            except ValueError:
                return {"message": "Offline stream could not be setup"}

        @self.post("/api/set-stream-params")
        async def set_stream_params(data: dict):
            try:
                self.pynm_state.stream.line_noise = float(data["line_noise"])
                self.pynm_state.stream.sfreq = float(data["sampling_rate"])
                self.pynm_state.experiment_name = data["experiment_name"]
                self.pynm_state.out_dir = data["out_dir"]
                self.pynm_state.decoding_model_path = data["decoding_path"]

                return {"message": "Stream parameters updated successfully"}
            except ValueError:
                return {"message": "Stream parameters could not be updated"}

        #######################
        ### PYNM ABOUT INFO ###
        #######################

        @self.get("/api/app-info")
        # TODO: fix this function
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

        # Get PYNM_DIR
        @self.get("/api/pynm_dir")
        async def get_pynm_dir():
            try:
                return {"pynm_dir": PYNM_DIR}
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
            await self.websocket_manager.connect(websocket)
            while True:
                try:
                    await websocket.receive_text()
                except Exception:
                    break
