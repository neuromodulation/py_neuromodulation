import logging
from pathlib import Path
import toml
from collections import defaultdict

import pandas as pd
import numpy as np

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

import py_neuromodulation as nm

from app_socket import PyNMSocket
from random_data_generator import RandomGenerator


class PyNMState:
    stream: nm.Stream = nm.Stream(sfreq=1500, data=np.random.random([1, 1]))
    settings: nm.NMSettings = nm.NMSettings(sampling_rate_features=17)


class PyNMApp:
    def __init__(self):
        # Intialize PyNM state
        self.state = PyNMState()
        self.app = Flask(
            "PyNeuromodulationGUI",
            static_folder="frontend",
            template_folder="templates",
        )
        CORS(self.app)  # Share resources between frontend and backend

        self.socketio = PyNMSocket(
            self.app,
            cors_allowed_origins="*",
            async_mode="threading",
        )

        # Catch-all endpoint that routes all requests to the frontend
        @self.app.route("/", defaults={"path": ""})
        @self.app.route("/<path:path>")
        def serve(path):
            print(self.app.static_folder)
            if path != "" and Path(self.app.static_folder, path).exists():
                return send_from_directory(self.app.static_folder, path)
            else:
                return send_from_directory(self.app.static_folder, "index.html")

        @self.app.route("/health")
        def healthcheck():
            return {"message": "API is working"}

        @self.app.route("/settings", methods=["GET", "POST"])  # type: ignore
        def handle_settings():
            if request.method == "GET":
                return jsonify(self.state.settings.model_dump())
            elif request.method == "POST":
                data = request.get_json()
                try:
                    self.state.settings = nm.NMSettings.model_validate(data)
                    if self.app.debug:
                        print(self.state.settings.features)
                    return jsonify(self.state.settings.model_dump())
                except ValueError as e:
                    # https://flask.palletsprojects.com/en/3.0.x/errorhandling/
                    return jsonify(
                        {"error": "Validation failed", "details": e.args}
                    ), 422  # 422 Unprocessable Entity

        @self.app.route("/channels", methods=["GET", "POST"])  # type: ignore
        def handle_channels():
            if request.method == "GET":
                # Generate the HTML table from the DataFrame
                channels_html = self.state.stream.nm_channels.to_html(
                    index=False, classes="table table-bordered"
                )
                return jsonify({"html": channels_html})
            elif request.method == "POST":
                # Update the DataFrame with the received data
                data = request.get_json()
                self.state.stream.nm_channels = pd.DataFrame(data)
                return jsonify({"message": "Channels updated successfully"})

        @self.app.route("/stream-control", methods=["GET", "POST"])  # type: ignore
        def handle_stream_control():
            data = request.get_json()
            action = data["action"]
            if request.method == "GET":
                pass
            elif request.method == "POST":
                match action:
                    case "start":
                        self.state.stream.run()
                    # case "stop":
                    #     self.state.stream.stop()
                    # case "reset":
                    #     self.state.stream.reset()

        @self.app.route("/api/app-info")
        def get_app_info():
            pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
            with open(pyproject_path, "r") as f:
                pyproject_data = toml.load(f)

            project_info = defaultdict(lambda: "", pyproject_data.get("project", {}))
            urls = defaultdict(str, project_info.get("urls", {}))

            return jsonify(
                {
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
                    "launchMode": "debug" if self.app.debug else "release",
                }
            )

    def run_app(self):
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)  # Disable dev server warning

        generator = RandomGenerator(self.socketio)

        self.socketio.run(self.app, debug=True, port=5000)  # type: ignore
