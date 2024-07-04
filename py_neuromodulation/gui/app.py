import pandas as pd
import numpy as np

from pathlib import Path 

import webview
from flask import Flask, jsonify, render_template, send_from_directory, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from concurrent.futures import ThreadPoolExecutor
import threading
import py_neuromodulation as nm

class PyNMState():
    stream : nm.Stream = nm.Stream(sfreq=1500, data=np.random.random([1,1]))
    settings : nm.NMSettings = nm.NMSettings(sampling_rate_features = 17)
    
    
    
class PyNMApp():
    def __init__(self):
        # Intialize PyNM state
        self.state = PyNMState()
        
        self.app = Flask(__name__, static_folder="frontend", template_folder="templates")
        CORS(self.app)
        
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        @self.socketio.on("connect")
        def handle_connect():
            # emit('message', {'data': 'Connected'})
            print("Client connected")

        @self.socketio.on("request_data")
        def handle_request_data():
            # data = compute_data()  # Your function to compute data
            # packed_data = msgpack.packb(data)
            data = np.random.random(1000).tobytes()
            self.socketio.emit("binary_data", data)

        
        # Catch-all endpoint that routes all requests to the frontend
        @self.app.route("/", defaults={"path": ""})
        @self.app.route('/<path:path>')
        def serve(path):
            print(self.app.static_folder)
            if path != "" and Path(self.app.static_folder, path).exists():
                return send_from_directory(self.app.static_folder, path)
            else:
                return send_from_directory(self.app.static_folder, 'index.html')

        @self.app.route('/health')
        def healthcheck():
            return {"message": "API is working"}
        
        @self.app.route('/settings', methods=['GET', 'POST'])
        def handle_settings():
            if request.method == 'GET':
                return jsonify(self.state.settings.model_dump())
            elif request.method == 'POST':
                data = request.get_json()
                try:
                    self.state.settings = nm.NMSettings.model_validate(data)
                    print(self.state.settings.features)
                    return jsonify(self.state.settings.model_dump())
                except ValueError as e:
                    # https://flask.palletsprojects.com/en/3.0.x/errorhandling/
                    return jsonify({
                        "error": "Validation failed",
                        "details": e.args
                    }), 422  # 422 Unprocessable Entity

    def run_app(self):

        self.socketio.run(self.app, debug=True, port=5000) # type: ignore
                     
                     
if __name__ == "__main__":
    app = PyNMApp()
    app.run_app()
  





# def start_server():
#     app.run(host="127.0.0.1", port=5000)


# if __name__ == "__main__":
#     t = threading.Thread(target=start_server)
#     t.daemon = True
#     t.start()

#     webview.create_window(app.name, "http://127.0.0.1:5000") # type: ignore
#     webview.start()

#     # flask_future = executor.submit(run_flask)

#     # run_app()
