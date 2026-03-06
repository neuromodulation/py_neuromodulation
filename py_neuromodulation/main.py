"""
This script helps to start the backend and creating a webview.
A .dmg can be created by running the following


    pyinstaller --noconsole --onefile --name NeuromodApp main.py

    -> Will use this script to run the backend and serve the FE files

    hdiutil create -volname "NeuromodApp" -srcfolder dist/NeuromodApp.app -ov -format UDZO NeuromodApp.dmg

    -> Will create a .dmg file

"""

import threading
import uvicorn
import webview
from .run_gui import main   # import your existing script

from py_neuromodulation.stream import LSLOfflinePlayer
from gui.backend.app_manager import run_uvicorn

SERVER_PORT = 8000
HOST = "localhost"

def start_backend():
    player = LSLOfflinePlayer(raw=raw, stream_name="example_stream")
    player.start_player(chunk_size=30, n_repeat=5999999)

    run_uvicorn(debug=False, reload=True, server_port=SERVER_PORT)

if __name__ == "__main__":
    t = threading.Thread(target=start_backend, daemon=True)
    t.start()
    webview.create_window("Neuromodulation App", "http://{}:{}".format(HOST, SERVER_PORT))
    webview.start()
