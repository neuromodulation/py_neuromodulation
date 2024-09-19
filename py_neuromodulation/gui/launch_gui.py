import subprocess
import threading
from pathlib import Path
from time import sleep

def run_flask():
    subprocess.run([".venv_windows/Scripts/python", "py_neuromodulation/gui/app.py"])

def run_vite():
    subprocess.run(["bun", "run", "dev"], cwd="gui_dev")
    
if __name__ == "__main__":
    print(Path.cwd())
    flask_thread = threading.Thread(target=run_flask)
    vite_thread = threading.Thread(target=run_vite)

    flask_thread.start()
    sleep(5)
    vite_thread.start()

    flask_thread.join()
    vite_thread.join()