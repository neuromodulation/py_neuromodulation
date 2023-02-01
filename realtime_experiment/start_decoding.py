import json
import pathlib

import realtime_decoding


if __name__ == "__main__":
    # out_dir = r"C:\Users\richa\GitHub\py_neuromodulation\data\test"
    this_dir = pathlib.Path(__file__).parent
    with open(this_dir / "bids_settings.json", "r") as file:
        settings = json.load(file) 
    
    realtime_decoding.run(settings["out_dir"], settings["filename"])