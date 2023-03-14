import json
import pathlib
import yaml
import os

import realtime_decoding

_PathLike = str | os.PathLike


def IO_config_yaml():
    with open("config.yaml", "r") as stream:
        try:
            yaml_settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_settings

def write_timeflux_settings(filename: str, out_dir: _PathLike, out_root: _PathLike):

    this_dir = pathlib.Path(__file__).parent
    with open(this_dir / "timeflux_decoding_template.yaml", "r") as file:
        timeflux_settings = yaml.load(file, yaml.loader.SafeLoader)

    # graph: "DataRecorder", node: "save"
    tf_saving_settings = timeflux_settings["graphs"][0]["nodes"][0]["params"]
    tf_saving_settings["filename"] = f"{filename}.hdf5"
    tf_saving_settings["path"] = str(out_dir)

    with open(out_root / "timeflux_decoding.yaml", "w") as file:
        yaml.dump(tf_saving_settings, file, sort_keys=False)

def read_yaml_settings()

def parse_settings(yaml_settings):

    out_root = yaml_settings["PATH_OUT"]

    bids_list_order = ["sub", "ses", "task", "acq", "run", "datatype"]

    out_dir = out_root / yaml_settings["sub"].full / yaml_settings["ses"].full
    out_dir.mkdir(exist_ok=True, parents=True)

    filename = "_".join([entity.full for entity in bids_list_order])
    yaml_settings["filename"] = filename

    with open(out_root / "config.yaml", "w") as file:
        yaml.dump(yaml_settings, file, default_flow_style=False)
    
    return settings

if __name__ == "__main__":

    yaml_settings = IO_config_yaml()    
    settings = parse_settings(yaml_settings)
    realtime_decoding.run(settings)
