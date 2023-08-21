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

def write_timeflux_settings(filename: str, out_save_dir: _PathLike, out_root: _PathLike):

    this_dir = pathlib.Path(__file__).parent
    with open(this_dir / "timeflux_decoding_template.yaml", "r") as file:
        timeflux_settings = yaml.load(file, yaml.loader.SafeLoader)

    # graph: "DataRecorder", node: "save"
    #tf_saving_settings = timeflux_settings["graphs"][0]["nodes"][0]["params"]
    timeflux_settings["graphs"][0]["nodes"][0]["params"]["filename"] = f"{filename}.hdf5"
    timeflux_settings["graphs"][0]["nodes"][0]["params"]["path"] = str(out_save_dir)

    with open(pathlib.Path(out_root) / "timeflux_decoding.yaml", "w") as file:
        yaml.dump(timeflux_settings, file, sort_keys=False)

def parse_settings(yaml_settings):

    out_root = pathlib.Path(yaml_settings["PATH_OUT_DIR"])

    bids_list_order = ["sub", "ses", "task", "acq", "run", "datatype"]

    out_dir = out_root / yaml_settings["sub"] / yaml_settings["ses"]
    out_dir.mkdir(exist_ok=True, parents=True)

    filename = out_dir / "_".join([str(yaml_settings[entity]) for entity in bids_list_order])
    yaml_settings["filename"] = str(filename)

    with open(out_root / "config.yaml", "w") as file:
        yaml.dump(yaml_settings, file, default_flow_style=False)
    
    return yaml_settings

if __name__ == "__main__":

    yaml_settings = IO_config_yaml()    
    settings = parse_settings(yaml_settings)
    write_timeflux_settings(
        os.path.basename(settings["filename"]),
        os.path.dirname(settings["filename"]),
        r"C:\CODE\py_neuromodulation\realtime_experiment"
    )

    realtime_decoding.run(settings)
