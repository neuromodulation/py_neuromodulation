from dataclasses import dataclass, field
import json
import pathlib
import pprint
import random
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
import yaml


@dataclass
class BIDSEntity:
    name: str
    key: str
    default_value: str
    string_var: tk.StringVar = field(init=False)

    def __post_init__(self) -> None:
        self.string_var = tk.StringVar(value=self.default_value)

    @property
    def value(self) -> str:
        return self.string_var.get()

    @property
    def full(self) -> str:
        if self.key == "datatype":
            return self.string_var.get()
        return f"{self.key}-{self.string_var.get()}"


def main() -> None:

    # root = tkinter.Tk()
    out_root = tkinter.filedialog.askdirectory(
        title="Select output root for results."
    )
    out_root = pathlib.Path(out_root)

    root = tk.Tk()
    root.title("Enter info")

    mainframe = ttk.Frame(root, padding="10 10 30 10")
    mainframe.grid(column=0, row=0, sticky=("N", "W", "E", "S"))  # type: ignore
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    entities = {
        "sub": BIDSEntity("Subject", "sub", f"{random.randint(0, 999):03d}"),
        "ses": BIDSEntity("Session", "ses", "EcogLfpMedOff01"),
        "task": BIDSEntity("Task", "task", "RealtimeDecodingR"),
        "acq": BIDSEntity("Acquisition", "acq", "StimOff"),
        "run": BIDSEntity("Run", "run", "1"),
        "datatype": BIDSEntity("Datatype", "datatype", "ieeg"),
    }

    for row, entity in enumerate(entities.values()):
        tk.Label(mainframe, text=entity.name).grid(
            row=row, column=0, stick="W"
        )
        entry = tk.Entry(mainframe, textvariable=entity.string_var)
        entry.grid(row=row, column=1, sticky="E")

    for child in mainframe.winfo_children():
        child.grid_configure(padx=5, pady=5)

    exit_button = tk.Button(mainframe, text="Done", command=root.destroy)
    exit_button.grid(row=len(entities), column=1, sticky=("S", "E"))

    root.mainloop()

    out_dir = out_root / entities["sub"].full / entities["ses"].full
    out_dir.mkdir(exist_ok=True, parents=True)
    settings = {key: entity.value for key, entity in entities.items()}
    settings = {"root": str(out_root), "out_dir": str(out_dir)} | settings
    this_dir = pathlib.Path(__file__).parent
    with open(this_dir / "bids_settings.json", "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4)

    filename = "_".join([entity.full for entity in entities.values()])
    # fullpath = out_dir / filename

    timeflux_yaml = this_dir / "timeflux_decoding.yaml"
    with open(timeflux_yaml, "r") as file:
        timeflux_settings = yaml.load(file, yaml.loader.SafeLoader)

    # graph: "DataRecorder", node: "save"
    saving_settings = timeflux_settings["graphs"][0]["nodes"][0]["params"]
    saving_settings["filename"] = f"{filename}.hdf5"
    saving_settings["path"] = str(out_dir)

    pprint.pprint(timeflux_settings)


if __name__ == "__main__":
    main()
