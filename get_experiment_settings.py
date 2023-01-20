from dataclasses import dataclass, field
import tkinter as tk
from tkinter import ttk


@dataclass
class BIDSEntity:
    name: str
    key: str
    default_value: str
    string_var: tk.StringVar = field(init=False)

    def __post_init__(self) -> None:
        self.string_var = tk.StringVar(value=self.default_value)

    @property
    def full(self) -> str:
        if self.key == "datatype":
            return self.string_var.get()
        return f"{self.key}-{self.string_var.get()}"


def main() -> None:
    root = tk.Tk()
    # root.geometry("400x300")
    root.title("Enter info")

    mainframe = ttk.Frame(root, padding="10 10 30 10")
    mainframe.grid(column=0, row=0, sticky=("N", "W", "E", "S"))  # type: ignore
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    entities = (
        BIDSEntity("Subject", "sub", ""),
        BIDSEntity("Session", "ses", "EcogLfpMedOff01"),
        BIDSEntity("Task", "task", "RealtimeDecodingR"),
        BIDSEntity("Acquisition", "acq", "StimOff"),
        BIDSEntity("Run", "run", "1"),
        BIDSEntity("Datatype", "datatype", "ieeg"),
    )

    for row, entity in enumerate(entities):
        tk.Label(mainframe, text=entity.name).grid(
            row=row, column=0, stick="W"
        )
        entry = tk.Entry(mainframe, textvariable=entity.string_var)
        entry.grid(row=row, column=1, sticky="E")
        # entries[label] = entry

    for child in mainframe.winfo_children():
        child.grid_configure(padx=5, pady=5)

    exit_button = tk.Button(mainframe, text="Done", command=root.destroy)
    exit_button.grid(row=len(entities), column=1, sticky=("S", "E"))

    root.mainloop()

    filename = "_".join([entity.full for entity in entities])
    print(filename)


if __name__ == "__main__":
    main()
