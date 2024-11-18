import msgpack
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
from py_neuromodulation.utils.types import _PathLike

class AbstractFileWriter(ABC):

    @abstractmethod
    def insert_data(self, feature_dict: dict):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load_all(self):
        pass

    @abstractmethod
    def save_as_csv(self, save_all_combined: bool = False):
        pass

class MsgPackFileWriter(AbstractFileWriter):
    """
    Class to store data in a serialized MessagePack file and load it back efficiently.
    Parameters
    ----------
    out_dir : _PathLike
        The directory to save the MessagePack database.
    """

    def __init__(
        self,
        name: str = "sub",
        out_dir: _PathLike = "",
    ):
        # Make sure out_dir exists

        self.out_dir = Path.cwd() if not out_dir else Path(out_dir)
        self.out_dir = self.out_dir / name

        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        self.idx = 0
        self.name = name
        self.csv_path = Path(self.out_dir, f"{name}_FEATURES.csv")
        self.data_l = []

    def insert_data(self, feature_dict: dict):
        """
        Insert data into the MessagePack database.
        Parameters
        ----------
        feature_dict : dict
            The dictionary with the feature names and values.
        """
        self.data_l.append(feature_dict)

    def save(self):
        """
        Save the current data to the MessagePack file.
        """
        if len(self.data_l) == 0:
            return
        with open(self.out_dir / f"{self.name}-{self.idx}.msgpack", "wb") as f:
            msgpack.pack(self.data_l, f)
        self.idx += 1
        self.data_l = []

    def load_all(self):
        """
        Load data from the MessagePack file into memory.
        """
        data_l = []
        for i in range(self.idx):
            with open(self.out_dir / f"{self.name}-{i}.msgpack", "rb") as f:
                data_l.append(msgpack.unpack(f))

        data = pd.DataFrame(list(np.concatenate(data_l)))
        return data

    def save_as_csv(self, save_all_combined: bool = False):
        """
        Save the data as a CSV file.
        """

        if save_all_combined:
            data = self.load_all()
            data.to_csv(self.csv_path, index=False)
        else:
            if len(self.data_l) > 0:
                self.data_l[-1].to_csv(self.csv_path, index=False)
            else:
                outpath =self.out_dir / f"{self.name}-0.msgpack"
                with open(outpath, "rb") as f:
                    data = msgpack.unpack(f)
                data.to_csv(self.csv_path, index=False)
