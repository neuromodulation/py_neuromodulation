from abc import ABC, abstractmethod
from typing import Tuple


class DataGeneratorABC(ABC):

    def __init__(self) -> Tuple[float, "pd.DataFrame"]:
        pass

    @abstractmethod
    def __next__(self) -> Tuple["np.ndarray", "np.ndarray"]:
        pass
