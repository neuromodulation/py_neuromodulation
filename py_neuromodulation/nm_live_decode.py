import joblib
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

class LiveDecoder:
    def __init__(self: str):
        """
        Initializes the LiveDecoder with the path to the model.

        :param model_path: Path to the joblib-saved sklearn model.
        """
        self.model_path = "./sub/model.joblib"
        self.model: BaseEstimator = self.load_model()

    def load_model(self) -> BaseEstimator:
        """
        Loads the model from the given path using joblib.

        :return: The loaded sklearn model.
        """
        try:
            model = joblib.load(self.model_path)
            return model
        except Exception as e:
            raise ValueError(f"Error loading the model from {self.model_path}: {e}")

    def get_results(self, feature_series: pd.DataFrame) -> np.ndarray:
        """
        Uses the loaded model to predict the results for the given data batch.

        :param feature_series: The data batch for which to run predictions.
        :return: The predicted results.
        """
        try:
            results = self.model.predict(pd.DataFrame(feature_series).drop(columns=['time']))
            print(results[-1])
            return type(results)
        except Exception as e:
            raise ValueError(f"Error predicting results: {e}")
