import os
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

class NMDatabase:
    """
    Class to create a database and insert data into it.
    Parameters
    ----------
    out_path_root : str
        The root path to save the database.
    folder_name : str
        The folder name to save the database.
    """
    def __init__(self, out_path_root, folder_name, csv_path = None):
        self.out_path_root = out_path_root
        self.folder_name = folder_name
        self.db_path = Path(out_path_root, folder_name, "stream.db") 
        if csv_path is None:
            self.csv_path = Path(out_path_root, folder_name, f"stream.csv")
        else:
            self.csv_path = Path(csv_path)

        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.cursor = self.conn.cursor()

    def infer_type(self, value):
        """Infer the type of the value to create the table schema.
        Parameters
        ----------
        value : int, float, str
            The value to infer the type."""
        
        if isinstance(value, (int, float)):
            return "REAL"
        elif isinstance(value, str):
            return "TEXT"
        else:
            return "BLOB"

    def cast_values(self, feature_dict):
        """Cast the int values of the dictionary to float.
        Parameters
        ----------
        feature_dict : dict
            The dictionary to cast the values."""
        for key, value in feature_dict.items():
            if isinstance(value, (int, float, np.int64)):
                feature_dict[key] = float(value)
        return feature_dict

    def create_table(self, feature_dict):
        """
        Create a table in the database.
        Parameters
        ----------
        feature_dict : dict
            The dictionary with the feature names and values.
        """
        columns_schema = ", ".join([f'"{column}" {self.infer_type(value)}' for column, value in feature_dict.items()])
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS stream_table ({columns_schema})")

    def insert_data(self, feature_dict):
        """
        Insert data into the database.  
        Parameters
        ----------
        feature_dict : dict
            The dictionary with the feature names and values.
        """
        columns = ", ".join([f'"{column}"' for column in feature_dict.keys()])
        placeholders = ", ".join(["?" for _ in feature_dict])
        insert_sql = f"INSERT INTO stream_table ({columns}) VALUES ({placeholders})"
        values = tuple(feature_dict.values())
        self.cursor.execute(insert_sql, values)

    def commit(self):
        self.conn.commit()

    def fetch_all(self):
        """"
        Fetch all the data from the database.
        Returns
        -------
        pd.DataFrame
            The data in a pandas DataFrame.
        """
        return pd.read_sql_query("SELECT * FROM stream_table", self.conn)
    
    def save_as_csv(self):
        df = self.fetch_all()
        df.to_csv(self.csv_path, index=False)

    def close(self):
        self.conn.close()