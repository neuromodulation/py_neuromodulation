import os
import sqlite3
import time
from pathlib import Path
import numpy as np
import pandas as pd
import glob

class NMDatabase:
    def __init__(self, out_path_root, folder_name):
        self.out_path_root = out_path_root
        self.folder_name = folder_name
        self.db_time_idx = int(time.time() * 1000)
        self.db_path = Path(out_path_root, folder_name, f"stream{self.db_time_idx}.db")
        self.csv_path = Path(out_path_root, folder_name, f"stream.csv")
        
        pattern = str(Path(out_path_root, folder_name, "stream*.db"))

        for file_path in glob.glob(pattern):
            if os.path.exists(file_path):
                if (self.db_time_idx/1000) - os.path.getctime(file_path) > 600:
                    os.remove(file_path)

        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.cursor = self.conn.cursor()

        db_dir = Path(folder_name)
        if os.path.exists(db_dir):
            os.chmod(db_dir, 0o777)
        if os.path.exists(self.db_path):
            os.chmod(self.db_path, 0o777)
        if os.path.exists(Path(out_path_root, folder_name)):
            os.chmod(Path(out_path_root, folder_name), 0o777)

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
        columns_schema = ", ".join([f'"{column}" {self.infer_type(value)}' for column, value in feature_dict.items()])
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS stream_table ({columns_schema})")

    def insert_data(self, feature_dict):
        columns = ", ".join([f'"{column}"' for column in feature_dict.keys()])
        placeholders = ", ".join(["?" for _ in feature_dict])
        insert_sql = f"INSERT INTO stream_table ({columns}) VALUES ({placeholders})"
        values = tuple(feature_dict.values())
        self.cursor.execute(insert_sql, values)

    def commit(self):
        self.conn.commit()

    def fetch_all(self):
        return pd.read_sql_query("SELECT * FROM stream_table", self.conn)
    
    def save_as_csv(self):
        df = self.fetch_all()
        df.to_csv(self.csv_path, index=False)

    def close(self):
        self.conn.close()