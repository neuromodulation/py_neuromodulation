import sqlite3
from pathlib import Path
import pandas as pd
from py_neuromodulation.nm_types import _PathLike
from py_neuromodulation.nm_IO import generate_unique_filename


class NMDatabase:
    """
    Class to create a database and insert data into it.
    Parameters
    ----------
    out_dir : _PathLike
        The directory to save the database.
    csv_path : str, optional
        The path to save the csv file. If not provided, it will be saved in the same folder as the database.
    """

    def __init__(
        self,
        name: str,
        out_dir: _PathLike,
        csv_path: _PathLike | None = None,
    ):
        # Make sure out_dir exists
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        self.db_path = Path(out_dir, f"{name}.db")

        self.table_name = f"{name}_data"  # change to param?
        self.table_created = False

        if self.db_path.exists():
            self.db_path = generate_unique_filename(self.db_path)
            name = self.db_path.stem

        if csv_path is None:
            self.csv_path = Path(out_dir, f"{name}.csv")
        else:
            self.csv_path = Path(csv_path)

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Database config and optimization, prioritize data integrity
        self.cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging mode
        self.cursor.execute("PRAGMA synchronous=FULL")  # Sync on every commit
        self.cursor.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
        self.cursor.execute(
            "PRAGMA wal_autocheckpoint = 1000"
        )  # WAL checkpoint every 1000 pages (default, 4MB, might change)
        self.cursor.execute(
            f"PRAGMA mmap_size = {2 * 1024 * 1024 * 1024}"
        )  # 2GB of memory mapped

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

    def create_table(self, feature_dict: dict):
        """
        Create a table in the database.
        Parameters
        ----------
        feature_dict : dict
            The dictionary with the feature names and values.
        """
        columns_schema = ", ".join(
            [
                f'"{column}" {self.infer_type(value)}'
                for column, value in feature_dict.items()
            ]
        )

        self.cursor.execute(
            f'CREATE TABLE IF NOT EXISTS "{self.table_name}" ({columns_schema})'
        )

        # Create column names and placeholders for insert statement
        self.columns: str = ", ".join([f'"{column}"' for column in feature_dict.keys()])
        # Use named placeholders for more resiliency against unexpected change in column order
        self.placeholders = ", ".join([f":{key}" for key in feature_dict.keys()])

    def insert_data(self, feature_dict: dict):
        """
        Insert data into the database.
        Parameters
        ----------
        feature_dict : dict
            The dictionary with the feature names and values.
        """

        if not self.table_created:
            self.create_table(feature_dict)
            self.table_created = True

        insert_sql = f'INSERT INTO "{self.table_name}" ({self.columns}) VALUES ({self.placeholders})'

        self.cursor.execute(insert_sql, feature_dict)

    def commit(self):
        self.conn.commit()

    def fetch_all(self):
        """ "
        Fetch all the data from the database.
        Returns
        -------
        pd.DataFrame
            The data in a pandas DataFrame.
        """
        return pd.read_sql_query(f'SELECT * FROM "{self.table_name}"', self.conn)

    def head(self, n: int = 5):
        """ "
        Returns the first N rows of the database.
        Parameters
        ----------
        n : int, optional
            The number of rows to fetch, by default 1
        -------
        pd.DataFrame
            The data in a pandas DataFrame.
        """
        return pd.read_sql_query(
            f'SELECT * FROM "{self.table_name}" LIMIT {n}', self.conn
        )

    def save_as_csv(self):
        df = self.fetch_all()
        df.to_csv(self.csv_path, index=False)

    def close(self):
        # Optimize before closing is recommended:
        # https://www.sqlite.org/pragma.html#pragma_optimize
        self.cursor.execute("PRAGMA optimize")
        self.conn.close()
