from py_neuromodulation.utils.types import _PathLike
from py_neuromodulation.utils import logger, io
from pathlib import Path

class DataWriter:
    
    def __init__(self, out_dir: _PathLike = "", save_csv: bool = False,
        save_interval: int = 10, experiment_name: str = "experiment"):

        self.batch_count: int = 0
        self.save_interval: int = save_interval
        self.save_csv: bool = save_csv
        self.out_dir: _PathLike = out_dir
        self.experiment_name: str = experiment_name

        self.out_dir_root = Path.cwd() if not out_dir else Path(out_dir)
        self.out_dir = self.out_dir_root / self.experiment_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        from py_neuromodulation.utils.database import NMDatabase
        self.db = NMDatabase(self.experiment_name, out_dir)

        logger.log_to_file(out_dir)


    def write_data(self, feature_dict):

        self.db.insert_data(feature_dict)
        self.batch_count += 1
        if self.batch_count % self.save_interval == 0:
            self.db.commit()

    def get_features(self, return_df: bool = False):
 
        self.db.commit()  # Save last batches

        # If save_csv is False, still save the first row to get the column names
        feature_df = (
            self.db.fetch_all() if (self.save_csv or return_df) else self.db.head()
        )

        self.db.close()
        return feature_df

    def save_csv_features(
        self,
        df_features: "pd.DataFrame"
    ) -> None:
        filename = f"{self.experiment_name}_FEATURES.csv" if self.experiment_name else "_FEATURES.csv"
        io.write_csv(df_features, self.out_dir / filename)
        logger.info(f"{filename} saved to {str(self.out_dir)}")
