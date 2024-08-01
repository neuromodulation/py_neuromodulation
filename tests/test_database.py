from py_neuromodulation import nm_database
import pandas as pd

def test_db_setup():
    nm_db = nm_database.NMDatabase(
        name="test_db",
        out_dir="test_data",   
    )

    assert nm_db.db_path.exists()

    features = {
        "feature_1": 1,
        "feature_2": 2,
        "feature_3": 3,
    }
    nm_db.create_table(features)

    # read table
    table = nm_db.cursor.execute(f"SELECT * FROM {nm_db.table_name}")

    assert table.fetchone() == None
    
    assert [info[0] for info in table.description] == ["feature_1", "feature_2", "feature_3"]

    nm_db.insert_data(features)

    entry = nm_db.head(1)

    nm_db.save_as_csv()

    assert nm_db.csv_path.exists()
    assert pd.read_csv(nm_db.csv_path).equals(entry)
