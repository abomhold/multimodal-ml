import preprocessing
from pathlib import Path
from pandas import DataFrame


def main(text_dir: Path, data: DataFrame) -> DataFrame:
    data = preprocessing.main(text_dir, data)
    print(data)
    data = data.dropna()
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    return data