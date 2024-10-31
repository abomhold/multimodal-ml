from pathlib import Path

from pandas import DataFrame

import text.clean as preprocessing
import text.test as test


def write_predictions(data, results):
    data["gender"] = results
    return data


def main(text_dir: Path, data: DataFrame) -> DataFrame:
    data = preprocessing.main(text_dir, data)
    results = test.main(data)
    data = write_predictions(data, results)
    return data
