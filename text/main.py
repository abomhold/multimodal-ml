from pathlib import Path

import joblib
from pandas import DataFrame

import text.clean as preprocessing


def write_predictions(data, results):
    data["gender"] = results
    return data


def load_model() -> DataFrame:
    return joblib.load('text/model.pkl')


def test(data: DataFrame) -> DataFrame:
    model = load_model()
    result = model.predict(data['text'])
    return result


def main(text_dir: Path, data: DataFrame) -> DataFrame:
    data = preprocessing.main(text_dir, data)
    results = test(data)
    data = write_predictions(data, results)
    return data
