import joblib
from pandas import DataFrame


def load_model() -> DataFrame:
    return joblib.load('text/model.pkl')


def main(data: DataFrame) -> DataFrame:
    model = load_model()
    result = model.predict(data['text'])
    return result
