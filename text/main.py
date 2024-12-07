import sys
from pathlib import Path
import text.personality_prediction as pp
from pandas import DataFrame

import text.clean as preprocessing


def main(text_dir: Path, data: DataFrame) -> DataFrame:
    data = preprocessing.main(text_dir, data)
    print(data['text'])
    print(data.columns)
    return pp.predict(data)
