import os
from pathlib import Path
import pandas as pd
import config


def profile_cvs(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.drop(columns=["Unnamed: 0"], errors='ignore')
    return data


def main(path: Path) -> pd.DataFrame:
    return profile_cvs(path)

