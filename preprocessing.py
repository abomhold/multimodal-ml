import argparse
import os
from pathlib import Path

import gdown
import pandas as pd

import config


# def download_cloud_assets():
#     if not os.path.exists("cloud_assets"):
#         os.mkdir("cloud_assets")
#     gdown.download(config.CLOUD_ASSETS_URL + config.CLOUD_ASSETS_ID, output="cloud_assets.zip", quiet=False, fuzzy=True)
#     os.system("unzip cloud_assets.zip -d cloud_assets")


def profile_cvs(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path).dropna()
    data = data.drop(columns=["Unnamed: 0"], errors='ignore')
    data["userid"] = data["userid"].astype(str)
    data["age"] = data["age"].astype(str)
    data["gender"] = data["gender"].astype(str)
    data["ope"] = data["ope"].astype(float)
    data["con"] = data["con"].astype(float)
    data["ext"] = data["ext"].astype(float)
    data["agr"] = data["agr"].astype(float)
    data["neu"] = data["neu"].astype(float)
    return data


def lwic_cvs(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path).dropna()
    return data


def combine_data(profile: pd.DataFrame, lwic: pd.DataFrame) -> pd.DataFrame:
    profile.set_index("userid")
    lwic.set_index("userId")
    data = pd.merge(profile, lwic, left_index=True, right_index=True)
    return data


def get_baseline(profile_path, lwic_path) -> pd.DataFrame:
    data = combine_data(profile_cvs(profile_path), lwic_cvs(lwic_path))
    return data


def main():
    # download_cloud_assets()
    data = combine_data(profile_cvs(config.PROFILE_PATH), lwic_cvs(config.LIWC_PATH))
    return data


if __name__ == '__main__':
    main()
