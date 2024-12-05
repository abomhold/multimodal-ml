from pathlib import Path
import get_cloud
import pandas as pd

import config


def profile_cvs(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path).dropna()
    data = data.drop(columns=["Unnamed: 0"], errors='ignore')

    data["userid"] = data["userid"]
    data["age"] = data["age"].apply(lambda x: 20 if x == "-" else x)
    data["gender"] = data["gender"].apply(lambda x: 1 if x == "-" else x)
    data["ope"] = data["ope"].apply(lambda x: 3.9 if x == "-" else x)
    data["con"] = data["con"].apply(lambda x: 3.4 if x == "-" else x)
    data["ext"] = data["ext"].apply(lambda x: 3.4 if x == "-" else x)
    data["agr"] = data["agr"].apply(lambda x: 3.5 if x == "-" else x)
    data["neu"] = data["neu"].apply(lambda x: 2.7 if x == "-" else x)
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
    #get_cloud.main()
    data = combine_data(profile_cvs(config.PROFILE_PATH), lwic_cvs(config.LIWC_PATH))
    return data


if __name__ == '__main__':
    main()
