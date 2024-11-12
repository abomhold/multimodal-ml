from pathlib import Path

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
    data.rename(columns={'userId': 'userid'}, inplace=True)
    return data


def get_baseline(profile_path, lwic_path) -> pd.DataFrame:
    pro = profile_cvs(profile_path).set_index("userid")
    lwic = lwic_cvs(lwic_path).set_index("userid")
    data = pd.merge(pro, lwic, left_index=True, right_index=True)
    data = data.reset_index()
    return data


def main():
    return get_baseline(config.PROFILE_PATH, config.LIWC_PATH)


if __name__ == '__main__':
    main()
