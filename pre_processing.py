import pandas as pd
from config import PROFILE_PATH

from config import PROFILE_PATH, OUTPUT_PATH


def get_most_gender():
    data = pd.read_csv(PROFILE_PATH)
    mode_gender = data["gender"].mode()[0]
    return mode_gender


def get_most_age():
    data = pd.read_csv(PROFILE_PATH)
    mode_age = data["age"].mode()[0]
    return mode_age


def get_avg_personality():
    data = pd.read_csv(PROFILE_PATH)
    avg_ope = data["ope"].mean()
    ave_con = data["con"].mean()
    ave_ext = data["ext"].mean()
    ave_agr = data["agr"].mean()
    ave_neu = data["neu"].mean()

    return avg_ope, ave_con, ave_ext, ave_agr, ave_neu


def row_to_xml(row):
    row = row[1]

    userid, age, gender = (
        row["userid"],
        row["age"],
        row["gender"]
    )

    ope, con, ext, agr, neu = (
        row["ope"],
        row["con"],
        row["ext"],
        row["agr"],
        row["neu"]

    )

    with open(f"{OUTPUT_PATH}/{userid}.xml", "w") as f:
        f.write(f"<user id=\"{userid}\" "
                f"age_group=\"xx-{age}\" "
                f"gender=\"{gender}\" "
                f"extrovert=\"{ext:.3f}\" "
                f"neurotic=\"{neu:.3f}\" "
                f"agreeable=\"{agr:.3f}\" "
                f"conscientiousness=\"{con:.3f}\" "
                f"open=\"{ope:.3f}\" />")


def build_baseline():
    data = pd.read_csv(PROFILE_PATH)
    data["gender"] = data["gender"].mode()[0]
    data["age"] = data["age"].mode()[0]
    data["ope"] = data["ope"].mean()
    data["con"] = data["con"].mean()
    data["ext"] = data["ext"].mean()
    data["agr"] = data["agr"].mean()
    data["neu"] = data["neu"].mean()

    for row in data.iterrows():
        row_to_xml(row)


if __name__ == '__main__':
    build_baseline()
