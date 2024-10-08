import os
import pandas as pd

PROFILE_PATH = "training/profile/profile.csv"
OUTPUT_PATH = "output"


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

    os.mkdir(OUTPUT_PATH)
    for row in data.iterrows():
        row_to_xml(row)


if __name__ == "__main__":
    build_baseline()
