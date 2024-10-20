import os
import pandas as pd
import config


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

    xml_string = (f"<user id=\"{userid}\" "
                  f"age_group=\"xx-{age}\" "
                  f"gender=\"{gender}\" "
                  f"extrovert=\"{ext:.3f}\" "
                  f"neurotic=\"{neu:.3f}\" "
                  f"agreeable=\"{agr:.3f}\" "
                  f"conscientiousness=\"{con:.3f}\" "
                  f"open=\"{ope:.3f}\" />")

    # print(xml_string)
    with open(f"{config.OUTPUT_PATH}/{userid}.xml", "w") as f:
        f.write(xml_string)


def build_baseline():
    data = pd.read_csv(config.INPUT_PATH + "profile/profile.csv")
    data["gender"] = data["gender"].mode()[0]
    data["age"] = data["age"].mode()[0]
    data["ope"] = data["ope"].mean()
    data["con"] = data["con"].mean()
    data["ext"] = data["ext"].mean()
    data["agr"] = data["agr"].mean()
    data["neu"] = data["neu"].mean()

    for row in data.iterrows():
        row_to_xml(row)


if __name__ == "__main__":
    build_baseline()
