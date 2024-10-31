import glob
import os

import pandas as pd
from pathlib import Path


def write_xml(path: Path, data: pd.DataFrame):
    if not os.path.exists(path):
        os.mkdir(path)
        
    for row in data.iterrows():
        row_to_xml(row, path)


def row_to_xml(row, path: Path):
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
                  f"extrovert=\"{ext}\" "
                  f"neurotic=\"{neu}\" "
                  f"agreeable=\"{agr}\" "
                  f"conscientiousness=\"{con}\" "
                  f"open=\"{ope}\" />")

    print(xml_string)
    with open(f"{path}/{userid}.xml", "x") as f:
        f.write(xml_string)
