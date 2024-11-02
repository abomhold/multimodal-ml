import os
from pathlib import Path

import pandas as pd


def majority(text_df, image_df, like_df):
    combined_df = pd.concat([text_df, image_df, like_df], axis=0, keys=['text_df', 'image_df', 'like_df']).reset_index(
        level=0,
        drop=True)

    majority_df = combined_df.groupby('userid').agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])

    return majority_df.reset_index()


def write_xml(path: Path, data: pd.DataFrame):
    if not os.path.exists(path):
        os.mkdir(path)

    for row in data.iterrows():
        row_to_xml(row, path)


def row_to_xml(row, path: Path):
    row = row[1]

    if row["gender"] == 0:
        gender = "male"
    else:
        gender = "female"

    xml_string = (f"<user id=\"{row["userid"].astype(str)}\" "
                  f"age_group=\"xx-{row["age"].astype(str)}\" "
                  f"gender=\"{gender}\" "
                  f"extrovert=\"{row["ext"].astype(float)}\" "
                  f"neurotic=\"{row["neu"].astype(float)}\" "
                  f"agreeable=\"{row["agr"].astype(float)}\" "
                  f"conscientiousness=\"{row["con"].astype(float)}\" "
                  f"open=\"{row["ope"].astype(float)}\" />")

    print(xml_string)
    with open(f"{path}/{row["userid"].astype(str)}.xml", "x") as f:
        f.write(xml_string)
