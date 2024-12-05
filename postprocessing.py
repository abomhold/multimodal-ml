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
        row_to_xml(row[1], path)


def row_to_xml(row: pd.Series, path: Path):
    age_value = int(row['age'])
    xml_string = (f"<user id=\"{row['userid']}\" "
                  f"age_group=\"xx-{age_value}\" "
                  f"gender=\"{'male' if row['gender'] == 0 else 'female'}\" "
                  f"extrovert=\"{row['ext']}\" "
                  f"neurotic=\"{row['neu']}\" "
                  f"agreeable=\"{row['agr']}\" "
                  f"conscientiousness=\"{row['con']}\" "
                  f"open=\"{row['ope']}\" />")

    print(xml_string)
    with open(f"{path}{row['userid']}.xml", "x") as f:
        f.write(xml_string)
