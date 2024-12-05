import os
from pathlib import Path

import pandas as pd

def get_age_group(age):
    """Convert numerical age to standardized age group format."""
    age = float(age)
    if age <= 24:
        return 'xx-24'
    elif age <= 34:
        return '25-34'
    elif age <= 49:
        return '35-49'
    else:
        return '50-xx'
    
def majority(text_df, image_df, like_df):
    combined_df = pd.concat([text_df, image_df, like_df], axis=0, keys=['text_df', 'image_df', 'like_df']).reset_index(
        level=0,
        drop=True)

    majority_df = combined_df.groupby('userid').agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])

    return majority_df.reset_index()


def write_xml(path: Path, data: pd.DataFrame):
    if not os.path.exists(path):
        os.mkdir(path)
    data['age'] = data['age'].apply(get_age_group)
    for row in data.iterrows():
        row_to_xml(row[1], path)


def row_to_xml(row: pd.Series, path: Path):

    xml_string = (f"<user id=\"{row['userid']}\" "
                  f"age_group=\"{row['age']}\" "
                  f"gender=\"{'male' if row['gender'] == 0 else 'female'}\" "
                  f"extrovert=\"{row['ext']}\" "
                  f"neurotic=\"{row['neu']}\" "
                  f"agreeable=\"{row['agr']}\" "
                  f"conscientiousness=\"{row['con']}\" "
                  f"open=\"{row['ope']}\" />")

    print(xml_string)
    with open(f"{path}{row['userid']}.xml", "x") as f:
        f.write(xml_string)
