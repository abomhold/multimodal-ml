import os
from cgitb import reset
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


def write_xml(path: Path, data: pd.DataFrame):
    if not os.path.exists(path):
        os.mkdir(path)
    data['age'] = data['age'].apply(get_age_group)
    data.reset_index(inplace=True)
    print(data.columns)
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
    with open(f"{path}/{row['userid']}.xml", "x") as f:
        f.write(xml_string)
