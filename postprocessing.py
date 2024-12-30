import os
from cgitb import reset
from pathlib import Path
import csv
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
    data.reset_index(inplace=True)
    print(data.columns)
    for row in data.iterrows():
        row_to_xml(row[1], path)


def row_to_xml(row: pd.Series, path: Path):

    xml_string = (f"<user id=\"{row['userid']}\" "
                  f"age_group=\"{row['age_range']}\" "
                  f"gender=\"{'male' if row['gender'] == 0 else 'female'}\" "
                  f"extrovert=\"{row['ext']}\" "
                  f"neurotic=\"{row['neu']}\" "
                  f"agreeable=\"{row['agr']}\" "
                  f"conscientiousness=\"{row['con']}\" "
                  f"open=\"{row['ope']}\" />")

    print(xml_string)
    with open(f"{path}/{row['userid']}.xml", "x") as f:
        f.write(xml_string)


def export_confusion_matrices(results, output_dir='./data/output/'):
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    for pred_type, data in results.items():
        if data is None or not isinstance(data, dict):
            print(f"Warning: Skipping {pred_type} - invalid data format")
            continue
            
        cm_df = data.get('confusion_matrix')
        if not isinstance(cm_df, pd.DataFrame):
            print(f"Warning: No valid DataFrame for {pred_type}")
            continue
            
        try:
            filename = f'{pred_type}_confusion_matrix.csv'
            filepath = os.path.join(output_dir, filename)
            cm_df.to_csv(filepath, index=True)
            created_files.append(filepath)
        except Exception as e:
            print(f"Error saving confusion matrix for {pred_type}: {e}")

    return created_files
