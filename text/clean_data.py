import os
import re
import pandas as pd


def clean_text(text):
    # Decode, encode, and clean the text
    cleaned = (text
               .decode("utf8", "ignore")
               .encode("ascii", "ignore")
               .decode("ascii", "ignore")
               .strip()
               .lower()
               )
    # Remove all non-alphabetic characters except spaces
    return re.sub(r'[^a-z\s]', '', cleaned)


# Paths
text_path = "../training/text"
profile_path = "../training/profile/profile.csv"

# Read the profile CSV
df = pd.read_csv(profile_path, index_col="userid")

# Remove 'Unnamed: 0' column if it exists
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# Dictionary to store user texts
user_texts = {}

# Process each text file
for filename in os.listdir(text_path):
    user_id = filename.split('.')[0]
    filepath = os.path.join(text_path, filename)

    with open(filepath, "rb") as text_file:
        content = text_file.read()
        cleaned_content = clean_text(content)
        user_texts[user_id] = cleaned_content

# Create a new DataFrame with the texts
text_df = pd.DataFrame.from_dict(user_texts, orient='index', columns=['text'])
text_df.index.name = 'userid'

# Join the profile DataFrame with the text DataFrame
combined_df = df.join(text_df, how='inner')

# Reset index to make 'userid' a column
combined_df = combined_df.reset_index()

# Drop rows with NA values
combined_df = combined_df.dropna()

# Print some information for verification
print("Shape of the combined DataFrame after dropping NA:", combined_df.shape)
print("\nColumns in the combined DataFrame:", combined_df.columns.tolist())
print("\nFirst few rows of the combined DataFrame:")
print(combined_df.head())

# Write to CSV
output_path = "../text/cleaned_text.csv"
combined_df.to_csv(output_path, index=False)

print(f"\nData has been written to {output_path}")
print(f"Number of rows in the final CSV: {len(combined_df)}")
