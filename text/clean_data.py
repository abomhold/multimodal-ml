import os
from pathlib import Path
import re
import unicodedata
from typing import Dict

import pandas as pd
import nltk

# Ensure necessary downloads for nltk
nltk.download(['punkt', 'stopwords', 'wordnet'], quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stop words set
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Clean and normalize the input text."""
    # Remove URLs
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'(\w)(\1{2,})', lambda m: m.group(1) + m.group(1), text)
    # Remove repeated whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize Unicode characters
    # text = re.sub(r'(\W)\1{2,}', r'\1', text)
    # Tokenize, remove stop words, and lemmatize
    words = [lemmatizer.lemmatize(word) for word in word_tokenize(text)
              if word.lower() not in stop_words]

    return ' '.join(words).lower()


def process_text_files(text_path: Path) -> Dict[str, str]:
    """Process text files in the given directory."""
    user_texts = {}
    for file_path in text_path.glob('*.txt'):
        user_id = file_path.stem
        with file_path.open('r', encoding='utf-8', errors='ignore') as text_file:
            content = text_file.read()
            user_texts[user_id] = clean_text(content)
    return user_texts


def main():
    # Paths
    base_path = Path(__file__).parent.parent
    text_path = base_path / "training" / "text"
    profile_path = base_path / "training" / "profile" / "profile.csv"
    output_path = base_path / "text" / "cleaned_text.csv"

    # Read the profile CSV
    df = pd.read_csv(profile_path, index_col="userid")
    df = df.drop(columns=["Unnamed: 0"], errors='ignore')

    # Process text files
    user_texts = process_text_files(text_path)

    # Create a new DataFrame with the texts
    text_df = pd.DataFrame.from_dict(user_texts, orient='index', columns=['text'])
    text_df.index.name = 'userid'

    # Join the profile DataFrame with the text DataFrame
    combined_df = df.join(text_df, how='inner').reset_index()
    combined_df = combined_df.dropna()

    # Write to CSV
    combined_df.to_csv(output_path, index=False)
    print(f"\nData has been written to {output_path}")
    print(f"Number of rows in the final CSV: {len(combined_df)}")


if __name__ == "__main__":
    main()
# import os
# import re
# import unicodedata
#
# import pandas as pd
# import nltk as tk
# # Ensure necessary downloads for nltk
# tk.download('punkt')
# tk.download('punkt_tab')
# tk.download('stopwords')
# tk.download('wordnet')
#
# # Initialize lemmatizer and stop words list
# lemmatizer = tk.WordNetLemmatizer()
# stop_words = set(tk.corpus.stopwords.words('english'))
#
#
# def clean_text(text):
#     # Decode, encode, and clean the text
#     text = text.decode('utf8', "ignore")
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     # Shuffle indices for train/test split
#
#     text = re.sub(r'[^a-zA-Z\s]', ' ', text, flags=re.MULTILINE)
#     text = re.sub(r'(\W)\1{2,}', r'\1', text)
#     print(text)
#     text = (unicodedata
#             .normalize('NFD', text)
#             )
#     words = tk.word_tokenize(text)
#     words = [word for word in words if word not in stop_words]
#     words = [lemmatizer.lemmatize(word) for word in words]
#     return ' '.join(words).lower().encode('ascii', 'ignore').decode('ascii')
#
#
#
# # Paths
# text_path = "../training/text"
# profile_path = "../training/profile/profile.csv"
#
# # Read the profile CSV
# df = pd.read_csv(profile_path, index_col="userid")
#
# # Remove 'Unnamed: 0' column if it exists
# if "Unnamed: 0" in df.columns:
#     df = df.drop("Unnamed: 0", axis=1)
#
# # Dictionary to store user texts
# user_texts = {}
#
# # Process each text file
# for filename in os.listdir(text_path):
#     user_id = filename.split('.')[0]
#     filepath = os.path.join(text_path, filename)
#
#     with open(filepath, "rb") as text_file:
#         content = text_file.read()
#         cleaned_content = clean_text(content)
#         user_texts[user_id] = cleaned_content
#
# # Create a new DataFrame with the texts
# text_df = pd.DataFrame.from_dict(user_texts, orient='index', columns=['text'])
# text_df.index.name = 'userid'
#
# # Join the profile DataFrame with the text DataFrame
# combined_df = (df
#                .join(text_df, how='inner')
#                .reset_index()
#                )
#
# combined_df = combined_df.dropna()
#
# print(combined_df.iloc[1])
#
# # Write to CSV
# output_path = "../text/cleaned_text.csv"
# combined_df.to_csv(output_path, index=False)
#
# print(f"\nData has been written to {output_path}")
# print(f"Number of rows in the final CSV: {len(combined_df)}")
