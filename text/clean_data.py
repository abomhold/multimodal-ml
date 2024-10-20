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
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'(\w)(\1{2,})', lambda m: m.group(1) + m.group(1), text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)

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
    combined_df = df.join(text_df, how='inner').reset_index()
    combined_df = combined_df.dropna()

    # Write to CSV
    combined_df.to_csv(output_path, index=False)
    print(f"\nData has been written to {output_path}")
    print(f"Number of rows in the final CSV: {len(combined_df)}")


if __name__ == "__main__":
    main()
