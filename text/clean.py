import re
from pathlib import Path
from typing import Dict

import charset_normalizer
import nltk
import pandas as pd
from nltk import word_tokenize

import text.emojis as emojis_py

emojis: Dict[list[str], str] = emojis_py.emoticons

# Ensure necessary downloads for nltk
nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'], quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    for key in emojis:
        text = re.sub('|'.join(map(re.escape, key)), f" {emojis[key]} emoji", text)

    text = re.sub(r'[^a-zA-Z\s]', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'(\w)(\1{2,})', lambda m: m.group(1) + m.group(1), text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)

    words = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word.lower() not in stop_words]
    return ' '.join(words).lower()


def process_text_files(text_path: Path) -> (Dict[str, str], Dict[str, str]):
    user_texts = {}
    user_words = {}
    for file_path in Path(text_path).glob('*.txt'):
        user_id = file_path.stem
        user_texts[user_id] = ''
        result = charset_normalizer.from_path(file_path)

        if result.first() is not None:
            encoding = result.first().encoding
        else:
            print(f"Could not determine encoding for {file_path}")
            print(f"Using default encoding iso-8859-1")
            encoding = 'iso-8859-1'

        with file_path.open('r', encoding=encoding) as text_file:
            content = text_file.read()
            user_words[user_id] = clean_text(content)
            user_texts[user_id] = content

    return user_texts, user_words


def main(path: Path, data: pd.DataFrame):
    # Read text files into a dictionary
    user_texts, user_words = process_text_files(path)
    # Transform the dictionary into a DataFrame
    # For original text:
    text_df = pd.DataFrame.from_dict(user_texts, orient='index', columns=['text'])
    text_df.index.name = 'userid'
    # For cleaned text:
    word_df = pd.DataFrame.from_dict(user_words, orient='index', columns=['words'])
    word_df.index.name = 'userid'
    # Combine text and word DataFrames
    text_df = text_df.join(word_df, how='inner', on='userid')
    # Combine text DataFrame with original data
    data.set_index('userid', inplace=True)
    combined_df = data.join(text_df, how='inner', on='userid')
    combined_df.reset_index(inplace=True)
    return combined_df
