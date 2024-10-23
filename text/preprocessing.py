import re
from pathlib import Path
from typing import Dict

import nltk
import pandas as pd
from nltk import word_tokenize


# Ensure necessary downloads for nltk
nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'], quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

happy = [":‑)", ":)", ":-]", ":]", ":->", ":>", "8-)", "8)", ":-}", ":}", ":^)", "=]", "=)", ":‑D", ":D", "8‑D", "8D",
         "=D", "=3", "B^D", "c:", "C:", "x‑D", "xD", "X‑D", "XD", ":-))", "^:))"]
sad = [":‑(", ":(", ":‑c", ":c", ":‑ < ", ": < ", ":‑[", ": [", ":- | | ", ": {", ": @ ", ":(", ";(", ":'‑(", ":'(",
       " := ("]


def clean_text(text: str) -> str:
    # text = unicodedata.normalize('NFD', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub('|'.join(map(re.escape, happy)), ' happy ', text)
    text = re.sub('|'.join(map(re.escape, sad)), ' sad ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'(\w)(\1{2,})', lambda m: m.group(1) + m.group(1), text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)

    words = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word.lower() not in stop_words]

    return ' '.join(words).lower()


def process_text_files(text_path: Path) -> Dict[str, str]:
    user_texts = {}
    for file_path in text_path.glob('*.txt'):
        user_id = file_path.stem
        with file_path.open('r', encoding='utf-8', errors='ignore') as text_file:
            content = text_file.read()
            user_texts[user_id] = clean_text(content)
    return user_texts


def main(path: Path, data: pd.DataFrame):
    # Process text files
    user_texts = process_text_files(path)
    # Create a new DataFrame with the texts
    text_df = pd.DataFrame.from_dict(user_texts, orient='index', columns=['text'])
    text_df.index.name = 'userid'
    data.set_index('userid', inplace=True)
    combined_df = data.join(text_df, how='inner').reset_index()
    combined_df = combined_df.dropna()
    return combined_df
