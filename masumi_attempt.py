import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf

PUNCTUATION = nltk.download('punkt_tab')
STOPWORDS = nltk.download('stopwords')


def preprocess(cleaned_csv="./cleaned.csv", liwc="./data/training/LIWC/LIWC.csv"):
    cleand_df = pd.read_csv(cleaned_csv).set_index('userid').drop(columns=['Seg'])
    cleand_df['words'] = cleand_df['words'].fillna("").astype(str)
    filtered_words = []
    for sentence in cleand_df['words']:
        words = word_tokenize(sentence)
        words = [word for word in words if word.isalpha() and word not in set(stopwords.words('english'))]
        filtered_words.extend(words)

    return filtered_words


def create_tokenizer(filtered_words):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(filtered_words)
    return tokenizer


if __name__ == '__main__':
    result = preprocess()
    print(result)

    tokenizer = create_tokenizer(result)
    print(tokenizer)