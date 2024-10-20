import os
import re
import unicodedata

import pandas as pd
import nltk as tk
# Ensure necessary downloads for nltk
tk.download('punkt')
tk.download('stopwords')
tk.download('wordnet')

# Initialize lemmatizer and stop words list
lemmatizer = tk.WordNetLemmatizer()
stop_words = set(tk.corpus.stopwords.words('english'))

def clean_text(text):
    # Decode, encode, and clean the text
    text = text.decode('utf8', "ignore")
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text, flags=re.MULTILINE)

    text = (unicodedata
            .normalize('NFD', text)
            )
    words = tk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words).lower().encode('ascii', 'ignore').decode('ascii')


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
combined_df = (df
               .join(text_df, how='inner')
               .reset_index()
               )

combined_df = combined_df.dropna()

print(combined_df.iloc[1])

# Write to CSV
output_path = "../text/cleaned_text.csv"
combined_df.to_csv(output_path, index=False)

print(f"\nData has been written to {output_path}")
print(f"Number of rows in the final CSV: {len(combined_df)}")
