import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split  # Better for splitting data
from sklearn.feature_selection import SelectKBest, chi2
from preprocessdata import preprocess_likes
X, y = preprocess_likes(
    relation_path="./training/relation/relation.csv",
    profile_path="./training/profile/profile.csv"
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorize (keeping sparse matrix)
vec = CountVectorizer(min_df=10)
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# Train and predict
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)
accuracy = clf.score(X_test_vec, y_test)

print(f"Accuracy: {accuracy:.3f}")