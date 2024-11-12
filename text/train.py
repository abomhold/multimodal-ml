import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
# import text.preprocessing as pre
import text.clean as clean
from pathlib import Path
import preprocessing


# Load dataset
target = "gender"
df = preprocessing.get_baseline(
    Path("../data/training/profile/profile.csv"),
    Path("../data/training/LIWC/LIWC.csv"))
print(df)

df = clean.main(Path("../data/training/text"), df)
print(df)




data = df[['text', target]]

x = data['text']
y = data[target]

# # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(data['text'], data[target], test_size=1, random_state=42)
# # Create a pipeline with CountVectorizer and Naive Bayes
# # pipeline = Pipeline([#     ('count_vec', CountVectorizer()),
# #     ('nb', MultinomialNB()),# ])# ('svm', SVC(kernel='rbf')),
#
# # Create a pipeline with CountVectorizer and Naive Bayes
# pipeline = Pipeline([
#     ('count_vec', TfidfVectorizer()),
#     ('nb', LogisticRegression(random_state=42, max_iter=100))
# ])
#
# # Perform 10-fold cross-validation
# cv_scores = cross_val_score(pipeline, x, y, cv=10)
# print("10-fold CV Average Accuracy: {:.3f} (+/- {:.3f})".format(cv_scores.mean(), cv_scores.std() * 2))
#
# # Train and test model
# pipeline.fit(x, y)
# y_pred = pipeline.predict(x)
# print(y_pred)
#
# # Print confusion matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(y, y_pred))
#
# # Print accuracy score
# print("\nTest Set Accuracy:", accuracy_score(y, y_pred))
#
# # Save the model
# import joblib
#
# joblib.dump(pipeline, 'model.pkl')