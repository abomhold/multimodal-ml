import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Load dataset
target = "gender"
df = pd.read_csv("cleaned_text.csv").dropna(subset=['text', target])
data = df[['text', target]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data[target], test_size=0.3, random_state=42)

# Create a pipeline with CountVectorizer and Naive Bayes
pipeline = Pipeline([
    ('count_vec', CountVectorizer()),
    ('nb', MultinomialNB()),
])

# Create a pipeline with CountVectorizer and Naive Bayes
# pipeline = Pipeline([
#     ('count_vec', TfidfVectorizer()),
#     # ('svm', SVC(kernel='rbf')),
#     ('nb', LogisticRegression(random_state=42, max_iter=100))
# ])

# Perform 10-fold cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=10)
print("10-fold CV Average Accuracy: {:.3f} (+/- {:.3f})".format(cv_scores.mean(), cv_scores.std() * 2))

# Train and test model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(y_pred)

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print accuracy score
print("\nTest Set Accuracy:", accuracy_score(y_test, y_pred))
