import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# Load dataset and clean NaN values
target = "gender"
df = pd.read_csv("cleaned_text.csv").dropna(subset=['text', target])
data = df[['text', target]].fillna("")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data[target], test_size=0.3, random_state=42)

# Create a pipeline with CountVectorizer and Naive Bayes
pipeline = Pipeline([
    ('count_vec', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Perform 10-fold cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=10)
print("10-fold CV Average Accuracy: {:.3f} (+/- {:.3f})".format(cv_scores.mean(), cv_scores.std() * 2))

# Fit the pipeline on the entire training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print accuracy score
print("\nTest Set Accuracy:", accuracy_score(y_test, y_pred))
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as IMBPipeline
#
# # Load dataset and clean NaN values
# target = "gender"
# df = pd.read_csv("cleaned_text.csv").dropna(subset=['text', target])
# data = df[['text', target]].fillna("")
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data['text'], data[target], test_size=0.2, random_state=42)
#
# # Create a pipeline with TF-IDF, SMOTE, and Naive Bayes
# pipeline = IMBPipeline([
#     ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
#     ('smote', SMOTE(random_state=42)),
#     ('nb', MultinomialNB())
# ])
#
# # Define parameters for GridSearchCV
# param_grid = {
#     'tfidf__ngram_range': [(1, 1), (1, 2)],
#     'tfidf__max_df': [0.5, 0.75, 1.0],
#     'nb__alpha': [0.1, 0.5, 1.0]
# }
#
# # Perform GridSearchCV
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
# grid_search.fit(X_train, y_train)
#
# # Print best parameters and score
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation score:", grid_search.best_score_)
#
# # Make predictions on the test set
# y_pred = grid_search.predict(X_test)
#
# # Print classification report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
#
# # Print confusion matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
#
# # Print accuracy score
# print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
# # import json
# # import random
# # import pandas as pd
# # import numpy as np
# # from sklearn.feature_extraction.text import CountVectorizer
# # from sklearn.model_selection import cross_val_score
# # from sklearn.naive_bayes import MultinomialNB
# # from sklearn.metrics import accuracy_score, confusion_matrix
# #
# # # Load dataset and clean NaN values
# # target = "gender"
# # df = pd.read_csv("cleaned_text.csv").dropna(subset=['text', target])
# # data = df[['text', target]].fillna("")  # Modifying the dataframe in place
# # # print(data)
# #
# # # Check a sample row
# # # print(df.iloc[58])
# #
# # # Break in to Test/Train
# # n = 3200
# # all_Ids = np.arange(len(data))
# # np.random.shuffle(all_Ids)  # Shuffle indices directly
# # data_test = data.iloc[all_Ids[:n], :]
# # data_train = data.iloc[all_Ids[n:], :]
# #
# # # Training a Naive Bayes model
# # count_vect = CountVectorizer()
# # X_train = count_vect.fit_transform(data_train['text'])
# # y_train = data_train[target]
# # clf = MultinomialNB()
# # clf.fit(X_train, y_train)
# #
# # # Testing the Naive Bayes model
# # X_test = count_vect.transform(data_test['text'])
# # y_test = data_test[target]
# # y_predicted = clf.predict(X_test)
# #
# # # 10 - cross fold validation
# # # Perform 10-fold cross-validation
# # cv_scores = cross_val_score(clf, X_train, y_train, cv=10)
# # # Print the mean cross-validation score
# # print("10-fold CV Average Accuracy: {:.3f}".format(cv_scores.mean()))
# #
# # # Reporting on classification performance
# # print("Accuracy: %.2f" % accuracy_score(y_test, y_predicted))
# #
# # # Confusion matrix
# # classes = sorted(data[target].unique())  # Automatically find class labels
# # cnf_matrix = confusion_matrix(y_test, y_predicted, labels=classes)
# # print("Confusion matrix:")
# # print(cnf_matrix)
# #
# # # import json
# # # import random
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.feature_extraction.text import CountVectorizer
# # # from sklearn.naive_bayes import MultinomialNB
# # # from sklearn.metrics import accuracy_score
# # # from sklearn.metrics import confusion_matrix
# # #
# # # # df = pd.read_csv("../training/LIWC/LIWC.csv")
# # # target = "gender"
# # # df = pd.read_csv("cleaned_text.csv").dropna()
# # # data = df.loc[:, ['text', target]]
# # # data.fillna("", inplace=False)
# # # print(data)
# # #
# # # n = 300
# # # print(df.iloc[58])
# # # all_Ids = np.arange(len(data))
# # # random.shuffle(all_Ids.tolist())
# # # test_Ids = all_Ids[0:n]
# # # train_Ids = all_Ids[n:]
# # # print(test_Ids)
# # # data_test = data.loc[test_Ids, :]
# # # data_train = data.loc[train_Ids, :]
# # #
# # # # Training a Naive Bayes model
# # # count_vect = CountVectorizer()
# # # X_train = count_vect.fit_transform(data_train['text'])
# # # y_train = data_train[target]
# # # clf = MultinomialNB()
# # # clf.fit(X_train, y_train)
# # #
# # # # Testing the Naive Bayes model
# # # X_test = count_vect.transform(data_test['text'])
# # # y_test = data_test[target]
# # # y_predicted = clf.predict(X_test)
# # #
# # # # Reporting on classification performance
# # # print("Accuracy: %.2f" % accuracy_score(y_test, y_predicted))
# # # # classes = ['Male', 'Female']
# # # # cnf_matrix = confusion_matrix(y_test, y_predicted, labels=classes)
# # # # print("Confusion matrix:")
# # # # print(cnf_matrix)
