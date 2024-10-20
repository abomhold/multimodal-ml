import json
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset and clean NaN values
target = "gender"
df = pd.read_csv("cleaned_text.csv").dropna(subset=['text', target])
data = df[['text', target]].fillna("")  # Modifying the dataframe in place
print(data)

# Check a sample row
print(df.iloc[58])

# Shuffle indices for train/test split
n = 300
all_Ids = np.arange(len(data))
np.random.shuffle(all_Ids)  # Shuffle indices directly

# Split into train and test sets
test_Ids = all_Ids[:n]
train_Ids = all_Ids[n:]
print(test_Ids)

data_test = data.iloc[test_Ids, :]
data_train = data.iloc[train_Ids, :]

# Training a Naive Bayes model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['text'])
y_train = data_train[target]
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Testing the Naive Bayes model
X_test = count_vect.transform(data_test['text'])
y_test = data_test[target]
y_predicted = clf.predict(X_test)

# Reporting on classification performance
print("Accuracy: %.2f" % accuracy_score(y_test, y_predicted))

# Confusion matrix
classes = sorted(data[target].unique())  # Automatically find class labels
cnf_matrix = confusion_matrix(y_test, y_predicted, labels=classes)
print("Confusion matrix:")
print(cnf_matrix)
0
# import json
# import random
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
#
# # df = pd.read_csv("../training/LIWC/LIWC.csv")
# target = "gender"
# df = pd.read_csv("cleaned_text.csv").dropna()
# data = df.loc[:, ['text', target]]
# data.fillna("", inplace=False)
# print(data)
#
# n = 300
# print(df.iloc[58])
# all_Ids = np.arange(len(data))
# random.shuffle(all_Ids.tolist())
# test_Ids = all_Ids[0:n]
# train_Ids = all_Ids[n:]
# print(test_Ids)
# data_test = data.loc[test_Ids, :]
# data_train = data.loc[train_Ids, :]
#
# # Training a Naive Bayes model
# count_vect = CountVectorizer()
# X_train = count_vect.fit_transform(data_train['text'])
# y_train = data_train[target]
# clf = MultinomialNB()
# clf.fit(X_train, y_train)
#
# # Testing the Naive Bayes model
# X_test = count_vect.transform(data_test['text'])
# y_test = data_test[target]
# y_predicted = clf.predict(X_test)
#
# # Reporting on classification performance
# print("Accuracy: %.2f" % accuracy_score(y_test, y_predicted))
# # classes = ['Male', 'Female']
# # cnf_matrix = confusion_matrix(y_test, y_predicted, labels=classes)
# # print("Confusion matrix:")
# # print(cnf_matrix)
