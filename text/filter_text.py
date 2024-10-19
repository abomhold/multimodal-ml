import json
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# df = pd.read_csv("../training/LIWC/LIWC.csv")
target = "gender"
df = pd.read_csv("cleaned_text.csv")
data = df.loc[:, ['text', target]]
# Splitting the data into 300 training instances and 104 test instances
n = 104
all_Ids = np.arange(len(data))
random.shuffle(all_Ids.tolist())
test_Ids = all_Ids[0:n]
train_Ids = all_Ids[n:]
data_test = data.loc[test_Ids, :]
data_train = data.loc[train_Ids, :]

# Training a Naive Bayes model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['text'].fillna(''))
y_train = data_train[target]
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Testing the Naive Bayes model
X_test = count_vect.transform(data_test['text'])
y_test = data_test[target]
y_predicted = clf.predict(X_test)

# Reporting on classification performance
print("Accuracy: %.2f" % accuracy_score(y_test, y_predicted))
# classes = ['Male', 'Female']
# cnf_matrix = confusion_matrix(y_test, y_predicted, labels=classes)
# print("Confusion matrix:")
# print(cnf_matrix)
