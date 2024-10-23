import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split  # Better for splitting data

# 1. Load and prepare data
relation = pd.read_csv("./training/relation/relation.csv")
profile = pd.read_csv('./training/profile/profile.csv')

# 2. Aggregate likes per user
likes_by_user = relation.groupby('userid')['like_id'].agg(list).reset_index()

# 3. Merge with gender data
merged_df = pd.merge(likes_by_user, 
                    profile[['userid', 'gender']], 
                    on='userid', 
                    how='inner')

# 4. Convert like_ids to string format for CountVectorizer
merged_df['likes_str'] = merged_df['like_id'].apply(lambda x: ' '.join(map(str, x)))

# 5. Split data into train and test (better way)
X_data = merged_df['likes_str']
y_data = merged_df['gender']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    random_state=42)

# 6. Convert likes to feature vectors
count_vect = CountVectorizer()
X_train_vectors = count_vect.fit_transform(X_train).toarray()
X_test_vectors = count_vect.transform(X_test).toarray()

# 7. Train model
clf = GaussianNB()
clf.fit(X_train_vectors, y_train)

# 8. Make predictions
y_pred = clf.predict(X_test_vectors)

# 9. Evaluate
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# Optional: Look at most predictive features
feature_names = count_vect.get_feature_names_out()
feature_importance = np.abs(clf.theta_[0] - clf.theta_[1])
top_features = pd.DataFrame({
    'like_id': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(10)

print("\nMost predictive likes:")
print(top_features)