import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from like.preprocessdata import preprocess_likes

def predict_gender(relation_path: str, profile_path: str) -> pd.DataFrame:
    """
    Predict gender based on likes data and return results as a DataFrame.
    
    Args:
        relation_path (str): Path to the relation CSV file
        profile_path (str): Path to the profile CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing userid, actual gender, and predicted gender
    """
    # Load and preprocess data
    X, y = preprocess_likes(relation_path, profile_path)
    
    # Get original userids (we'll need these for the final DataFrame)
    relation = pd.read_csv(relation_path)
    profile = pd.read_csv(profile_path)
    user_data = profile[['userid', 'gender']].drop_duplicates()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Keep track of which users are in test set
    test_indices = y_test.index
    
    # Vectorize
    vec = CountVectorizer(min_df=10)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    
    # Train model
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    
    # Make predictions
    predictions = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'userid': user_data.iloc[test_indices]['userid'],
        'actual_gender': y_test,
        'predicted_gender': predictions
    })
    
    # Add accuracy as metadata
    results_df.attrs['model_accuracy'] = accuracy
    
    return results_df
