import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def preprocess_likes(relation_path: str, profile_path: str) -> tuple:
    """
    Simple preprocessing of likes data for gender prediction.
    Returns features (likes) and labels (gender).
    """
    # Load data
    relation = pd.read_csv(relation_path)
    profile = pd.read_csv(profile_path)
    
    # Group likes by user
    likes = (relation.groupby('userid')['like_id']
            .agg(lambda x: ' '.join(map(str, x)))
            .reset_index())
    
    # Merge with gender
    data = pd.merge(likes, profile[['userid', 'gender']], on='userid')
    
    return data['like_id'], data['gender']

def train_and_save_model(relation_path: str, profile_path: str, model_save_path: str):
    """Train the model and save it for later use."""
    # Preprocess data
    likes, gender = preprocess_likes(relation_path, profile_path)
    
    # Create and fit vectorizer
    vectorizer = CountVectorizer(min_df=5)  # Ignore terms that appear in less than 5 documents
    X = vectorizer.fit_transform(likes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, gender, test_size=0.2, random_state=42, stratify=gender
    )
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")
    
    # Save model and vectorizer
    with open(model_save_path, 'wb') as f:
        pickle.dump((vectorizer, model), f)
    print(f"Model saved to {model_save_path}")

def predict_gender(new_relation_path: str, model_path: str) -> pd.DataFrame:
    """Predict gender for new data using the pretrained model."""
    # Load the model and vectorizer
    with open(model_path, 'rb') as f:
        vectorizer, model = pickle.load(f)
    
    # Load and preprocess new data
    relation = pd.read_csv(new_relation_path)
    likes = (relation.groupby('userid')['like_id']
             .agg(lambda x: ' '.join(map(str, x)))
             .reset_index())
    
    # Transform likes using the pretrained vectorizer
    X_new = vectorizer.transform(likes['like_id'])
    
    # Make predictions
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)
    
    # Create results dataframe
    results = pd.DataFrame({
        'userid': likes['userid'],
        'predicted_gender': predictions,
        'prediction_probability': np.max(probabilities, axis=1)
    })
    
    return results

# Example usage
if __name__ == "__main__":
    # Training phase
    train_and_save_model(
        relation_path="data/training/relation/relation.csv",
        profile_path="data/training/profile/profile.csv",
        model_save_path="like/gender_prediction_model.pkl"
    )
    