from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def preprocess_likes(relation_path: Path, data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess likes data from relation file and merge with user data. a
    """
    # Ensure the path exists
    relation_file = Path(relation_path)
    if not relation_file.exists():
        print(f"Warning: No relation file found at {relation_file}")
        data['likes'] = None
        return data

    # Load relation data
    relation = pd.read_csv(relation_file)

    # Group likes by user
    likes = (relation.groupby('userid')['like_id'].agg(lambda x: ' '.join(map(str, x))).reset_index().rename(columns={
        'like_id': 'likes'}))

    # Merge with user data
    return pd.merge(data, likes, on='userid', how='left')


def train_model(df: pd.DataFrame) -> tuple:
    """
    Train gender prediction model on preprocessed likes data.
    """
    vec = CountVectorizer(min_df=10)
    X_vec = vec.fit_transform(df['likes'])

    clf = MultinomialNB()
    clf.fit(X_vec, df['gender'])

    return vec, clf


def predict(df: pd.DataFrame, vec: CountVectorizer, clf: MultinomialNB) -> np.ndarray:
    """
    Make predictions using trained model.
    """
    X_vec = vec.transform(df['likes'])
    return clf.predict(X_vec)


def write_predictions(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """
    Add predictions to DataFrame.
    """
    df = df.copy()
    df['predicted_gender'] = predictions
    print(df)
    return df


def predict_gender(relation_dir: Path, data: pd.DataFrame) -> pd.DataFrame:
    """
    Main function for gender prediction from likes data.
    
    Parameters:
    -----------
    relation_dir : Path
        Directory containing relation.csv file
    data : pd.DataFrame
        DataFrame containing user data with 'userid' and 'gender' columns

    Returns:
    --------
    pd.DataFrame
        Input DataFrame with added gender predictions
    """
    # Check for required input columns
    if 'userid' not in data.columns or 'gender' not in data.columns:
        raise ValueError("Input DataFrame must contain 'userid' and 'gender' columns")

    # Preprocess likes data
    data = preprocess_likes(relation_dir, data)

    # Handle missing likes
    if 'likes' not in data.columns or data['likes'].isna().all():
        print("Warning: No likes data found for any users")
        data['predicted_gender'] = data['gender']  # fallback to original gender
        return data

    # Remove rows with missing likes
    data_with_likes = data.dropna(subset=['likes'])
    if len(data_with_likes) == 0:
        print("Warning: No valid likes data found")
        data['predicted_gender'] = data['gender']  # fallback to original gender
        return data

    # Split into train/test
    train_df, test_df = train_test_split(data_with_likes, test_size=0.2, random_state=42)

    # Train model
    vectorizer, model = train_model(train_df)

    # Make predictions on test set
    test_predictions = predict(test_df, vectorizer, model)
    print(f"TEST PREDICTION{test_predictions}")
    # Calculate and print accuracy
    accuracy = accuracy_score(test_df['gender'], test_predictions)
    print(f"Like-based gender prediction accuracy: {accuracy:.2f}")

    # Make predictions on all users with likes
    predictions = predict(data_with_likes, vectorizer, model)

    # Create result DataFrame with predictions
    data.loc[data_with_likes.index, 'gender'] = predictions

    # Fill missing predictions with original gender
    # data['gender'].fillna(data['gender'], inplace=True)

    # data.attrs['model_accuracy'] = accuracy

    return data
