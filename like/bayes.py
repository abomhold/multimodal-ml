# like/bayes.py

import pickle

import numpy as np
import pandas as pd

import config

MODEL_PATH = "like/gender_prediction_model.pkl"


def predict_gender(relation_path: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict gender using pretrained model.
    Args:
        relation_path: Path to relation CSV file
        data: DataFrame containing user data
        MODEL_PATH: Path to saved model file
    Returns:
        DataFrame with predictions
    """
    global MODEL_PATH
    print("\nProcessing likes data...")
    result_df = data.copy()

    try:
        # Load model and vectorizer
        with open(MODEL_PATH, 'rb') as f:
            vectorizer, model = pickle.load(f)

        # Load and process relation data
        relation = pd.read_csv(relation_path)
        likes = (relation.groupby('userid')['like_id'].agg(lambda x: ' '.join(map(str, x))).reset_index())

        # Merge with input data
        merged_data = pd.merge(data, likes, on='userid', how='left')
        valid_data = merged_data.dropna(subset=['like_id'])

        if len(valid_data) == 0:
            print("Warning: No valid likes data found")
            return result_df

        # Transform likes using pretrained vectorizer
        X_new = vectorizer.transform(valid_data['like_id'])

        # Make predictions
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)

        # Update original dataframe with predictions
        result_df.loc[valid_data.index, 'gender'] = predictions
        result_df.loc[valid_data.index, 'confidence'] = np.max(probabilities, axis=1)

        print("\nPrediction distribution:")
        print(pd.Series(predictions).value_counts(normalize=True))

    except Exception as e:
        print(f"Error in likes-based prediction: {str(e)}")
        return result_df

    return result_df
