import pickle
import numpy as np
import pandas as pd
from pathlib import Path

class UserTraitsPredictor:
    def __init__(self, model_path: str = "like/user_traits_prediction_models.pkl"):
        self.model_path = model_path
        self.load_models()
    
    def load_models(self):
        """Load the saved models and preprocessing objects."""
        try:
            with open(self.model_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.vectorizer = saved_data['vectorizer']
                self.models = saved_data['models']
                self.label_encoder = saved_data['label_encoder']
                self.regression_traits = saved_data['regression_traits']
                self.classification_traits = saved_data['classification_traits']
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")

def predict_all(relation_path: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict user traits using pretrained model.
    Args:
        relation_path: Path to relation CSV file
        data: DataFrame containing user data
    Returns:
        DataFrame with predictions
    """
    print("\nProcessing likes data...")
    result_df = data.copy()

    try:
        # Initialize predictor with saved models
        predictor = UserTraitsPredictor()

        # Load and process relation data
        relation = pd.read_csv(relation_path)
        likes = (relation.groupby('userid')['like_id']
                .agg(lambda x: ' '.join(map(str, x)))
                .reset_index())
        
        # Merge with input data
        merged_data = pd.merge(data, likes, on='userid', how='left')
        valid_data = merged_data.dropna(subset=['like_id'])

        if len(valid_data) == 0:
            print("Warning: No valid likes data found")
            return result_df

        # Transform likes using pretrained vectorizer
        X_new = predictor.vectorizer.transform(valid_data['like_id'])

        # Make predictions for regression traits (age, personality)
        for trait in predictor.regression_traits:
            if trait in predictor.models:
                predictions = predictor.models[trait].predict(X_new)
                
                if trait == 'age':
                    predictions = np.clip(np.round(predictions).astype(int), 13, 90)
                    age_ranges = [f"{age-5}-{age+5}" for age in predictions]
                    result_df.loc[valid_data.index, 'age_range'] = age_ranges
                else:
                    predictions = np.clip(np.round(predictions, 2), 1, 5)
                
                result_df.loc[valid_data.index, trait] = predictions
                print(f"\n{trait.upper()} prediction statistics:")
                print(pd.Series(predictions).describe())

        # Make predictions for classification traits (gender)
        for trait in predictor.classification_traits:
            if trait in predictor.models:
                predictions = predictor.models[trait].predict(X_new)
                probabilities = predictor.models[trait].predict_proba(X_new)
                
                predictions = predictor.label_encoder.inverse_transform(predictions)
                
                result_df.loc[valid_data.index, trait] = predictions
                result_df.loc[valid_data.index, f'{trait}_confidence'] = np.max(probabilities, axis=1)

        print("\nPrediction distribution:")
        for trait in predictor.regression_traits + predictor.classification_traits:
            if trait in result_df.columns:
                print(f"\n{trait.upper()} predictions:")
                print(result_df[trait].describe())

    except Exception as e:
        print(f"Error in traits prediction: {str(e)}")
        return result_df

    return result_df