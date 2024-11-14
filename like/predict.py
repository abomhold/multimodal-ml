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
        
        # Count number of likes per user
        likes['like_count'] = likes['like_id'].str.count(' ') + 1

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
                    # Round age predictions
                    predictions = np.round(predictions).astype(int)
                    
                    # Set minimum and maximum reasonable age limits
                    MIN_AGE = 13
                    MAX_AGE = 90
                    
                    # Clip age predictions to reasonable range
                    predictions = np.clip(predictions, MIN_AGE, MAX_AGE)
                    
                    # Add age ranges
                    age_ranges = [f"{age-5}-{age+5}" for age in predictions]
                    result_df.loc[valid_data.index, 'age_range'] = age_ranges
                    
                else:
                    # For personality traits (ope, con, ext, agr, neu)
                    # Clip between 1 and 5, as these are the valid ranges for Big Five traits
                    predictions = np.clip(np.round(predictions, 2), 1, 5)
                    
                    # Print debugging information for personality predictions
                    print(f"\n{trait.upper()} prediction statistics:")
                    print(pd.Series(predictions).describe())
                
                result_df.loc[valid_data.index, trait] = predictions

        # Make predictions for classification traits (gender)
        for trait in predictor.classification_traits:
            if trait in predictor.models:
                predictions = predictor.models[trait].predict(X_new)
                probabilities = predictor.models[trait].predict_proba(X_new)
                
                # Transform predictions back to original labels
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