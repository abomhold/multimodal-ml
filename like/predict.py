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
        print("Loaded models for traits:", predictor.models.keys())

        # Load and process relation data
        relation = pd.read_csv(relation_path)
        print(f"Loaded {len(relation)} relations")
        
        likes = (relation.groupby('userid')['like_id']
                .agg(lambda x: ' '.join(map(str, x)))
                .reset_index())
        print(f"Processed {len(likes)} unique users' likes")
        
        # Merge with input data
        merged_data = pd.merge(data, likes, on='userid', how='left')
        valid_data = merged_data.dropna(subset=['like_id'])
        print(f"Found valid likes for {len(valid_data)} out of {len(data)} users")

        if len(valid_data) == 0:
            print("Warning: No valid likes data found")
            return result_df

        # Transform likes using pretrained vectorizer
        X_new = predictor.vectorizer.transform(valid_data['like_id'])
        print(f"Created feature matrix with shape: {X_new.shape}")

        # Make predictions for regression traits
        for trait in predictor.regression_traits:
            if trait in predictor.models:
                print(f"\nPredicting {trait}...")
                model = predictor.models[trait]
                print(f"Model type: {type(model)}")
                
                predictions = model.predict(X_new)
                print(f"Raw predictions stats for {trait}:")
                print(pd.Series(predictions).describe())
                
                if trait == 'age':
                    # Age remains as integers
                    predictions = np.clip(np.round(predictions).astype(int), 13, 90)
                    age_ranges = [f"{age-5}-{age+5}" for age in predictions]
                    result_df.loc[valid_data.index, 'age_range'] = age_ranges
                else:
                    # Personality traits are clipped to [1, 5] range and rounded to 2 decimal places
                    predictions = np.clip(predictions, 1, 5)
                    predictions = np.round(predictions, 2)
                
                result_df.loc[valid_data.index, trait] = predictions
                print(f"Final predictions stats for {trait}:")
                print(pd.Series(predictions).describe())

        # Make predictions for classification traits
        for trait in predictor.classification_traits:
            if trait in predictor.models:
                print(f"\nPredicting {trait}...")
                predictions = predictor.models[trait].predict(X_new)
                probabilities = predictor.models[trait].predict_proba(X_new)
                
                predictions = predictor.label_encoder.inverse_transform(predictions)
                
                result_df.loc[valid_data.index, trait] = predictions
                # Round confidence scores to 2 decimal places as well
                result_df.loc[valid_data.index, f'{trait}_confidence'] = np.round(
                    np.max(probabilities, axis=1), 2
                )
                

    except Exception as e:
        import traceback
        print(f"Error in traits prediction: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return result_df

    return result_df