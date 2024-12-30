import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class UserTraitsPredictor:
    def __init__(self, model_dir: str = "like/user_traits_prediction_models"):
        """
        Initialize the predictor with saved models from training.
        The model_dir should contain all saved components from our training process.
        """
        self.model_dir = Path(model_dir)
        self.load_models()
    
    def load_models(self):
        """
        Load all necessary model components from the saved directory.
        This includes the vectorizer, encoders, and all trained models.
        """
        try:
            # Load text processing components
            with open(self.model_dir / 'vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load classification components
            with open(self.model_dir / 'age_encoder.pkl', 'rb') as f:
                self.age_encoder = pickle.load(f)
            with open(self.model_dir / 'gender_encoder.pkl', 'rb') as f:
                self.gender_encoder = pickle.load(f)
            
            # Load trained models
            self.age_classifier = joblib.load(self.model_dir / 'age_classifier.joblib')
            self.gender_classifier = joblib.load(self.model_dir / 'gender_classifier.joblib')
            
            # Load personality models
            self.personality_models = {}
            for trait in ['ope', 'neu', 'ext', 'agr', 'con']:
                self.personality_models[trait] = joblib.load(
                    self.model_dir / f'{trait}_model.joblib'
                )
                
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")

    def predict_all(self, likes_path: str, output_df: pd.DataFrame) -> pd.DataFrame:
        print("\nProcessing likes data...")
        result_df = output_df.copy()
        
        try:
            # Read and process likes data
            likes_data = pd.read_csv(likes_path)
            print(f"Loaded {len(likes_data)} like entries")
            
            # Convert likes to the format expected by our models
            user_likes = (likes_data.groupby('userid')['like_id']
                        .agg(lambda x: ' '.join(map(str, x)))
                        .reset_index())
            print(f"Processed likes for {len(user_likes)} unique users")
            
            # Create feature matrix using trained vectorizer
            X = self.vectorizer.transform(user_likes['like_id'])
            print(f"Created feature matrix with shape: {X.shape}")
            
            # Create predictions DataFrame
            predictions_df = pd.DataFrame({'userid': user_likes['userid']})
            
            # Age predictions
            print("Making age predictions...")
            age_encoded = self.age_classifier.predict(X)
            predictions_df['age_range'] = self.age_encoder.inverse_transform(age_encoded)
            
            # Gender predictions
            print("Making gender predictions...")
            gender_encoded = self.gender_classifier.predict(X)
            predictions_df['gender'] = self.gender_encoder.inverse_transform(gender_encoded)
            gender_probs = self.gender_classifier.predict_proba(X)
            predictions_df['gender_confidence'] = np.round(np.max(gender_probs, axis=1), 2)
            
            # Personality predictions
            print("Making personality predictions...")
            for trait, model in self.personality_models.items():
                predictions = model.predict(X)
                predictions = np.clip(predictions, 1, 5)
                predictions = np.round(predictions, 2)
                predictions_df[trait] = predictions
            
            # Drop any existing prediction columns from result_df before merging
            columns_to_drop = ['age_range', 'gender', 'gender_confidence', 'ope', 'con', 'ext', 'agr', 'neu']
            result_df = result_df.drop(columns=[col for col in columns_to_drop if col in result_df.columns])
            
            # Merge predictions
            result_df = pd.merge(
                result_df,
                predictions_df,
                on='userid',
                how='left'
            )
            
            print(f"Made predictions for {len(predictions_df)} users")
            print("Final columns:", result_df.columns.tolist())
            
        except Exception as e:
            import traceback
            print(f"Error in prediction process: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
        
        return result_df