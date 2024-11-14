import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (mean_squared_error, r2_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

class UserTraitsPredictor:
    def __init__(self):
        self.vectorizer = CountVectorizer(min_df=5)
        self.models = {}
        self.label_encoder = LabelEncoder()
        
       # Traits
        self.regression_traits = ['age', 'ope', 'con', 
                                'ext', 'agr', 'neu']
        self.classification_traits = ['gender']
        
    def preprocess_likes(self, relation_path: str, profile_path: str) -> tuple:
        """
        Preprocess likes data for prediction of all traits.
        Returns features (likes) and labels dictionary.
        """
        # Load data
        relation = pd.read_csv(relation_path)
        profile = pd.read_csv(profile_path)
        
        # Group likes by user
        likes = (relation.groupby('userid')['like_id']
                .agg(lambda x: ' '.join(map(str, x)))
                .reset_index())
        
        # Get all traits
        all_traits = self.regression_traits + self.classification_traits
        data = pd.merge(likes, profile[['userid'] + all_traits], on='userid')
        
        # Prepare labels dictionary
        labels = {}
        
        # Process continuous variables (age, personality traits)
        for trait in self.regression_traits:
            if trait in data.columns:
                labels[trait] = data[trait].fillna(data[trait].mean())
        
        # Process (gender)
        for trait in self.classification_traits:
            if trait in data.columns:
                labels[trait] = self.label_encoder.fit_transform(data[trait].fillna('unknown'))
        
        return data['like_id'], labels
    
    def train_and_save_model(self, relation_path: str, profile_path: str, model_save_path: str):
        """Train models for all traits and save them."""
        # Preprocess data
        likes, labels = self.preprocess_likes(relation_path, profile_path)
        
        # Create feature matrix
        X = self.vectorizer.fit_transform(likes)
        
        # Train regression models (age and personality traits)
        print("\nTraining regression models...")
        for trait in self.regression_traits:
            if trait in labels:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels[trait], test_size=0.2, random_state=42
                )
                
                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)
                self.models[trait] = model
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                print(f"\n{trait.upper()} Prediction:")
                print(f"Training RMSE: {train_rmse:.2f}")
                print(f"Testing RMSE: {test_rmse:.2f}")
                print(f"Training R²: {train_r2:.3f}")
                print(f"Testing R²: {test_r2:.3f}")
        

        print("\nTraining classification models...")
        for trait in self.classification_traits:
            if trait in labels:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels[trait], test_size=0.2, random_state=42, stratify=labels[trait]
                )
                
                # Train model
                model = MultinomialNB()
                model.fit(X_train, y_train)
                self.models[trait] = model
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                print(f"\n{trait.upper()} Prediction:")
                print("Training Classification Report:")
                print(classification_report(y_train, train_pred))
                print("Testing Classification Report:")
                print(classification_report(y_test, test_pred))
        
        # Save models and preprocessing objects
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'models': self.models,
                'label_encoder': self.label_encoder,
                'regression_traits': self.regression_traits,
                'classification_traits': self.classification_traits
            }, f)
        print(f"\nModels saved to {model_save_path}")
    
    def predict_traits(self, new_relation_path: str, model_path: str) -> pd.DataFrame:
        """Predict all traits for new data using the pretrained models."""
        # Load the saved models and preprocessing objects
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.vectorizer = saved_data['vectorizer']
            self.models = saved_data['models']
            self.label_encoder = saved_data['label_encoder']
            self.regression_traits = saved_data['regression_traits']
            self.classification_traits = saved_data['classification_traits']
        
        # Load and preprocess new data
        relation = pd.read_csv(new_relation_path)
        likes = (relation.groupby('userid')['like_id']
                .agg(lambda x: ' '.join(map(str, x)))
                .reset_index())
        
        # Transform likes using the pretrained vectorizer
        X_new = self.vectorizer.transform(likes['like_id'])
        
        # Initialize results dictionary
        results = {'userid': likes['userid']}
        
        # Make predictions for regression traits
        for trait in self.regression_traits:
            if trait in self.models:
                predictions = self.models[trait].predict(X_new)
                results[f'predicted_{trait}'] = np.round(predictions, 2)
                
                # Add age range for age predictions
                if trait == 'age':
                    results['predicted_age_range'] = [
                        f"{max(0, age-5):.0f}-{age+5:.0f}" 
                        for age in predictions
                    ]
        
        # Make predictions for classification traits
        for trait in self.classification_traits:
            if trait in self.models:
                predictions = self.models[trait].predict(X_new)
                probabilities = self.models[trait].predict_proba(X_new)
                
                # Transform predictions back to original labels
                predictions = self.label_encoder.inverse_transform(predictions)
                
                results[f'predicted_{trait}'] = predictions
                results[f'{trait}_probability'] = np.max(probabilities, axis=1)
        
        return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    predictor = UserTraitsPredictor()
    
    # Training phase
    predictor.train_and_save_model(
        relation_path="data/training/relation/relation.csv",
        profile_path="data/training/profile/profile.csv",
        model_save_path="like/user_traits_prediction_models.pkl"
    )
    