import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

from evaluation import TraitEvaluator

class UserTraitsPredictor:
    def __init__(self):
        # Initialize our TF-IDF vectorizer for converting user likes to features
        # Initialize our TF-IDF vectorizer for converting user likes to features
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95
        )
        
        # Set up our age classification model
        self.age_classifier = OneVsRestClassifier(
            LogisticRegression(
                multi_class='ovr',
                max_iter=1000,
                class_weight='balanced'
            )
        )
        
        # Set up our gender classification model
        self.gender_classifier = OneVsRestClassifier(
            LogisticRegression(
                multi_class='ovr',
                max_iter=1000,
                class_weight='balanced'
            )
        )
        self.test_size = 0.2  # 80% train, 20% test
        self.random_state = 42  # For reproducibility
        # Create separate models for each personality trait
        self.personality_models = {
            trait: Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('regressor', Ridge(alpha=1.0))
            ]) for trait in ['ope', 'neu', 'ext', 'agr', 'con']
        }
        
        # Initialize our encoders
        self.age_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        
        # Create our evaluator instance
        self.evaluator = TraitEvaluator()

    def preprocess_data(self, relation_path: str, profile_path: str) -> tuple:
            """Process the raw data files into features and labels with train/test split."""
            # Load data files
            relation = pd.read_csv(relation_path)
            profile = pd.read_csv(profile_path)
            
            # Convert user likes into text format for TF-IDF
            user_likes = (relation.groupby('userid')['like_id']
                        .agg(lambda x: ' '.join(map(str, x)))
                        .reset_index())
            
            # Merge with profile data to ensure alignment
            merged_data = user_likes.merge(profile, on='userid')
            
            # Split users first to prevent data leakage
            train_indices, test_indices = train_test_split(
                np.arange(len(merged_data)),
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            # Split the data
            train_data = merged_data.iloc[train_indices]
            test_data = merged_data.iloc[test_indices]
            
            # Process training data
            X_train = self.vectorizer.fit_transform(train_data['like_id'])
            y_train_age_ranges = train_data['age'].apply(self._convert_age_to_range)
            y_train_age_encoded = self.age_encoder.fit_transform(y_train_age_ranges)
            y_train_gender = train_data['gender']
            y_train_gender_encoded = self.gender_encoder.fit_transform(y_train_gender)
            y_train_personality = {
                trait: train_data[trait].values
                for trait in self.personality_models.keys()
            }
            
            # Process test data - using fitted vectorizer and encoders
            X_test = self.vectorizer.transform(test_data['like_id'])
            y_test_age_ranges = test_data['age'].apply(self._convert_age_to_range)
            y_test_age_encoded = self.age_encoder.transform(y_test_age_ranges)
            y_test_gender = test_data['gender']
            y_test_gender_encoded = self.gender_encoder.transform(y_test_gender)
            y_test_personality = {
                trait: test_data[trait].values
                for trait in self.personality_models.keys()
            }
            
            train_data = {
                'X': X_train,
                'age_ranges': y_train_age_ranges.values,
                'age_encoded': y_train_age_encoded,
                'gender': y_train_gender.values,
                'gender_encoded': y_train_gender_encoded,
                'personality': y_train_personality,
                'userids': train_data['userid'].values
            }
            
            test_data = {
                'X': X_test,
                'age_ranges': y_test_age_ranges.values,
                'age_encoded': y_test_age_encoded,
                'gender': y_test_gender.values,
                'gender_encoded': y_test_gender_encoded,
                'personality': y_test_personality,
                'userids': test_data['userid'].values
            }
            
            return train_data, test_data
        
    def _convert_age_to_range(self, age: int) -> str:
        """Convert a numeric age to its corresponding range category."""
        if age <= 24:
            return 'xx-24'
        elif age <= 34:
            return '25-34'
        elif age <= 49:
            return '35-49'
        else:
            return '50-xx'

    def train_and_save(self, relation_path: str, profile_path: str, save_dir: str) -> Dict:
        """Train models, evaluate performance, and save to disk."""
        print("Starting model training process...")
        
        # Get train and test data
        train_data, test_data = self.preprocess_data(relation_path, profile_path)
        
        # Train models
        print("\nTraining models...")
        self.age_classifier.fit(train_data['X'], train_data['age_encoded'])
        self.gender_classifier.fit(train_data['X'], train_data['gender_encoded'])
        
        for trait, labels in train_data['personality'].items():
            print(f"Training {trait.upper()} model...")
            self.personality_models[trait].fit(train_data['X'], labels)
        
        # Evaluate on training data
        print("\nTraining Set Performance:")
        train_results = self.evaluator.evaluate_all_traits(
            true_age=train_data['age_ranges'],
            predicted_age_ranges=self.age_encoder.inverse_transform(
                self.age_classifier.predict(train_data['X'])
            ),
            true_gender=train_data['gender'],
            predicted_gender=self.gender_encoder.inverse_transform(
                self.gender_classifier.predict(train_data['X'])
            ),
            true_traits=train_data['personality'],
            predicted_traits={
                trait: model.predict(train_data['X'])
                for trait, model in self.personality_models.items()
            }
        )
        self.evaluator.print_evaluation_results(train_results)
        
        # Evaluate on test data
        print("\nTest Set Performance:")
        test_results = self.evaluator.evaluate_all_traits(
            true_age=test_data['age_ranges'],
            predicted_age_ranges=self.age_encoder.inverse_transform(
                self.age_classifier.predict(test_data['X'])
            ),
            true_gender=test_data['gender'],
            predicted_gender=self.gender_encoder.inverse_transform(
                self.gender_classifier.predict(test_data['X'])
            ),
            true_traits=test_data['personality'],
            predicted_traits={
                trait: model.predict(test_data['X'])
                for trait, model in self.personality_models.items()
            }
        )
        self.evaluator.print_evaluation_results(test_results)
        
        # Combine results for metadata
        evaluation_summary = {
            'train_results': train_results,
            'test_results': test_results
        }
        
        print("\nSaving model components...")
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save each component separately
        joblib.dump(self.age_classifier, save_path / 'age_classifier.joblib')
        joblib.dump(self.gender_classifier, save_path / 'gender_classifier.joblib')
        with open(save_path / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(save_path / 'age_encoder.pkl', 'wb') as f:
            pickle.dump(self.age_encoder, f)
        with open(save_path / 'gender_encoder.pkl', 'wb') as f:
            pickle.dump(self.gender_encoder, f)
        
        for trait, model in self.personality_models.items():
            joblib.dump(model, save_path / f'{trait}_model.joblib')
        
        # Save metadata with both training and test results
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'evaluation_results': evaluation_summary,
            'vectorizer_config': self.vectorizer.get_params(),
            'personality_traits': list(self.personality_models.keys()),
            'version': '1.0'
        }
        
        with open(save_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nModel saved successfully to {save_path}")
        return evaluation_summary

if __name__ == "__main__":
    # Define paths according to project structure
    relation_path = "data/training/relation/relation.csv"
    profile_path = "data/training/profile/profile.csv"
    save_dir = "like/user_traits_prediction_models"
    
    # Create and train our model
    predictor = UserTraitsPredictor()
    evaluation_results = predictor.train_and_save(
        relation_path=relation_path,
        profile_path=profile_path,
        save_dir=save_dir
    )
