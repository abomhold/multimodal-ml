import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

class UserTraitsPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            min_df=10,
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        
        self.model_candidates = {
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),    
            'elastic': ElasticNet(random_state=42)
        }
        
        self.regression_traits = ['age', 'ope', 'neu', 'ext', 'agr', 'con']
        self.classification_traits = ['gender']
    
    def remove_outliers(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers from a specific column using IQR method."""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    def preprocess_likes(self, relation_path: str, profile_path: str) -> tuple:
        """Enhanced preprocessing with consistent sample sizes after outlier removal."""
        relation = pd.read_csv(relation_path)
        profile = pd.read_csv(profile_path)
        
        # Add feature engineering
        likes_count = relation.groupby('userid').size().reset_index(name='likes_count')
        
        likes = (relation.groupby('userid')['like_id']
                .agg(lambda x: ' '.join(map(str, x)))
                .reset_index())
        
        # Merge all features
        all_traits = self.regression_traits + self.classification_traits
        data = (likes
               .merge(likes_count, on='userid')
               .merge(profile[['userid'] + all_traits], on='userid'))
        
        # Remove outliers from regression traits
        for trait in self.regression_traits:
            if trait in data.columns:
                data = self.remove_outliers(data, trait)
        
        # Prepare features and labels
        features = {
            'like_id': data['like_id'],
            'likes_count': data['likes_count']
        }
        
        labels = {trait: data[trait] for trait in self.regression_traits}
        
        # Process gender separately
        for trait in self.classification_traits:
            if trait in data.columns:
                labels[trait] = self.label_encoder.fit_transform(data[trait].fillna('unknown'))
        
        return features, labels, data['userid']
    
    def train_and_save_model(self, relation_path: str, profile_path: str, model_save_path: str):
        """Enhanced training with consistent sample sizes."""
        features, labels, userids = self.preprocess_likes(relation_path, profile_path)
        
        # Create feature matrix
        X_text = self.vectorizer.fit_transform(features['like_id'])
        
        print("\nTraining regression models...")
        for trait in self.regression_traits:
            if trait in labels:
                best_rmse = float('inf')
                best_model = None
                best_model_name = None
                
                y = labels[trait]
                
                for model_name, model in self.model_candidates.items():
                    pipeline = Pipeline([
                        ('scaler', StandardScaler(with_mean=False)),
                        ('model', model)
                    ])
                    
                    cv_scores = cross_val_score(
                        pipeline, X_text, y,
                        cv=KFold(n_splits=5, shuffle=True, random_state=42),
                        scoring='neg_root_mean_squared_error'
                    )
                    
                    mean_rmse = -cv_scores.mean()
                    std_rmse = cv_scores.std()
                    
                    if mean_rmse < best_rmse:
                        best_rmse = mean_rmse
                        best_model = model
                        best_model_name = model_name
                
                print(f"\n{trait.upper()} Prediction:")
                print(f"Best model: {best_model_name}")
                print(f"Cross-validation RMSE: {best_rmse:.2f} (+/- {std_rmse:.2f})")
                
                pipeline = Pipeline([
                    ('scaler', StandardScaler(with_mean=False)),
                    ('model', best_model)
                ])
                pipeline.fit(X_text, y)
                self.models[trait] = pipeline

        print("\nTraining classification models...")
        for trait in self.classification_traits:
            if trait in labels:
                model = MultinomialNB()
                model.fit(X_text, labels[trait])
                self.models[trait] = model
                
                # Calculate and print cross-validation scores
                cv_scores = cross_val_score(
                    model, X_text, labels[trait],
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='accuracy'
                )
                print(f"\n{trait.upper()} Prediction:")
                print(f"Cross-validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
        
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'models': self.models,
                'label_encoder': self.label_encoder,
                'regression_traits': self.regression_traits,
                'classification_traits': self.classification_traits
            }, f)
        print(f"\nModels saved to {model_save_path}")
        
if __name__ == "__main__":
    predictor = UserTraitsPredictor()
    predictor.train_and_save_model(
        relation_path="data/training/relation/relation.csv",
        profile_path="data/training/profile/profile.csv",
        model_save_path="user_traits_prediction_models.pkl"
    )