import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

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
        self.age_model = RandomForestRegressor(
            n_estimators=50,  # Number of trees
            max_depth=None,    # Let trees grow fully
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,        # Use all available cores
            random_state=42
        )
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
        """
        Preprocess the likes and profile data for training.
        
        Args:
            relation_path: Path to the relation CSV file containing user-like pairs
            profile_path: Path to the profile CSV file containing user traits
            
        Returns:
            tuple containing:
            - features: Dict with 'like_id' and 'likes_count' features
            - labels: Dict with trait values for each user
            - userids: Series of user IDs
        """
        # Load the raw data files
        relation = pd.read_csv(relation_path)
        profile = pd.read_csv(profile_path)
        
        # Calculate the number of likes per user as an additional feature
        likes_count = relation.groupby('userid').size().reset_index(name='likes_count')
        
        # Convert like IDs to space-separated strings for TF-IDF vectorization
        likes = (relation.groupby('userid')['like_id']
                .agg(lambda x: ' '.join(map(str, x)))
                .reset_index())
        
        # Merge all features and profile data
        all_traits = self.regression_traits + self.classification_traits
        data = (likes
               .merge(likes_count, on='userid')
               .merge(profile[['userid'] + all_traits], on='userid'))
        
        # Remove outliers from regression traits
        for trait in self.regression_traits:
            if trait in data.columns:
                data = self.remove_outliers(data, trait)
                print(f"After removing outliers for {trait}: {len(data)} samples")
        
        # Prepare features dictionary
        features = {
            'like_id': data['like_id'],
            'likes_count': data['likes_count']
        }
        
        # Prepare labels dictionary for regression traits
        labels = {trait: data[trait] for trait in self.regression_traits}
        
        # Process classification traits (gender) separately
        for trait in self.classification_traits:
            if trait in data.columns:
                # Fill missing values and encode categorical labels
                labels[trait] = self.label_encoder.fit_transform(data[trait].fillna('unknown'))
        
        return features, labels, data['userid']
    
    def get_age_group(self, age):
        """Convert age to standardized age group format."""
        lower = age - (age % 5)  # Round down to nearest 5
        return f"{lower}-{lower+5}"
    
    def calculate_age_accuracy(self, y_true, y_pred):
        """Calculate accuracy of age group predictions."""
        true_groups = np.array([self.get_age_group(age) for age in y_true])
        pred_groups = np.array([self.get_age_group(age) for age in y_pred])
        return (true_groups == pred_groups).mean()


    def train_and_save_model(self, relation_path: str, profile_path: str, model_save_path: str):
        """Enhanced training with proper train-test split and evaluation metrics."""
        features, labels, userids = self.preprocess_likes(relation_path, profile_path)
        
        # Create feature matrix
        X_text = self.vectorizer.fit_transform(features['like_id'])
        
        # Create train-test split first (85-15)
        indices = np.arange(X_text.shape[0])
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=0.15,  # 15% for testing
            random_state=42
        )
        
        # Split features into train and test
        X_train = X_text[train_idx]
        X_test = X_text[test_idx]
        
        print(f"\nData split sizes:")
        print(f"Training samples: {len(train_idx)} ({len(train_idx)/len(indices):.1%})")
        print(f"Testing samples: {len(test_idx)} ({len(test_idx)/len(indices):.1%})")
        
        print("\nTraining and evaluating models...")
        
        # Train regression models with proper evaluation
        for trait in self.regression_traits:
            if trait in labels:
                y = np.array(labels[trait])
                y_train = y[train_idx]
                y_test = y[test_idx]
                
                if trait == 'age':
                    # Special handling for age using Random Forest
                    predictions = []
                    true_values = []
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    
                    # Cross-validation on training data only
                    for fold_train_idx, val_idx in kf.split(X_train):
                        # Train Random Forest on fold
                        self.age_model.fit(
                            X_train[fold_train_idx], 
                            y_train[fold_train_idx]
                        )
                        # Predict and store results
                        fold_preds = self.age_model.predict(X_train[val_idx])
                        predictions.extend(fold_preds)
                        true_values.extend(y_train[val_idx])
                    
                    # Calculate cross-validation accuracy
                    cv_accuracy = self.calculate_age_accuracy(true_values, predictions)
                    
                    # Train final model on all training data
                    self.age_model.fit(X_train, y_train)
                    
                    # Evaluate on test set
                    test_predictions = self.age_model.predict(X_test)
                    test_accuracy = self.calculate_age_accuracy(y_test, test_predictions)
                    
                    print(f"\n{trait.upper()} Prediction:")
                    print(f"Cross-validation Age Group Accuracy: {cv_accuracy:.2f}")
                    print(f"Test Set Age Group Accuracy: {test_accuracy:.2f}")
                    
                    # Save the model trained on all training data
                    self.models[trait] = self.age_model
                    
                else:
                    # Handle personality traits
                    best_metric = float('inf')
                    best_model = None
                    best_model_name = None
                    
                    for model_name, model in self.model_candidates.items():
                        pipeline = Pipeline([
                            ('scaler', StandardScaler(with_mean=False)),
                            ('model', model)
                        ])
                        
                        # Cross-validation on training data only
                        cv_scores = cross_val_score(
                            pipeline, X_train, y_train,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            scoring='neg_root_mean_squared_error'
                        )
                        metric = -cv_scores.mean()
                        
                        if metric < best_metric:
                            best_metric = metric
                            best_model = model
                            best_model_name = model_name
                    
                    # Train final pipeline on all training data
                    pipeline = Pipeline([
                        ('scaler', StandardScaler(with_mean=False)),
                        ('model', best_model)
                    ])
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate on test set
                    test_predictions = pipeline.predict(X_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
                    
                    print(f"\n{trait.upper()} Prediction:")
                    print(f"Best model: {best_model_name}")
                    print(f"Cross-validation RMSE: {-best_metric:.2f}")
                    print(f"Test Set RMSE: {test_rmse:.2f}")
                    
                    self.models[trait] = pipeline

        # Handle gender classification
        for trait in self.classification_traits:
            if trait in labels:
                y = np.array(labels[trait])
                y_train = y[train_idx]
                y_test = y[test_idx]
                
                model = MultinomialNB()
                
                # Cross-validation on training data
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='accuracy'
                )
                
                # Train final model on all training data
                model.fit(X_train, y_train)
                
                # Evaluate on test set
                test_accuracy = model.score(X_test, y_test)
                
                print(f"\n{trait.upper()} Prediction:")
                print(f"Cross-validation Accuracy: {cv_scores.mean():.2f}")
                print(f"Test Set Accuracy: {test_accuracy:.2f}")
                
                self.models[trait] = model
        
        # Save the models
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