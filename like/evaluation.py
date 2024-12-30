import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import Dict, Tuple

class TraitEvaluator:
    def __init__(self):
        # Define just the personality traits since age and gender are handled separately
        self.personality_traits = ['ope', 'neu', 'ext', 'agr', 'con']
    
    def evaluate_all_traits(self, 
                          true_age: np.ndarray,
                          predicted_age_ranges: np.ndarray,
                          true_gender: np.ndarray,
                          predicted_gender: np.ndarray,
                          true_traits: Dict[str, np.ndarray],
                          predicted_traits: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate predictions using accuracy for classifications (age and gender) 
        and RMSE for personality traits.
        
        Args:
            true_age: Array of actual age ranges
            predicted_age_ranges: Array of predicted age ranges
            true_gender: Array of actual gender labels
            predicted_gender: Array of predicted gender labels
            true_traits: Dictionary of true personality scores
            predicted_traits: Dictionary of predicted personality scores
        
        Returns:
            Dictionary containing accuracies for classifications and RMSE for traits
        """
        # Calculate classification accuracies
        age_accuracy = accuracy_score(true_age, predicted_age_ranges)
        gender_accuracy = accuracy_score(true_gender, predicted_gender)
        
        # Initialize results with classification accuracies
        results = {
            'age_accuracy': age_accuracy,
            'gender_accuracy': gender_accuracy
        }
        
        # Calculate RMSE for personality traits
        for trait in self.personality_traits:
            trait_rmse = np.sqrt(mean_squared_error(
                true_traits[trait],
                predicted_traits[trait]
            ))
            results[trait] = trait_rmse
        
        return results

    def print_evaluation_results(self, results: Dict[str, float]):
        """
        Print evaluation results with accuracies as decimals to 2 places. 
        
        Args:
            results: Dictionary containing accuracies and RMSE values
        """
        print("\nEvaluation Results")
        print("-" * 40)
        
        # Print classification accuracies as decimals to 2 places
        print("Classification Tasks (Accuracy):")
        print(f"Age Prediction:    {results['age_accuracy']:.2f}")
        print(f"Gender Prediction: {results['gender_accuracy']:.2f}")
        
        print("\nPersonality Traits (RMSE):")
        for trait in self.personality_traits:
            print(f"{trait.upper():>5} RMSE: {results[trait]:.4f}")
            
# Example usage in the UserTraitsPredictor class:
def evaluate(self, X_test, y_test_age, y_test_personality):
    """
    Evaluate predictions using RMSE for all traits.
    
    Args:
        X_test: Test features
        y_test_age: True ages (integers)
        y_test_personality: Dictionary of true personality scores
    """
    # Make predictions
    age_range_predictions = self.age_classifier.predict(X_test)
    personality_predictions = {
        trait: model.predict(X_test)
        for trait, model in self.personality_models.items()
    }
    
    # Evaluate predictions
    evaluator = TraitEvaluator()
    results = evaluator.evaluate_all_traits(
        true_age=y_test_age,
        predicted_age_ranges=age_range_predictions,
        true_traits=y_test_personality,
        predicted_traits=personality_predictions
    )
    
    # Print results
    evaluator.print_evaluation_results(results)
    
    return results