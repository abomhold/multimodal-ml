from sklearn.model_selection import KFold
import torch
import pandas as pd
import image.image_testrun as image_testrun
import config
import numpy as np
import like.predict
import preprocessing as pre
import postprocessing as post
from pathlib import Path
import argparse
import split
from split import analyze_predictions, convert_age_to_range, print_analysis_results, split_personality_data
import text.main as text
from text import personality_prediction
from like.predict import UserTraitsPredictor
from typing import Dict, List
import copy

device =  "cpu"
model_name = 'resnet50'


def parse_args():
    parser = argparse.ArgumentParser(description='Process input and output paths')
    parser.add_argument('-i', '--input', dest='input_path', default='input',
                        help='Input path (default: input)')
    parser.add_argument('-o', '--output', dest='output_path', default='output',
                        help='Output path (default: output)')

    args = parser.parse_args()
    return args.input_path, args.output_path


def perform_cross_validation(data: pd.DataFrame, n_splits: int = 10) -> Dict:
    """
    Perform k-fold cross validation for evaluation purposes only.
    
    Args:
        data: DataFrame containing the full dataset
        n_splits: Number of folds for cross validation
    
    Returns:
        Dictionary containing averaged results and individual fold results
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (_, test_idx) in enumerate(kf.split(data)):
        print(f"\nProcessing fold {fold + 1}/{n_splits}")
        # Get test data for this fold
        test_data = data.iloc[test_idx].copy()
        
        # Run predictions for this fold
        fold_results_dict = run_fold_predictions(test_data, data)
        fold_results.append(fold_results_dict)
    
    # Aggregate results across all folds
    aggregated_results = aggregate_fold_results(fold_results)
    
    results = {
    'averaged_results': aggregated_results,
    'fold_results': fold_results
}
    return results


def run_fold_predictions(test_data: pd.DataFrame, dataframe:pd.DataFrame) -> Dict:
    """
    Run predictions for a single fold.
    """
    # # Run predictions
    image_df = image_testrun.test(config.IMAGE_DIR, test_data.copy(), model_name, device)
    personality_df = text.main(Path(config.TEXT_DIR), test_data.copy())
    like_predictor = UserTraitsPredictor()
    like_df = like_predictor.predict_all(likes_path=config.LIKE_PATH, output_df=test_data.copy())
    
    # Combine predictions
    combined_df = pd.merge(
        test_data.copy(),
        image_df.loc[:, ['userid', 'gender']],
        on='userid'
    )

    combined_df = pd.merge(
        combined_df,  # Use the result from the first merge
        personality_df.loc[:, ['userid', 'ope', 'con', 'ext', 'agr', 'neu']],
        on='userid'
    )
    
    combined_df = pd.merge(
        combined_df,    
        like_df.loc[:,['userid','age_range']],
        on='userid',
    )
    combined_df['age_true_range'] = combined_df['age'].apply(convert_age_to_range)
    
    # Analyze results for this fold
    results = analyze_predictions(combined_df, dataframe)

    return results
    


            

def aggregate_fold_results(fold_results: List[Dict], labels: List[str] = None) -> Dict:
    """
    Aggregate results across all folds, including confusion matrices and metrics.
    
    Args:
        fold_results: List of dictionaries containing results from each fold
        labels: List of class labels for formatting confusion matrices
    
    Returns:
        Dictionary containing averaged results for each trait
    """
    aggregated = {}
    all_traits = set()

    # Identify all traits across the folds
    for fold in fold_results:
        for key in fold.keys():
            if isinstance(fold[key], dict):
                all_traits.add(key)

    # Aggregate metrics and confusion matrices for each trait
    for trait in all_traits:
        trait_matrices = []
        trait_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'one_off_accuracy': [],
            'average_range_distance': [],
            'valid_predictions': [],
            'total_samples': []
        }

        # Collect metrics and matrices for the current trait
        for fold in fold_results:
            if trait in fold:
                fold_trait_data = fold[trait]
                metrics_data = fold_trait_data.get('metrics', fold_trait_data)
                for metric in trait_metrics:
                    if metric in metrics_data:
                        trait_metrics[metric].append(float(metrics_data[metric]))
                confusion_matrix = fold_trait_data.get('confusion_matrix')
                if confusion_matrix is not None:
                    trait_matrices.append(np.array(confusion_matrix))

        # Calculate averaged metrics for the current trait
        metrics_dict = {metric: float(np.mean(values)) if values else 0.0 for metric, values in trait_metrics.items()}

        # Average confusion matrices if present
        mean_matrix = average_confusion_matrices(trait_matrices) if trait_matrices else None

        # Convert to DataFrame if labels are provided
        if mean_matrix is not None and labels:
            mean_matrix = format_confusion_matrix(mean_matrix, labels)

        # Store aggregated results for the current trait
        aggregated[trait] = {
            'confusion_matrix': mean_matrix,
            'metrics': metrics_dict,
            'report': metrics_dict
        }

    # Return the aggregated results
    return aggregated

def get_unique_incorrect_users(cv_results):
    """
    Extract unique user IDs from incorrect predictions across all folds.
    
    Parameters:
    -----------
    cv_results : dict
        Dictionary containing fold_results from cross validation
    
    Returns:
    --------
    dict
        Dictionary with traits as keys and sets of unique incorrect user IDs
    """
    unique_incorrect_users = {
        'gender': {
            'false_positives': set(),
            'true_negatives': set()
        }
    }
    
    # Go through each fold's results
    for fold_result in cv_results['fold_results']:
        if 'gender' in fold_result and 'user_ids_incorrect' in fold_result['gender']:
            user_ids_df = fold_result['gender']['user_ids_incorrect']
            
            # Add false positives
            false_positives = user_ids_df['false_positive_users'].dropna().tolist()
            unique_incorrect_users['gender']['false_positives'].update(false_positives)
            
            # Add true negatives
            true_negatives = user_ids_df['true_negative_users'].dropna().tolist()
            unique_incorrect_users['gender']['true_negatives'].update(true_negatives)
    
    # Convert sets to sorted lists for better readability
    result = {
        'gender': {
            'false_positives': sorted(list(unique_incorrect_users['gender']['false_positives'])),
            'true_negatives': sorted(list(unique_incorrect_users['gender']['true_negatives'])),
            'total_unique_incorrect': len(unique_incorrect_users['gender']['false_positives'] | 
                                       unique_incorrect_users['gender']['true_negatives'])
        }
    }
    
    return result
def average_confusion_matrices(fold_matrices):
    """
    Calculate the average confusion matrix across multiple cross-validation folds.
    
    Args:
        fold_matrices: List of numpy arrays, where each array is a confusion matrix from one fold
                      Each matrix should have the same shape and represent the same classes
    
    Returns:
        numpy.ndarray: The averaged confusion matrix
        
    Example:
        fold_matrices = [
            np.array([[440, 100, 10, 5],
                     [25, 180, 20, 4],
                     [5, 15, 85, 8],
                     [2, 8, 6, 24]]),
            np.array([[448, 108, 6, 3],
                     [29, 192, 26, 6],
                     [1, 23, 77, 6],
                     [0, 6, 4, 28]])
        ]
        average_matrix = average_confusion_matrices(fold_matrices)
    """
    # Convert list of matrices to 3D numpy array for easier processing
    matrices_array = np.array(fold_matrices)
    
    # Calculate mean across the first axis (across all folds)
    average_matrix = np.mean(matrices_array, axis=0)
    
    # Round to integers since confusion matrices typically use whole numbers
    average_matrix = np.round(average_matrix).astype(int)
    
    return average_matrix

def format_confusion_matrix(matrix, labels):
    """
    Format a confusion matrix into a pandas DataFrame with proper labels, handling matrices of different sizes.
    
    Args:
        matrix: numpy.ndarray containing the confusion matrix
        labels: List of string labels for the categories
    
    Returns:
        pd.DataFrame: Formatted confusion matrix with proper labels
    """

    # Create DataFrame from matrix
    df = pd.DataFrame(matrix)
    
    # Determine the actual size of the matrix
    matrix_size = matrix.shape[0]
    
    # If the matrix is 2x2, use binary labels
    if matrix_size == 2:
        column_labels = ['Negative', 'Positive']
        row_labels = ['Negative', 'Positive']
    else:
        # Use the provided labels for larger matrices
        column_labels = [f'Pred {label}' for label in labels[:matrix_size]]
        row_labels = [f'True {label}' for label in labels[:matrix_size]]
    
    # Set the labels
    df.columns = column_labels
    df.index = row_labels
    
    return df

def print_confusion_matrices(results, age_labels):
    """
    Print all confusion matrices with appropriate formatting based on their size.
    
    Args:
        results: Dictionary containing results for different traits
        age_labels: List of age range labels
    """
    for trait, trait_results in results.items():
        if 'confusion_matrix' in trait_results and trait_results['confusion_matrix'] is not None:
            print(f"\nAveraged {trait.capitalize()} Confusion Matrix:")
            matrix = trait_results['confusion_matrix']
            formatted_matrix = format_confusion_matrix(matrix, age_labels)
            print(formatted_matrix)
            print()  # 
            
def main():
    print("Starting...")
    # Set up input and output paths
    input_dir, output_dir = parse_args()
    config.set_configs(input_dir, output_dir)
    config.get_configs()
    
    # Load initial data
    data = pre.main()
    # # Run predictions for evaluation
    # image_df = image_testrun.test(config.IMAGE_DIR, data.copy(), model_name, device)
    # personality_df = text.main(Path(config.TEXT_DIR), data.copy())
    # like_predictor = UserTraitsPredictor()
    # like_df = like_predictor.predict_all(likes_path=config.LIKE_PATH, output_df=data.copy())
    
    # # Combine predictions
    # combined_df = pd.merge(
    #     data.copy(),
    #     image_df.loc[:, ['userid', 'gender']],
    #     on='userid'
    # )

    # combined_df = pd.merge(
    #     combined_df,  # Use the result from the first merge
    #     personality_df.loc[:, ['userid', 'ope', 'con', 'ext', 'agr', 'neu']],
    #     on='userid'
    # )
    
    # combined_df = pd.merge(
    #     combined_df,    
    #     like_df.loc[:,['userid','age_range']],
    #     on='userid',
    # )
    
    # Perform cross validation and get all results
    cv_results = perform_cross_validation(data.copy(), n_splits=10)
    unique_users = get_unique_incorrect_users(cv_results)

    # Define age labels that we'll use for formatting confusion matrices
    age_labels = ['xx-24', '25-34', '35-49', '50-xx']
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    incorrect_users_df = pd.DataFrame({
    'false_positive_users': pd.Series(unique_users['gender']['false_positives']),
    'true_negative_users': pd.Series(unique_users['gender']['true_negatives'])
})
    incorrect_users_df.to_csv(output_path / "unique_incorrect_users.csv", index=False)
    # First, let's handle individual fold results
    for fold_idx, fold_result in enumerate(cv_results['fold_results']):
        # Create a directory for this fold's results
        fold_dir = output_path / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(exist_ok=True)
        
        print(f"\nResults for Fold {fold_idx + 1}:")
        print_analysis_results(fold_result)
        
        # Save confusion matrices for this fold
        for trait, results in fold_result.items():
            if isinstance(results, dict) and 'confusion_matrix' in results:
                matrix = results['confusion_matrix']
                if matrix is not None:
                    formatted_matrix = format_confusion_matrix(
                        matrix,
                        age_labels if trait == 'age' else None
                    )
                    matrix_path = fold_dir / f"{trait}_confusion_matrix.csv"
                    formatted_matrix.to_csv(matrix_path)
                    print(f"Saved {trait} confusion matrix for fold {fold_idx + 1}")
                if 'metrics' in results:
                    metrics_df = pd.DataFrame([results['metrics']])
                    metrics_path = fold_dir / f"{trait}_metrics.csv"
                    metrics_df.to_csv(metrics_path, index=False)
                    print(f"Saved averaged {trait} metrics")
                # Save user IDs if they exist (for gender predictions)
                if trait == 'gender' and 'user_ids_incorrect' in results:
                    user_ids_df = results['user_ids_incorrect']
                    user_ids_path = fold_dir / f"{trait}_incorrect_users.csv"
                    user_ids_df.to_csv(user_ids_path, index=False)
                    print(f"Saved {trait} inccorect classification user IDs for fold {fold_idx + 1}")
    
    
    # Now handle averaged results
    print("\nAveraged Results Across All Folds:")
    print_analysis_results(cv_results['averaged_results'])
    
    # Save averaged confusion matrices
    averaged_dir = output_path / "averaged_results"
    averaged_dir.mkdir(exist_ok=True)
    
    for trait, results in cv_results['averaged_results'].items():
        if 'confusion_matrix' in results:
            formatted_matrix = format_confusion_matrix(
                results['confusion_matrix'], 
                age_labels if trait == 'age' else None
            )
            print(f"\nAveraged {trait.capitalize()} Confusion Matrix:")
            print(formatted_matrix)
            
            # Save the averaged matrix
            matrix_path = averaged_dir / f"{trait}_confusion_matrix.csv"
            formatted_matrix.to_csv(matrix_path)
        if 'metrics' in results:
            metrics_df = pd.DataFrame([results['metrics']])
            metrics_path = averaged_dir / f"{trait}_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            print(f"Saved averaged {trait} metrics")
        
    print("\nAll results have been saved to the output directory")
    
if __name__ == "__main__":
    main()