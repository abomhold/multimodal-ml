import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def analyze_predictions(combined_df, original_data):
    """
    Analyzes predictions from different models by comparing against ground truth data.
    Creates confusion matrices and calculates performance metrics for each prediction type.

    Parameters:
    -----------
    combined_df : pandas.DataFrame
        DataFrame containing the model predictions
    original_data : pandas.DataFrame
        DataFrame containing the ground truth values

    Returns:
    --------
    dict
        Dictionary containing analysis results for each prediction type
    """
    results = {}

    # Merge with original data to get ground truth values
    analysis_df = pd.merge(
        combined_df,
        original_data[['userid', 'gender', 'age', 'ope', 'con', 'ext', 'agr', 'neu']],
        on='userid',
        suffixes=('_pred', '_true')
    )

    # Analyze gender predictions (binary classification)
    if 'gender_pred' in analysis_df.columns and 'gender_true' in analysis_df.columns:
        gender_cm = confusion_matrix(
            analysis_df['gender_true'],
            analysis_df['gender_pred']
        )
        results['gender'] = {
            'confusion_matrix': pd.DataFrame(
                gender_cm,
                index=['True Female', 'True Male'],
                columns=['Pred Female', 'Pred Male']
            ),
            'report': classification_report(
                analysis_df['gender_true'],
                analysis_df['gender_pred'],
                target_names=['Female', 'Male'],
                output_dict=True
            )
        }

    # Analyze personality traits
    # Define RMSE thresholds for each trait
    trait_thresholds = {
        'ope': 0.65,
        'neu': 0.80,
        'ext': 0.79,
        'agr': 0.66,
        'con': 0.73
    }

    # Analyze each personality trait
    for trait, threshold in trait_thresholds.items():
        true_col = f'{trait}_true'
        pred_col = f'{trait}_pred'

        if pred_col in analysis_df.columns and true_col in analysis_df.columns:
            # Calculate absolute differences
            differences = abs(analysis_df[pred_col] - analysis_df[true_col])

            # Create binary classifications based on threshold
            within_threshold = differences < threshold

            # Calculate mean of true values for this trait
            mean_true = analysis_df[true_col].mean()
            true_high = analysis_df[true_col] >= mean_true

            # Create confusion matrix
            trait_cm = confusion_matrix(true_high, within_threshold)

            # Calculate additional metrics
            rmse = np.sqrt(np.mean((analysis_df[pred_col] - analysis_df[true_col]) ** 2))
            mae = np.mean(np.abs(analysis_df[pred_col] - analysis_df[true_col]))

            results[trait] = {
                'confusion_matrix': pd.DataFrame(
                    trait_cm,
                    index=['True Low', 'True High'],
                    columns=['Outside Threshold', 'Within Threshold']
                ),
                'metrics': {
                    'rmse': rmse,
                    'mae': mae,
                    'within_threshold_pct': (within_threshold.sum() / len(within_threshold)) * 100,
                    'mean_difference': differences.mean(),
                    'max_difference': differences.max()
                }
            }

    # Analyze age predictions
    if 'age_pred' in analysis_df.columns and 'age_true' in analysis_df.columns:
        # Define the exact age ranges we want to use
        age_bins = [0, 24, 34, 49, 100]  # The outer bounds capture all possible ages
        age_labels = ['xx-24', '25-34', '35-49', '50-xx']

        # Convert numeric true ages to our age range categories
        analysis_df['age_group_true'] = pd.cut(
            analysis_df['age_true'],
            bins=age_bins,
            labels=age_labels,
            include_lowest=True  # This ensures we capture the lowest age value
        )

        # The predicted column should already be in these ranges
        analysis_df['age_group_pred'] = analysis_df['age_pred']

        # Create confusion matrix for age groups
        age_cm = confusion_matrix(
            analysis_df['age_group_true'],
            analysis_df['age_group_pred'],
            labels=age_labels
        )

        # Calculate accuracy metrics
        total_predictions = age_cm.sum()
        correct_predictions = age_cm.diagonal().sum()
        accuracy = (correct_predictions / total_predictions) * 100

        # Calculate per-category metrics
        per_category_metrics = {}
        for i, age_range in enumerate(age_labels):
            true_positives = age_cm[i, i]
            total_actual = age_cm[i, :].sum()
            total_predicted = age_cm[:, i].sum()

            precision = true_positives / total_predicted if total_predicted > 0 else 0
            recall = true_positives / total_actual if total_actual > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            per_category_metrics[age_range] = {
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1_score': round(f1 * 100, 2)
            }

        results['age'] = {
            'confusion_matrix': pd.DataFrame(
                age_cm,
                index=[f'True {label}' for label in age_labels],
                columns=[f'Pred {label}' for label in age_labels]
            ),
            'metrics': {
                'overall_accuracy': round(accuracy, 2),
                'correct_predictions': int(correct_predictions),
                'total_predictions': int(total_predictions),
                'per_category_metrics': per_category_metrics
            }
        }

    return results

def print_analysis_results(results):
    """
    Prints the analysis results in a clear, formatted way.

    Parameters:
    -----------
    results : dict
        Dictionary containing the analysis results from analyze_predictions()
    """
    print("\nPrediction Analysis Results")
    print("=" * 50)

    # Print gender results if available
    if 'gender' in results:
        print("\nGender Prediction Results:")
        print("\nConfusion Matrix:")
        print(results['gender']['confusion_matrix'])
        print("\nClassification Report:")
        report = results['gender']['report']
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"\n{label.capitalize()}:")
                print(f"Precision: {metrics['precision']:.3f}")
                print(f"Recall: {metrics['recall']:.3f}")
                print(f"F1-score: {metrics['f1-score']:.3f}")

    # Print personality trait results
    trait_names = {
        'ope': 'Openness',
        'con': 'Conscientiousness',
        'ext': 'Extraversion',
        'agr': 'Agreeableness',
        'neu': 'Neuroticism'
    }

    for trait, name in trait_names.items():
        if trait in results:
            print(f"\n{name} Prediction Results:")
            print("\nConfusion Matrix:")
            print(results[trait]['confusion_matrix'])
            print("\nMetrics:")
            for metric, value in results[trait]['metrics'].items():
                print(f"{metric}: {value:.3f}")

    # Print age results if available
    if 'age' in results:
        print("\nAge Prediction Results:")
        print("\nAge Group Confusion Matrix:")
        print(results['age']['confusion_matrix'])
        print("\nAge Metrics:")
        print(f"RMSE: {results['age']['metrics']['rmse']:.2f} years")
        print(f"MAE: {results['age']['metrics']['mae']:.2f} years")


def split_personality_data(data, train_size=0.9, random_state=42):
    """
    Splits the personality dataset into training and test sets while maintaining
    proper randomization and data integrity.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing all personality data
    train_size : float, default=0.8
        Proportion of data to use for training (between 0 and 1)
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (train_data, test_data) - Two DataFrames containing the split data
    """
    # First, let's get all unique user IDs
    unique_users = data['userid'].unique()

    # Calculate the number of users for training
    n_train_users = int(len(unique_users) * train_size)

    # Randomly shuffle the user IDs
    np.random.seed(random_state)
    shuffled_users = np.random.permutation(unique_users)

    # Split user IDs into train and test sets
    train_users = shuffled_users[:n_train_users]
    test_users = shuffled_users[n_train_users:]

    # Create train and test datasets based on user IDs
    train_data = data[data['userid'].isin(train_users)].copy()
    test_data = data[data['userid'].isin(test_users)].copy()

    # Print split information
    print("\nData Split Summary:")
    print(f"Total users: {len(unique_users)}")
    print(f"Training users: {len(train_users)} ({train_size*100:.0f}%)")
    print(f"Test users: {len(test_users)} ({(1-train_size)*100:.0f}%)")
    print(f"\nTraining set shape: {train_data.shape}")
    print(f"Test set shape: {test_data.shape}")

    # Verify no user overlap between sets
    assert len(set(train_data['userid']) & set(test_data['userid'])) == 0, \
        "Error: Found users in both train and test sets!"

    return train_data, test_data

# Function to check data distribution in splits
def verify_split_distribution(train_data, test_data, columns_to_check=None):
    """
    Verifies that the data distribution is similar between train and test sets.

    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training dataset
    test_data : pandas.DataFrame
        Test dataset
    columns_to_check : list, optional
        List of columns to check distributions for. If None, checks all numeric columns.

    Returns:
    --------
    pandas.DataFrame
        Summary statistics comparing train and test distributions
    """
    if columns_to_check is None:
        # Get all numeric columns
        columns_to_check = train_data.select_dtypes(include=[np.number]).columns

    distribution_stats = []

    for col in columns_to_check:
        if col in train_data.columns and col in test_data.columns:
            train_mean = train_data[col].mean()
            test_mean = test_data[col].mean()
            train_std = train_data[col].std()
            test_std = test_data[col].std()

            stats = {
                'column': col,
                'train_mean': train_mean,
                'test_mean': test_mean,
                'mean_diff': abs(train_mean - test_mean),
                'train_std': train_std,
                'test_std': test_std,
                'std_diff': abs(train_std - test_std)
            }
            distribution_stats.append(stats)

    return pd.DataFrame(distribution_stats)