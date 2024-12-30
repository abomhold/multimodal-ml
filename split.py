import numpy as np
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix, classification_report

def analyze_predictions(combined_df, original_data):
    """
    Analyzes predictions from different models by comparing against ground truth data.
    """
    results = {}
    

    print("Available columns in combined_df:", combined_df.columns.tolist())
    print("Available columns in original_data:", original_data.columns.tolist())
    # Now analyze gender predictions
    print("\nAnalyzing gender predictions:")
    print(f"Looking for prediction column: gender_y")
    print(f"Looking for truth column: gender_x")
    print(f"Prediction column exists: {'gender_y' in combined_df.columns}")
    print(f"Truth column exists: {'gender_x' in original_data.columns}")
    
# In analyze_predictions function
# For gender analysis
    if 'gender_y' in combined_df.columns and 'gender_x' in combined_df.columns:
        try:
            gender_cm = confusion_matrix(
                combined_df['gender_x'],  # Truth
                combined_df['gender_y']   # Prediction
            )
            gender_user_ids = get_classification_user_ids(
                combined_df,
                'gender_x',
                'gender_y'
            )
            results['gender'] = {
                'confusion_matrix': pd.DataFrame(
                    gender_cm,
                    index=['True Male', 'True Female'],
                    columns=['Pred Male', 'Pred Female']
                ),
                'user_ids_incorrect': gender_user_ids,
                'metrics': classification_report(
                    combined_df['gender_x'],   # Truth
                    combined_df['gender_y'],   # Prediction
                    target_names=['Male', 'Female'],
                    output_dict=True
                )
            }
            print("\nSuccessfully created gender confusion matrix:")
            print(results['gender']['confusion_matrix'])
        except Exception as e:
            print(f"\nError creating gender confusion matrix: {str(e)}")
    else:
        print("Skipping gender analysis due to missing columns")
    # Analyze personality traits
    # Define RMSE thresholds for each trait
    trait_thresholds = {
        'ope': 0.65,
        'neu': 0.80,
        'ext': 0.79,
        'agr': 0.66,
        'con': 0.73
    }
    
     # Debug print to verify column presence
    print("\nAnalyzing trait columns:")
    
    # Analyze each personality trait
# In analyze_predictions function
# For personality traits
    for trait, threshold in trait_thresholds.items():
        pred_col = f"{trait}_y"  # prediction column with suffix
        true_col = f"{trait}_x"  # truth column with suffix
        
        print(f"\nAnalyzing {trait}:")
        print(f"Looking for prediction column: {pred_col}")
        print(f"Looking for truth column: {true_col}")
        print(f"Prediction column exists: {pred_col in combined_df.columns}")
        print(f"Truth column exists: {true_col in combined_df.columns}")
        
        if pred_col in combined_df.columns and true_col in combined_df.columns:
            # Calculate absolute differences
            differences = abs(combined_df[pred_col] - combined_df[true_col])
            
            # Binary classification: Is prediction error within threshold?
            within_threshold = differences < threshold
            
            # Calculate RMSE for each prediction
            individual_rmse = np.sqrt((combined_df[pred_col] - combined_df[true_col]) ** 2)
            
            # Binary classification: Is RMSE within threshold?
            rmse_within_threshold = individual_rmse < threshold
            
            # Create confusion matrix comparing predicted vs actual threshold compliance
            trait_cm = np.array([
                [(~within_threshold).sum(), 0],
                [0, within_threshold.sum()]
            ])
            
            # Other metrics
            rmse = np.sqrt(np.mean((combined_df[pred_col] - combined_df[true_col]) ** 2))
            mae = np.mean(np.abs(combined_df[pred_col] - combined_df[true_col]))
            
            results[trait] = {
                'confusion_matrix': pd.DataFrame(
                    trait_cm,
                    index=['True Outside Threshold', 'True Within Threshold'],
                    columns=['Pred Outside Threshold', 'Pred Within Threshold']
                ),
                'metrics': {
                    'rmse': rmse,
                    'mae': mae,
                    'within_threshold_pct': (within_threshold.sum() / len(within_threshold)) * 100,
                    'mean_difference': differences.mean()
                }
            }
        else:
            print(f"Skipping {trait} analysis due to missing columns")
    print("\nChecking age prediction columns:")
    print(f"Prediction column (age_range) exists: {'age_range' in combined_df.columns}")
    print(f"Truth column (age_true_range) exists: {'age_true_range' in combined_df.columns}")

    if 'age_range' in combined_df.columns and 'age_true_range' in combined_df.columns:
        print("\nProcessing age predictions...")
        print(f"Number of predictions: {len(combined_df)}")
        print("Sample of age data:")
        print(combined_df[['age_true_range', 'age_range']].head())  # Use combined_df instead of combined_df
        
        # Pass the merged dataframe to analyze_age_predictions
        results['age'] = analyze_age_predictions(combined_df)
    return results

def get_classification_user_ids(combined_df, true_col, pred_col):
    """
    Extracts user IDs for false positives and true negatives from prediction results.
    
    Parameters:
    -----------
    combined_df : pandas.DataFrame
        DataFrame containing predictions and ground truth
    true_col : str
        Name of column containing ground truth values
    pred_col : str
        Name of column containing predicted values
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing user IDs for false positives and true negatives
    """
    # Create boolean masks for different classification cases
    true_negative_mask = (combined_df[true_col] == 0) & (combined_df[pred_col] == 0)
    false_positive_mask = (combined_df[true_col] == 0) & (combined_df[pred_col] == 1)
    
    # Extract user IDs
    true_negatives = combined_df[true_negative_mask]['userid'].tolist()
    false_positives = combined_df[false_positive_mask]['userid'].tolist()
    
    # Create DataFrame with results
    max_length = max(len(true_negatives), len(false_positives))
    results_df = pd.DataFrame({
        'true_negative_users': true_negatives + [None] * (max_length - len(true_negatives)),
        'false_positive_users': false_positives + [None] * (max_length - len(false_positives))
    })
    
    return results_df

def convert_range_to_midpoint(age_range):
    """
    Converts an age range string to a numerical midpoint for comparison.
    
    Parameters:
    -----------
    age_range : str
        Age range category (e.g., 'xx-24', '25-34', etc.)
    
    Returns:
    --------
    float
        Midpoint age value
    """
    range_midpoints = {
        'xx-24': 20,  # Assuming 16-24 range
        '25-34': 29.5,
        '35-49': 42,
        '50-xx': 60   # Assuming up to 70
    }
    return range_midpoints.get(age_range, None)

def analyze_age_predictions(combined_df):
    """
    Analyzes age predictions comparing numerical age with predicted age ranges.
    """
    print("\n=== Age Analysis Debug Information ===")
    
    required_columns = ['age_range', 'age_true_range']
    for col in required_columns:
        print(f"\nChecking {col}:")
        print(f"Column present: {col in combined_df.columns}")
        if col in combined_df.columns:
            print("Sample values:")
            print(combined_df[col].head())
            print("Unique values:", combined_df[col].unique())
    
    valid_ranges = ['xx-24', '25-34', '35-49', '50-xx']
    
    # Create valid dataset
    valid_df = combined_df[
        combined_df['age_true_range'].isin(valid_ranges) & 
        combined_df['age_range'].isin(valid_ranges)
    ].copy()
    
    print(f"\nValid data summary:")
    print(f"Total rows: {len(combined_df)}")
    print(f"Valid rows: {len(valid_df)}")
    
    try:
        cm = confusion_matrix(
            valid_df['age_true_range'],
            valid_df['age_range'],
            labels=valid_ranges
        )
        
        # Calculate one-off accuracy
        range_order = {range_: idx for idx, range_ in enumerate(valid_ranges)}
        range_differences = abs(
            valid_df['age_true_range'].map(range_order) - 
            valid_df['age_range'].map(range_order)
        )
        one_off_accuracy = (range_differences <= 1).mean() * 100
        
        results = {
            'confusion_matrix': pd.DataFrame(
                cm,
                index=[f'True {label}' for label in valid_ranges],
                columns=[f'Pred {label}' for label in valid_ranges]
            ),
            'metrics': {
                'accuracy': (valid_df['age_true_range'] == valid_df['age_range']).mean() * 100,
                'one_off_accuracy': one_off_accuracy,
                'average_range_distance': range_differences.mean(),
                'valid_predictions': len(valid_df),
                'total_samples': len(combined_df)  # Changed from working_df to combined_df
            }
        }
        
        print("\nSuccessfully created confusion matrix:")
        print(results['confusion_matrix'])
        return results
        
    except Exception as e:
        print(f"\nError creating confusion matrix: {str(e)}")
        return None
    
def convert_age_to_range(age):
    """
    Converts a numerical age to its corresponding range category.
    Handles potential NA values.
    
    Parameters:
    -----------
    age : float or int
        Numerical age value
    
    Returns:
    --------
    str or None
        Age range category or None if age is invalid
    """
    try:
        if pd.isna(age):
            return None
        age = float(age)
        if age <= 24:
            return 'xx-24'
        elif age <= 34:
            return '25-34'
        elif age <= 49:
            return '35-49'
        else:
            return '50-xx'
    except (ValueError, TypeError):
        return None
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
        report = results['gender']['metrics']
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
    if 'age' in results and results['age'] is not None:
        print("\nAge Prediction Results:")
        print("\nAge Group Confusion Matrix:")
        print(results['age']['confusion_matrix'])
        print("\nAge Prediction Metrics:")
        
        # Print the new metrics
        metrics = results['age']['metrics']
        print(f"\nAccuracy (exact range): {metrics['accuracy']:.2f}%")
        print(f"One-off Accuracy (within one range): {metrics['one_off_accuracy']:.2f}%")
        print(f"Average Range Distance: {metrics['average_range_distance']:.2f}")
        print(f"Valid Predictions: {metrics['valid_predictions']} out of {metrics['total_samples']}")

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


