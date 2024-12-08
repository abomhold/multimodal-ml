import os
import shutil
from pathlib import Path
import pandas as pd

def setup_output_directories(base_output_path):
    """
    Create output directories for different types of incorrect predictions.
    """
    directories = {
        'false_positives': base_output_path / 'false_positives',
        'true_negatives': base_output_path / 'true_negatives'
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return directories

def copy_incorrect_images(image_source_path, output_base_path, incorrect_users_path):
    # Convert paths to Path objects
    image_source_path = Path(image_source_path)
    output_base_path = Path(output_base_path)
    
    # Read incorrect users CSV
    incorrect_users = pd.read_csv(incorrect_users_path)
    
    # Setup output directories
    directories = setup_output_directories(output_base_path)
    
    # Create a log of copied files and any errors
    log = {
        'copied_files': [],
        'not_found': [],
        'errors': []
    }
    
    # Process false positives
    if 'false_positive_users' in incorrect_users.columns:
        users = incorrect_users['false_positive_users'].dropna()
        for user_id in users:
            try:
                # Remove the int() conversion since IDs are hex strings
                source_file = image_source_path / f"{user_id}.jpg"
                if source_file.exists():
                    dest_file = directories['false_positives'] / f"{user_id}.jpg"
                    shutil.copy2(source_file, dest_file)
                    log['copied_files'].append(f"Copied {user_id} to false_positives")
                else:
                    log['not_found'].append(f"Image not found for user {user_id}")
            except Exception as e:
                log['errors'].append(f"Error processing user {user_id}: {str(e)}")
    
    # Process true negatives
    if 'true_negative_users' in incorrect_users.columns:
        users = incorrect_users['true_negative_users'].dropna()
        for user_id in users:
            try:
                # Remove the int() conversion since IDs are hex strings
                source_file = image_source_path / f"{user_id}.jpg"
                if source_file.exists():
                    dest_file = directories['true_negatives'] / f"{user_id}.jpg"
                    shutil.copy2(source_file, dest_file)
                    log['copied_files'].append(f"Copied {user_id} to true_negatives")
                else:
                    log['not_found'].append(f"Image not found for user {user_id}")
            except Exception as e:
                log['errors'].append(f"Error processing user {user_id}: {str(e)}")
    
    # Save log to file
    log_file = output_base_path / 'copy_log.txt'
    with open(log_file, 'w') as f:
        f.write("=== Image Copy Log ===\n\n")
        
        f.write("Successfully Copied Files:\n")
        for entry in log['copied_files']:
            f.write(f"{entry}\n")
            
        f.write("\nFiles Not Found:\n")
        for entry in log['not_found']:
            f.write(f"{entry}\n")
            
        f.write("\nErrors:\n")
        for entry in log['errors']:
            f.write(f"{entry}\n")
    
    # Print summary
    print(f"Copying complete!")
    print(f"Successfully copied: {len(log['copied_files'])} images")
    print(f"Files not found: {len(log['not_found'])}")
    print(f"Errors encountered: {len(log['errors'])}")
    print(f"Full log saved to: {log_file}")

if __name__ == "__main__":
    # Paths
    IMAGE_SOURCE_PATH = "./data/training/image/"
    OUTPUT_BASE_PATH = "./data/output/incorrect_users/"
    INCORRECT_USERS_CSV = "./data/output/unique_incorrect_users.csv"  # Adjust this path as needed
    
    # Run the copy operation
    copy_incorrect_images(IMAGE_SOURCE_PATH, OUTPUT_BASE_PATH, INCORRECT_USERS_CSV)