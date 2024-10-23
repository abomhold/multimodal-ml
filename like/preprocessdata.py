import pandas as pd

def preprocess_likes(relation_path: str, profile_path: str) -> tuple:
    """
    Simple preprocessing of likes data for gender prediction.
    Returns features (likes) and labels (gender).
    """
    # Load data
    relation = pd.read_csv(relation_path)
    profile = pd.read_csv(profile_path)
    
    # Group likes by user
    likes = (relation.groupby('userid')['like_id']
            .agg(lambda x: ' '.join(map(str, x)))
            .reset_index())
    
    # Merge with gender
    data = pd.merge(likes, profile[['userid', 'gender']], on='userid')
    
    return data['like_id'], data['gender']