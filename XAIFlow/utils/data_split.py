import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import os

def custom_discretization(y: pd.DataFrame, num_bins: int = 4) -> pd.DataFrame:
    """
    Discretization of the continuous target attribute.
    :param y: dataframe with target
    :param num_bins: number of capacity bins to use
    :return: discretized target attribute
    """
    binned_capacity = pd.qcut(y[y.columns[0]], q=num_bins, labels=False)
    return binned_capacity


def custom_data_kfold(X: pd.DataFrame, y: pd.DataFrame, num_splits: int, num_bins: int = 4,
                      random_state: int = 42) -> list:
    """
    Performs custom data split on the provided data
    :param X: dataframe with molecules and generated features
    :param y: dataframe with target
    :param num_splits: number of folds
    :param num_bins: number of capacity bins to use
    :param random_state: random state (default: 23)
    :return: generated splits (indices)
    """
    binned_capacity = custom_discretization(y, num_bins)
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    kfolds = list(skf.split(X, binned_capacity))
    return kfolds

def save_fold_indices(folds_data, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    
    for i, (train, test) in enumerate(folds_data):
        train_df = pd.DataFrame(train)
        train_df.to_csv(os.path.join(results_dir, f'train_{i}.txt'), index=False, header=False, sep='\n')
        test_df = pd.DataFrame(test)
        test_df.to_csv(os.path.join(results_dir, f'test_{i}.txt'), index=False, header=False, sep='\n')
    print(f"Saved train/test indices for all folds.")
    
def load_fold_indices(folds_dir):
    """
    Load fold indices from saved txt files in a directory.
    
    Args:
        folds_dir (str): Path to the directory containing saved fold indices
        
    Returns:
        list: A list of tuples, each containing (train_indices, test_indices) as numpy arrays
    """
    import os
    import numpy as np
    import pandas as pd
    
    if not os.path.exists(folds_dir):
        raise FileNotFoundError(f"Directory not found: {folds_dir}")
    
    # Find all train and test files
    train_files = sorted([f for f in os.listdir(folds_dir) if f.startswith('train_') and f.endswith('.txt')])
    test_files = sorted([f for f in os.listdir(folds_dir) if f.startswith('test_') and f.endswith('.txt')])
    if not train_files or not test_files:
        raise FileNotFoundError(f"No train/test files found in {folds_dir}")
    
    # Load each pair of files and create folds
    folds = []
    for i in range(min(len(train_files), len(test_files))):
        train_file = os.path.join(folds_dir, f'train_{i}.txt')
        test_file = os.path.join(folds_dir, f'test_{i}.txt')
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            FileExistsError(f"Missing files for fold {i}")
            
        train_indices = np.array(pd.read_csv(train_file, header=None).to_numpy().flatten(), dtype=int)
        test_indices = np.array(pd.read_csv(test_file, header=None).to_numpy().flatten(), dtype=int)
        folds.append((train_indices, test_indices))
    
    if not folds:
        raise ValueError("No valid folds could be loaded")
    return folds

# def custom_data_kfold_new(X: pd.DataFrame, y: pd.DataFrame, num_splits: int, num_bins: int = 4,
#                       random_state: int = 42, smiles_mapping_path: str = None, 
#                       features_file: str = "new_maccs_marged.csv") -> list:
#     """
#     Performs custom data split on the provided data with SMILES mapping integration
    
#     :param X: dataframe with molecules and generated features
#     :param y: dataframe with target
#     :param num_splits: number of folds
#     :param num_bins: number of capacity bins to use
#     :param random_state: random state (default: 42)
#     :param smiles_mapping_path: path to the smiles_mapping.txt file
#     :param features_file: path to CSV file with features (default: "new_maccs_marged.csv")
#     :return: generated splits (indices) with enhanced feature data
#     """
#     if smiles_mapping_path is None:
#         binned_capacity = custom_discretization(y, num_bins)
#         skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
#         kfolds = list(skf.split(X, binned_capacity))
#         return kfolds
    
#     # Load the features dataframe
#     try:
#         features_df = pd.read_csv(features_file)
#         print(f"Loaded features from {features_file}, shape: {features_df.shape}")
#     except Exception as e:
#         print(f"Error loading features file: {e}")
#         # Fall back to standard stratification
#         binned_capacity = custom_discretization(y, num_bins)
#         skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
#         kfolds = list(skf.split(X, binned_capacity))
#         return kfolds
    
#     # Read SMILES mapping and organize by fold
#     fold_assignments = {}  # Dictionary of fold -> list of indices
#     smiles_mapping = {}    # Dictionary of instance_id -> SMILES
    
#     with open(smiles_mapping_path, 'r') as f:
#         for line in f:
#             # Parse line in format "{fold}_{id}: {smiles}"
#             if ':' in line:
#                 key, smiles = line.strip().split(':', 1)
#                 key = key.strip()
#                 smiles = smiles.strip()
                
#                 # Parse fold and instance ID from key
#                 try:
#                     fold_str, idx_str = key.split('_', 1)
#                     fold = int(fold_str)
#                     idx = int(idx_str)
                    
#                     # Store mapping
#                     smiles_mapping[key] = smiles
                    
#                     # Group by fold
#                     # TODO: add to fold assignments actual id from csv file
#                     if fold not in fold_assignments:
#                         fold_assignments[fold] = []
#                     fold_assignments[fold].append(idx)
                    
#                 except ValueError:
#                     print(f"Warning: Could not parse fold/index from key: {key}")
#                     continue
                    
#     print(f"Loaded {len(smiles_mapping)} SMILES mappings across {len(fold_assignments)} folds")
    
#     # Create enhanced kfolds based on the fold assignments from the SMILES mapping
#     enhanced_kfolds = []
    
#     # Sort folds numerically
#     sorted_folds = sorted(fold_assignments.keys())
    
#     for fold in sorted_folds:
#         # For each fold, all other folds' indices are training data
#         test_indices = fold_assignments[fold]
#         train_indices = []
        
#         # Collect training indices from all other folds
#         for other_fold in sorted_folds:
#             if other_fold != fold:
#                 train_indices.extend(fold_assignments[other_fold])
        
#         # Create mapping between indices and features from the SMILES
#         train_features = []
#         for idx in train_indices:
#             instance_key = f"{fold}_{idx}"  # Use current fold for lookup
#             for fold_key in fold_assignments.keys():  # Try all possible fold keys
#                 instance_id = f"{fold_key}_{idx}"
#                 if instance_id in smiles_mapping:
#                     smiles = smiles_mapping[instance_id]
#                     # Find row in features_df that matches this SMILES
#                     matching_row = features_df[features_df['smiles'] == smiles]
#                     if not matching_row.empty:
#                         feature_dict = matching_row.iloc[0].to_dict()
#                         # Remove the last 2 columns
#                         feature_dict = dict(list(feature_dict.items())[:-2])
#                         train_features.append(feature_dict)
#                         break
#             else:
#                 # If no match found in any fold, use original features (if idx is valid)
#                 if 0 <= idx < len(X):
#                     train_features.append(X.iloc[idx].to_dict())
#                 else:
#                     # Create empty features dict with same keys as feature_df
#                     empty_features = {col: None for col in features_df.columns}
#                     train_features.append(empty_features)
        
#         test_features = []
#         for idx in test_indices:
#             instance_id = f"{fold}_{idx}"
#             if instance_id in smiles_mapping:
#                 smiles = smiles_mapping[instance_id]
#                 # Find row in features_df that matches this SMILES
#                 matching_row = features_df[features_df['smiles'] == smiles]
#                 if not matching_row.empty:
#                     test_features.append(matching_row.iloc[0].to_dict())
#                 else:
#                     # If no match found, use original features (if idx is valid)
#                     if 0 <= idx < len(X):
#                         test_features.append(X.iloc[idx].to_dict())
#                     else:
#                         # Create empty features dict with same keys as feature_df
#                         empty_features = {col: None for col in features_df.columns}
#                         test_features.append(empty_features)
#             else:
#                 # If no match found, use original features (if idx is valid)
#                 if 0 <= idx < len(X):
#                     test_features.append(X.iloc[idx].to_dict())
#                 else:
#                     # Create empty features dict with same keys as feature_df
#                     empty_features = {col: None for col in features_df.columns}
#                     test_features.append(empty_features)
        
#         # Store the enhanced indices with their features
#         enhanced_kfolds.append({
#             'fold': fold,
#             'train_indices': train_indices,
#             'test_indices': test_indices,
#             'train_features': train_features,
#             'test_features': test_features
#         })
#     print(f"Fold {fold}:")
#     print(f"  Train indices: {train_indices}")
#     print(f"  Test indices: {test_indices}")
#     print(f"  Number of train features: {len(train_features)}")
#     print(f"  Number of test features: {len(test_features)}")
#     return enhanced_kfolds
    

def custom_data_split(X: pd.DataFrame, y: pd.DataFrame, train_size: float, num_bins: int = 4,
                      random_state: int = 42) -> list:
    """

    :param X: dataframe with features
    :param y:dataframe with target
    :param train_size: required train size (percentage
    :param num_bins: number of bins for disretization
    :param random_state: random state (default: 42)
    :return: custom splits (indices)
    """
    idx = X.index.tolist()
    binned_capacity = custom_discretization(y, num_bins)
    train_ids, test_ids = train_test_split(idx, train_size=train_size, random_state=random_state, shuffle=True,
                                           stratify=binned_capacity)
    return [(train_ids, test_ids)]