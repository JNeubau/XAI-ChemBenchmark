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