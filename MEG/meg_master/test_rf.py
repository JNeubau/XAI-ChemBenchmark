import os
import sys
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
import argparse
from utils import get_fingerprints
import torch


"""
This is a test file to make sure the model will predict certain value for given smiles.
"""

def load_model(model_path):
    """
    Load a trained Random Forest model
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model object
    """
    try:
        model = joblib.load(model_path)
        print(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def smiles_to_fingerprint(smiles, fp_type="maccs"):
    """
    Convert a SMILES string to molecular fingerprint
    
    Args:
        smiles: SMILES string representation of molecule
        fp_type: Fingerprint type (maccs, morgan2, morgan3)
        
    Returns:
        Numpy array containing the fingerprint
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        if fp_type == "maccs":
            # MACCS keys (166 bits)
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)
        elif fp_type == "morgan2":
            # Morgan fingerprint with radius 2 (2048 bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            return np.array(fp)
        elif fp_type == "morgan3":
            # Morgan fingerprint with radius 3 (2048 bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
            return np.array(fp)
        else:
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")
    except Exception as e:
        print(f"Error converting SMILES to fingerprint: {e}")
        sys.exit(1)

def get_features_from_dataset(smiles, dataset_path="maccs_marged.csv"):
    """
    Read data from maccs_marged dataset and find matching SMILES
    
    Args:
        smiles: SMILES string to search for
        dataset_path: Path to the dataset CSV file
        
    Returns:
        Tuple of (features, predicted_capacity) if found, otherwise None
    """
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        
        # Find the row matching the SMILES string
        matching_row = df[df['smiles'] == smiles]
        
        # Check if we found a match
        if matching_row.empty:
            print(f"No matching SMILES found in dataset: {smiles}")
            return None, None
        
        # Get the first matching row (should be unique)
        row = matching_row.iloc[0]
        
        # Extract features and predicted_capacity
        feature_columns = [col for col in df.columns if col.startswith('maccsfingerprint')]
        print(len(feature_columns))
        
        # Extract features as numpy array
        features = row[feature_columns].values
        
        # Extract predicted capacity
        predicted_capacity = row['capacity_max']
        
        return features, predicted_capacity
        
    except Exception as e:
        print(f"Error reading from dataset: {e}")
        return None, None

def predict_property_from_dataset(smiles, model_path, dataset_path="data/maccs_marged.csv"):
    """
    Find molecule in dataset and predict property using model
    
    Args:
        smiles: SMILES string of the molecule
        model_path: Path to the saved model
        dataset_path: Path to the dataset
        
    Returns:
        Tuple of (features, original_capacity, predicted_capacity)
    """
    # Get features from dataset
    features, original_capacity = get_features_from_dataset(smiles, dataset_path)
    
    # if features is None:
    #     print("Using fingerprint calculation as fallback")
    #     # Fall back to fingerprint calculation if not found in dataset
    #     features = smiles_to_fingerprint(smiles, "maccs")
    #     original_capacity = None
    
    # Load model
    model = load_model(model_path)
    
    # Reshape features for prediction
    features_reshaped = features.reshape(1, -1)
    
    # Make prediction
    predicted_capacity = model.predict(features_reshaped)[0]
    
    return features, original_capacity, predicted_capacity

def predict_property_from_fp(smiles, model_path, dataset_path="data/maccs_marged.csv"):
    """
    Find molecule in dataset and predict property using model
    
    Args:
        smiles: SMILES string of the molecule
        model_path: Path to the saved model
        dataset_path: Path to the dataset
        
    Returns:
        Tuple of (features, original_capacity, predicted_capacity)
    """
    # Get features from dataset
    features_df = get_fingerprints(smiles)
    numeric_fingerprints = features_df.applymap(lambda x: int(x) if isinstance(x, str) else x)
    features = torch.tensor(numeric_fingerprints.values, dtype=torch.float32)
    original_capacity = 0
    
    # Load model
    model = load_model(model_path)
    # Reshape features for prediction
    features_reshaped = features.reshape(1, -1)
    
    # Make prediction
    predicted_capacity = model.predict(features_reshaped)[0]
    
    return features_reshaped, original_capacity, predicted_capacity

def main():
    smiles = 'O=C(O[Na])C(=O)O[Na]'
    model_path = os.path.join(os.getcwd(), 'runs_meg', 'battery', 'test', 'ckpt', 'model.joblib')
    dataset_path = os.path.join(os.getcwd(), 'data', 'maccs_merged.csv')
    
    # Get data from dataset and predict
    features, original_capacity, predicted_capacity = predict_property_from_dataset(smiles, model_path, dataset_path)
    features_fp, _, predicted_capacity_fp = predict_property_from_fp(smiles, model_path, dataset_path)
    
    # Output results
    print(f"\nSMILES: {smiles}")
    if original_capacity is not None:
        print(f"Original capacity from dataset: {original_capacity:.4f}")
    print(f"Predicted capacity: {predicted_capacity:.4f}")
    print(f"Predicted capacity from fp: {predicted_capacity_fp:.4f}")
    print(f"Features: {features}")
    features = features.tolist()
    features_fp = features_fp.flatten().detach().cpu().numpy().astype(int).tolist()
    
    print(f"Features from fp: {features_fp}")
    
    count = 0
    for i in range(len(features)):
        if features[i] != features_fp[i]:
            print(f"Feature {i} mismatch: {features[i]} vs {features_fp[i]}")
            count += 1
    print(f"Number of mismatches: {count}")

if __name__ == "__main__":
    main()