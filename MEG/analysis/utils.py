from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import pandas as pd
import numpy as np
import os

def get_fingerprints(smiles, maccs_merge_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Invalid SMILES string: {smiles}")
        # Return a dummy fingerprint with all zeros
        dummy_fps = np.zeros((1, 166), dtype=int)
        return pd.DataFrame(dummy_fps, columns=[f'maccsfingerprint{i}' for i in range(166)])
    # Generate MACCS fingerprints for the input SMILES
    # print(f"Generating MACCS fingerprints for SMILES: {x}")
    fps = [list(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles)).ToBitString())]
    fps = np.array(fps)[:, 1:]
    fps_df = pd.DataFrame(fps, columns=[f'maccsfingerprint{i}' for i in range(len(fps[0]))])

    if not os.path.exists(maccs_merge_path):
        raise FileNotFoundError(f"MACCS merge file not found: {maccs_merge_path}")

    maccs_merge = pd.read_csv(maccs_merge_path)
    maccs_merge = maccs_merge.loc[:, maccs_merge.columns.str.contains('maccs', case=False)]
    selected_keys = maccs_merge.columns.tolist()

    selected_keys = [key for key in selected_keys if key in fps_df.columns]

    # Filter the fingerprints DataFrame to include only the selected keys
    filtered_fps = fps_df[selected_keys]
    return filtered_fps

def get_smarts(fp_name, smarts_mapping_path):
    if not os.path.exists(smarts_mapping_path):
        raise FileNotFoundError(f"SMARTS mapping file not found: {smarts_mapping_path}")
    smarts_mapping = pd.read_json(smarts_mapping_path)
    smarts = smarts_mapping.loc[0, fp_name] if fp_name in smarts_mapping.columns else ''
    return smarts

def tanimoto_similarity(smiles1, smiles2):
    fp1 = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles1))
    fp2 = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles2))
    return DataStructs.TanimotoSimilarity(fp1, fp2)