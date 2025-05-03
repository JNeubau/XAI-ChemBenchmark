from rdkit import Chem
from rdkit.Chem import Draw

# Original molecule SMILES
original_smiles = 'CCO'
original_mol = Chem.MolFromSmiles(original_smiles)

# Counterfactual molecule SMILES
counterfactual_smiles = 'CCN'
counterfactual_mol = Chem.MolFromSmiles(counterfactual_smiles)

# Draw molecules side by side
Draw.MolsToGridImage([original_mol, counterfactual_mol], molsPerRow=2, subImgSize=(200,200),
                     legends=['Original', 'Counterfactual'])
