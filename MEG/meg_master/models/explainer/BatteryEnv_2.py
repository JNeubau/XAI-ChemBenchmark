import utils
import torch

import numpy as np
from rdkit import Chem, DataStructs
from models.explainer.Environment import Molecule
from torch.nn import functional as F
from utils import get_similarity, mol_to_smiles, mol_from_smiles, pyg_to_mol_battery, mol_to_battery_pyg

# Define a custom environment for Battery dataset
class CF_Battery(Molecule):
    def __init__(
            self,
            model_to_explain,
            original_molecule,
            discount_factor,
            fp_len,
            fp_rad,
            similarity_set=None,
            weight_sim=0.5,
            similarity_measure="tanimoto",
            optimize_direction=1,
            **kwargs
    ):
        super(CF_Battery, self).__init__(**kwargs)

        self.class_to_optimise = 1 - original_molecule.y.item()
        self.discount_factor = discount_factor
        self.model_to_explain = model_to_explain
        self.weight_sim = weight_sim
        self.orig_pred = float(model_to_explain.predict(original_molecule.x.reshape(1, -1).numpy())[0])
        self.optimize_direction = optimize_direction

        self.similarity, self.make_encoding, \
            self.original_encoding = get_similarity(similarity_measure,
                                                    model_to_explain,
                                                    original_molecule,
                                                    fp_len,
                                                    fp_rad)

    def _reward(self):
               
        molecule = mol_from_smiles(self._state)
        molecule = mol_to_battery_pyg(molecule)

        print('mol shape x: ', molecule.x.shape)
        # Convert PyG representation to flat feature vector for RF
        features = molecule.x.reshape(1, -1)
        
        # Make prediction with RF model - returns numpy array
        prediction = self.model_to_explain.predict(features.numpy())
        pred_value = float(prediction[0])
        
        # For encoding, just use the features
        encoding = features
        # Calculate similarity score
        sim_score = self.similarity(self.make_encoding(molecule), self.original_encoding)
        
        # For regression, calculate reward based on change in the desired direction
        # Higher reward if value is moving in the desired direction from original
        value_change = (pred_value - self.orig_pred) * self.optimize_direction
        pred_score = 1.0 / (1.0 + np.exp(-value_change))  # Sigmoid to keep between 0-1
        
        # out, (_, encoding) = self.model_to_explain(molecule.x, molecule.edge_index)
        # out = F.softmax(out, dim=-1).squeeze().detach()

        # sim_score = self.similarity(self.make_encoding(molecule), self.original_encoding)
        # pred_score = out[self.class_to_optimise].item()
        # pred_class = torch.argmax(out).item()

        reward = pred_score * (1 - self.weight_sim) + sim_score * self.weight_sim

        return {
            'pyg': molecule,
            'reward': reward * self.discount_factor,
            'reward_pred': pred_score,
            'reward_sim': sim_score,
            'encoding': encoding.numpy(),
            'smiles': self._state,
            'prediction': {
                'type': 'regression',
                'output': pred_value,
                'original': self.orig_pred, 
                'difference': pred_value - self.orig_pred
            },
            'features': features.reshape(-1).numpy().tolist(),
        }