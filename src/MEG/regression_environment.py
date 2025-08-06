from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from sklearn.metrics import mean_absolute_error

from src.MEG.environment import MoleculeEnvironment
from src.core.fingerprints import Fingerprints


class RegressionEnvironment(MoleculeEnvironment):
    def __init__(
            self,
            atom_types: np.ndarray,
            model_to_explain: Any,
            target_diff: float,
            transition_point: float,
            original_molecule: str,
            fingerprint_type: str,
            fp_len: int,
            fp_rad: int | float,
            discount_factor: float,
            filter_columns: list | None = None,
            weight_sim: float = 0.5,
            **kwargs
    ):
        super(RegressionEnvironment, self).__init__(
            atom_types,
            original_molecule,
            **kwargs
        )
        self.model_to_explain = model_to_explain
        self.discount_factor = discount_factor
        self.weight_sim = weight_sim
        self.fingerprint_type = fingerprint_type
        self.filter_columns = filter_columns
        self.distance = lambda x, y: mean_absolute_error(x, y)
        self.original_molecule_dict = self.mol_to_dict(original_molecule)
        self.origin_pred = model_to_explain.predict(self.original_molecule_dict['x'])

        self.optim_direction = np.sign(transition_point - self.origin_pred)
        self.target_diff = np.max([target_diff, self.distance([transition_point], self.origin_pred)])

        self.similarity, self.make_encoding, self.original_encoding = self.get_similarity(self.original_molecule_dict, fp_len,
                                                                                          fp_rad)

    @staticmethod
    def get_similarity(original_molecule, fp_len, fp_rad):
        similarity = lambda x, y: TanimotoSimilarity(x, y)
        make_encoding = lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x['smiles']), fp_rad, fp_len, bitInfo=None)
        original_encoding = make_encoding(original_molecule)

        return similarity, make_encoding, original_encoding

    def mol_to_dict(self, molecule: str):
        fps, column_names = Fingerprints().apply(self.fingerprint_type, smiles=[molecule])
        fps_df = pd.DataFrame(fps, columns=column_names)
        if self.filter_columns:
            fps_df = fps_df.loc[:, self.filter_columns]
        fingerprint = fps_df.values
        return {
            'smiles': molecule,
            'x': fingerprint
        }

    def reward(self):
        molecule = self.mol_to_dict(self.state)
        prediction = self.model_to_explain.predict(molecule['x'])
        value_change = self.distance(self.origin_pred, prediction)
        change_direction = np.sign(prediction - self.origin_pred)
        is_in_optim_direction = change_direction == self.optim_direction

        if prediction < 0:
            diff = -1.0
        elif value_change >= self.target_diff and change_direction == self.optim_direction:
            diff = 1.0
        else:
            diff = np.min([value_change / self.target_diff, 1.0])
            diff = diff if is_in_optim_direction else -diff
            diff = 1 / (1 + np.exp(-diff))

        sim_score = self.similarity(self.make_encoding(molecule), self.original_encoding)
        reward = diff * (1 - self.weight_sim) + sim_score * self.weight_sim

        return {
            'molecule': molecule,
            'reward': reward * self.discount_factor,
            'reward_pred': diff,
            'reward_sim': sim_score,
            'encoding': molecule['x'],
            'smiles': molecule['smiles'],
            'prediction': {
                'type': 'regression',
                'output': prediction,
                'original': self.origin_pred,
                'difference': value_change,
                'target_difference': self.target_diff,
            }
        }
