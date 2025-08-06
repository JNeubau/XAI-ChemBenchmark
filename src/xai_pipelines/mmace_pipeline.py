from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from src.MMACE import exmol
from src.core.fingerprints import Fingerprints
from src.xai_pipelines.base import BaseXAIPipeline


class MMacePipeline(BaseXAIPipeline):
    """
    MMACE cross-validation XAI pipeline.
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, z: pd.Series, folds: list,
                 fingerprint_type: str = 'maccs', num_samples: int = 2500, alphabet: list | None = None,
                 num_mutations: int = 2, delta: float | tuple = 1.0, nmols: int = 4):

        super().__init__(X, y, z, folds)
        self.fingerprint_type = fingerprint_type
        self.num_samples = num_samples
        self.num_mutations = num_mutations
        self.alphabet = alphabet if alphabet is not None else exmol.get_basic_alphabet()
        for smiles in z.values:
            atoms = np.unique([atom.GetSymbol() for atom in Chem.MolFromSmiles(smiles).GetAtoms()])
            for atom in atoms:
                format_atoms = [f"[{atom}]", f"[#{atom}]", f"[={atom}]"]
                self.alphabet.update(format_atoms)
        self.delta = delta
        self.nmols = nmols

    def init_explainer(self, **kwargs) -> object:
        y_train = kwargs['y_train'].to_numpy()
        y_train_median = np.median(y_train.flatten())
        return {
            'num_samples': self.num_samples,
            'alphabet': self.alphabet,
            'max_mutations': self.num_mutations,
            'fp_type': 'ECFP4',
        }, y_train_median

    def explain_model(self, model: object, X_test: pd.DataFrame, explainer: Any, smiles_list: pd.Series):
        samples_fold = []
        cfs_fold = []
        explainer, transition_point = explainer

        def get_representation(x):
            fps, column_names = Fingerprints().apply(self.fingerprint_type, smiles=[x])
            fps_df = pd.DataFrame(fps, columns=column_names)
            fps_df = fps_df.loc[:, self.filter_columns]
            return fps_df.values

        def local_predict_fn(x):
            fps = get_representation(x)
            return model.predict(fps).flatten()[0]

        for i, instance in X_test.iterrows():
            smiles = smiles_list.iloc[i]
            print(f"Processing instance {i} with SMILES: {smiles}")
            #try:
            samples = exmol.sample_space(
                smiles,
                local_predict_fn,
                stoned_kwargs=explainer,
                quiet=True,
                batched=False, )
            samples_fold.append(samples)

            # except Exception as e:
            #     print(f"An error occurred while sampling space: {e}")
            #     cfs_fold.append([])
            #     continue

            print(f"Samples: {len(samples)}")
            cfs = exmol.rcf_explain(
                samples,
                transition_point=transition_point,
                filter_nondrug=False,
                delta=self.delta,
                nmols = self.nmols,
            )

            cfs_fold.append(cfs)
        self.values['smiles'].append(smiles_list)
        self.values['counterfactuals_smiles'].append([[cf.smiles for cf in cfs] for cfs in cfs_fold])
        self.values['counterfactuals_similarity'].append([[cf.similarity for cf in cfs] for cfs in cfs_fold])
        self.values['counterfactuals_encoding'].append([[get_representation(cf.smiles) for cf in cfs] for cfs in cfs_fold])
        self.values['pred_original'].append([local_predict_fn(s) for s in smiles_list])
        self.values['pred_counterfactual'].append([[local_predict_fn(cf.smiles) for cf in cfs] for cfs in cfs_fold])

    def init_values(self):
        self.values = {
            'smiles': [],
            'counterfactuals_smiles': [],
            'counterfactuals_encoding': [],
            'counterfactuals_similarity': [],
            'pred_original': [],
            'pred_counterfactual': []
        }
