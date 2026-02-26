import os

import pickle
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter

from src.MEG.explainer import MegRegressionExplainer
from src.core.fingerprints import Fingerprints
from src.xai_pipelines.base import BaseXAIPipeline

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)



class MegPipeline(BaseXAIPipeline):
    """
    MMACE cross-validation XAI pipeline.
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, z: pd.Series, folds: list,
                 fingerprint_type: list[str], fp_params: dict, delta: float = 1.0, samples: int = 10, epochs:int = 10):

        super().__init__(X, y, z, folds)
        self.fingerprint_type = fingerprint_type
        for key, value in fp_params.items():
            if value is None:
                fp_params[key] = {}
        self.fingerprint_params = fp_params
        self.delta = delta
        self.samples = samples
        self.epochs = epochs

    def init_explainer(self, **kwargs) -> object:
        os.makedirs(f'{kwargs["base_path"]}/meg_explainer', exist_ok=True)
        writer = f"{kwargs['base_path']}/meg_explainer" #SummaryWriter(f'{kwargs['base_path']}/meg_explainer')

        y_train = kwargs['y_train'].to_numpy()
        y_train_median = np.median(y_train.flatten())

        kwargs['env_params']['filter_columns'] = self.filter_columns
        kwargs['env_params']['fingerprint_type'] = self.fingerprint_type
        kwargs['env_params']['fp_params'] = self.fingerprint_params
        kwargs['agent_params']['sample'] = kwargs['sample']

        explainer = MegRegressionExplainer(
            writer=writer,
            target_diff=self.delta,
            transition_point=y_train_median,
            env_params=kwargs['env_params'],
            agent_params=kwargs['agent_params'],
            samples=self.samples,
            epochs=self.epochs
        )
        return explainer

    @staticmethod
    def preprocess_results(results: list) -> dict:
        """
        Preprocess the results from the explainer.
        :param results: List of results from the explainer.
        :return: Dictionary with processed results.
        """
        processed_results = {
            'counterfactuals_smiles': [],
            'counterfactuals_encoding': [],
            'counterfactuals_similarity': [],
            'counterfactuals_pred_reward': [],
            'pred_original': [],
            'pred_counterfactual': []
        }
        cfs = [r for r in results if r['marker'] == 'cf']

        for cf in cfs:
            processed_results['counterfactuals_smiles'].append(cf['smiles'])
            processed_results['counterfactuals_encoding'].append(cf['encoding'])
            processed_results['counterfactuals_similarity'].append(cf['reward_sim'])
            processed_results['counterfactuals_pred_reward'].append(cf['reward_pred'])
            processed_results['pred_counterfactual'].append(cf['prediction']['output'])
            processed_results['pred_original'].append(cf['prediction']['original'])

        return processed_results

    def explain_model(self, model: object, X_test: pd.DataFrame, explainer: Any, smiles_list: pd.Series):
        values_fold = {
            'counterfactuals_smiles': [],
            'counterfactuals_encoding': [],
            'counterfactuals_similarity': [],
            'counterfactuals_pred_reward': [],
            'pred_original': [],
            'pred_counterfactual': []
        }

        def run_example(i, smiles):
            import warnings
            from rdkit import RDLogger
            warnings.filterwarnings("ignore")
            RDLogger.DisableLog('rdApp.*')

            results = explainer.explain(model, smiles, sample=i)
            processed_results = self.preprocess_results(results)
            return processed_results, smiles

        results_fold = Parallel(n_jobs=49)(delayed(run_example)(i, smiles) for i, smiles in enumerate(smiles_list))
        results_cf = [c[0] for c in results_fold]
        smiles_results = [c[1] for c in results_fold]
        for r in results_cf:
            for k in r:
                values_fold[k].append(r[k])
        self.values['smiles'].append(smiles_results)
        for k in values_fold:
            self.values[k].append(values_fold[k])

        with open(f'{explainer.writer}/results.pickle', 'wb') as f:
            pickle.dump(self.values, f)

    def init_values(self):
        self.values = {
            'smiles': [],
            'counterfactuals_smiles': [],
            'counterfactuals_encoding': [],
            'counterfactuals_similarity': [],
            'counterfactuals_pred_reward': [],
            'pred_original': [],
            'pred_counterfactual': []
        }