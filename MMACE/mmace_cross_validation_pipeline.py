from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import copy
from typing import Any, Iterable
import numpy as np
import pandas as pd
import os
from datetime import datetime
from XAIFlow.AI_models.models import Models
from XAIFlow.AI_models.eval_metrics import EvalMetrics, smape
from XAIFlow.utils.data_split import custom_data_split
# from XAIFlow.utils.fingerprints import Fingerprints
import exmol
from rdkit import Chem
import signal
import matplotlib.pyplot as plt
from MMACE.timeoutexception import timeout
import joblib

class CrossValidationMMACEPipeline:
    """
    Cross-validation pipeline for training and evaluating models with MMACE explanations.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        z: pd.DataFrame,
        folds: list,
        metrics: list,
        save_dir: str,
        data_name: str,
        hyperparam_opt: bool = True,
        verbose: bool = False,
        custom_alphabet: set = None
    ):
        """
        Initialize the cross-validation pipeline.
        :param X: dataframe with features.
        :param y: dataframe with target variable.
        :param z: dataframe with SMILES strings.
        :param folds: list with cross-validation folds.
        :param metrics: list with metrics to evaluate.
        :param save_dir: path to save scores.
        :param data_name: name of the dataset.
        :param verbose: whether to print model scores.
        """
        self.X = X
        self.y = y
        self.z = z
        self.folds = folds
        self.data_name = data_name
        self.save_dir = save_dir
        self.verbose = verbose
        self.metrics = metrics
        self.hyperparam_opt = hyperparam_opt
        self.scores = None
        self.MMACE_results = None
        self.cfs = None
        self.samples = None
        self.custom_alphabet = custom_alphabet

    def tune_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model: object, param_grid: dict | None) -> object:
        """
        Perform model tuning.
        :param X_train: training data.
        :param y_train: target variable data.
        :param model: prediction model.
        :param param_grid: dictionary with parameter grid.
        :return: model with optimized parameters.
        """
        if self.hyperparam_opt:
            scorer = make_scorer(smape, greater_is_better=False)
            folds = custom_data_split(X_train, y_train, train_size=0.6)
            opt = GridSearchCV(estimator=model, param_grid=param_grid, cv=folds, scoring=scorer, refit=True, n_jobs=-1,
                               return_train_score=True)
            opt.fit(X_train.to_numpy(), y_train[y_train.columns[0]].to_numpy())
            best_score, best_params = opt.best_score_, opt.best_params_
            model.set_params(**best_params)
            if self.verbose:
                print(f"Best score: {best_score}\nBest params: {best_params}")
        return model

    def eval_model(self, y_pred: np.array, y_test: np.array) -> dict:
        """
        Evaluate the model.
        :param y_pred: predicted values.
        :param y_test: true values.
        :return: dictionary with evaluation metrics.
        """
        partial_scores = {}
        for metric in self.metrics:
            partial_scores[metric] = EvalMetrics().evaluate(metric, y_test, y_pred)
        return partial_scores

    def init_scores_MMACE(self):
        """
        Initialize scores dictionary and MMACE results.
        """
        scores = {}
        for metric in self.metrics:
            scores[metric] = []
        self.scores = scores
        self.MMACE_results = []
        self.cfs = []
        self.samples = []

    def update_scores(self, model_scores: dict):
        """
        Update scores dictionary.
        :param model_scores: dictionary with model scores.
        """
        for metric in self.metrics:
            self.scores[metric].append(model_scores[metric])

    def aggregate_scores(self):
        """
        Aggregate scores.
        :return: aggregated scores.
        """
        results = {}
        for metric in self.metrics:
            metric_scores = self.scores[metric]
            results[metric] = round(sum(metric_scores) / len(metric_scores), 4)
        return results

    def save_results(self, results: dict, model_name: str, model_params: Any):
        """
        Save results to a file.
        :param results: dictionary with results.
        :param model_name: name of the model.
        :param model_params: model parameters.
        """
        save_scores_path = os.path.join(self.save_dir, f"results_{model_name}.txt")
        metric_lines = [f"{metric}: {score}\n" for metric, score in results.items()]
        lines = [
            "\n******************************************************\n",
            f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n",
            f"{model_name}\nModel parameters: {model_params}\n",
            f"Training data: {self.data_name}\n",
            *metric_lines,
            "******************************************************\n",
        ]
        if self.verbose:
            for l in lines:
                print(l)
        with open(save_scores_path, "a", encoding="utf-8") as f:
            f.writelines(lines)

    def generate_MMACE_explanations(self, model, X_test: pd.DataFrame, list_smiles: pd.Series, fold: int = 0, maccs_merge_path=None) -> tuple:
        """
        Generate MMACE explanations using the exmol library.
        :param model: trained model.
        :param X_test: test data.
        :param list_smiles: series with SMILES string for each instance.
        :param fold: fold number for tracking.
        :return: tuple with explanations, samples and counterfactuals.
        """
        MMACE_explanations = []
        samples_fold = []
        cfs_fold = []

        # parent_dir = os.path.dirname(os.getcwd())
        # # Ensure maccs_merge_path is always set
        # if maccs_merge_path is None:
        #     maccs_merge_path = os.path.join(parent_dir, 'data', 'new_maccs_merged.csv')

        def local_predict_fn(x):
            fps = [list(Chem.MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x)).ToBitString())]
            fps = np.array(fps)[:, 1:]
            fps_df = pd.DataFrame(fps, columns=[f'maccsfingerprint{i}' for i in range(len(fps[0]))])

            # # maccs_merge_path is now always defined
            # if not os.path.exists(maccs_merge_path):
            #     raise FileNotFoundError(f"MACCS merge file not found: {maccs_merge_path}")

            # maccs_merge = pd.read_csv(maccs_merge_path)
            # maccs_merge = maccs_merge.loc[:, maccs_merge.columns.str.contains('maccs', case=False)]
            # selected_keys = maccs_merge.columns.tolist()

            # selected_keys = [key for key in selected_keys if key in fps_df.columns]
            # filtered_fps = fps_df[selected_keys]
            return model.predict(fps_df).flatten()

        for i, instance in X_test.iterrows():
            smiles = list_smiles.iloc[i]
            print(f"Processing instance {i} with SMILES: {smiles}")
            try:
                stoned_kwargs = {
                    "num_samples": 20,
                    "alphabet": self.custom_alphabet if self.custom_alphabet else exmol.get_basic_alphabet(),
                    "max_mutations": 2,
                }
                
                samples = exmol.sample_space(
                    smiles, 
                    local_predict_fn,
                    stoned_kwargs=stoned_kwargs, 
                    quiet=True,
                    batched=False,)
                samples_fold.append(samples)
                    
            except Exception as e:
                print(f"An error occurred while sampling space: {e}")
                cfs_fold.append([])
                continue
            
            print(f"Samples: {len(samples)}")
            cfs = exmol.rcf_explain(
                samples,
                filter_nondrug = False,
                delta=0.25,
                nmols=4
                )
            
            cfs_fold.append(cfs)
        
        self.MMACE_results.append({"fold": fold, "explanations": cfs_fold})

        return MMACE_explanations, samples_fold, cfs_fold

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> tuple:
        """
        Train the model.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: tuple with results, scores, and MMACE results.
        """
        proper_model_name, model, param_grid = Models().get_model(model_name, model_path=model_path)
        self.init_scores_MMACE()

        if self.verbose:
            print(f"Training model {proper_model_name}")
        foldid=0

        for fold in self.folds:
            train_idx, test_idx = fold

            X_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)
            smiles = copy.deepcopy(self.z.loc[test_idx, :]).reset_index(drop=True)

            model = self.tune_model(X_train, y_train, model, param_grid)

            model.fit(X_train, y_train[y_train.columns[0]])
            y_pred = model.predict(X_test).flatten()

            y_test_numpy = y_test.to_numpy().flatten()
            model_scores = self.eval_model(y_pred, y_test_numpy)
            self.update_scores(model_scores)

            self.generate_MMACE_explanations(model, X_test, smiles['smiles'], fold=foldid)
            foldid+=1

        results = self.aggregate_scores()

        if len(self.save_dir) > 0:
            self.save_results(results, proper_model_name, model.get_params())

        return results, self.scores, self.cfs, self.samples, self.MMACE_results

    def load_model(self, model_path: str) -> object:
        """
        Load a trained model from a file.
        :param model_path: path to the saved model.
        :return: loaded model.
        """
        if os.path.exists(model_path):
            loaded_model = joblib.load(model_path)
            if self.verbose:
                print(f"Model loaded from {model_path}")
            return loaded_model
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
    def load_pipeline(self, model_path: str | None = None) -> tuple:
        """
        Run the pipeline by loading a model for each fold and generating MMACE explanations.
        :param model_name: name of the model.
        :param model_path: path to directory with saved models (should contain model_{i}.joblib for each fold).
        :return: tuple with results, scores, and MMACE results.
        """
        # proper_model_name, model, param_grid = Models().get_model(model_name, model_path=model_path)
        # model_name = "MMACE"
        self.init_scores_MMACE()
        if self.verbose:
            print(f"Training model MMACE")

        foldid = 0
        for i, fold in enumerate(self.folds):
            train_idx, test_idx = fold

            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            # y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)
            smiles = copy.deepcopy(self.z.loc[test_idx, :]).reset_index(drop=True)

            # Load model for this fold
            if model_path is None:
                raise ValueError("model_path must be provided for loading models per fold.")
            model_file = os.path.join(model_path, f"model_{i}.joblib")
            model = self.load_model(model_file)

            self.model=model
            # Evaluate model
            # y_pred = model.predict(X_test).flatten()
            # y_test_numpy = y_test.to_numpy().flatten()
            # model_scores = self.eval_model(y_pred, y_test_numpy)
            # self.update_scores(model_scores)

            # Generate MMACE explanations
            self.generate_MMACE_explanations(model, X_test, smiles['smiles'], fold=foldid)
            foldid += 1 # why not i zamiast specjalne foldid?

        # results = self.aggregate_scores()

        # if len(self.save_dir) > 0:
        #     self.save_results(results, model_name, model.get_params())

        return self.cfs, self.samples, self.MMACE_results
