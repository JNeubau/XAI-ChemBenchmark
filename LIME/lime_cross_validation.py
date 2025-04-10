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
import exmol
# from exmol import explanation
from rdkit import Chem


class CrossValidationLIMEPipeline:
    """
    Cross-validation pipeline for training and evaluating models with LIME explanations.
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
        self.LIME_results = None
        self.cfs = None

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

    def init_scores_LIME(self):
        """
        Initialize scores dictionary and LIME results.
        """
        scores = {}
        for metric in self.metrics:
            scores[metric] = []
        self.scores = scores
        self.LIME_results = []
        self.cfs = []

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

    def generate_lime_explanations(self, model, X_test: pd.DataFrame, list_smiles: pd.Series):
        """
        Generate LIME explanations using the exmol library.
        :param model: trained model.
        :param X_test: test data.
        :param list_smiles: series with SMILES string dla każdej instancji.
        :return: lista słowników z wyjaśnieniami dla poszczególnych instancji.
        """
        lime_explanations = []

        def local_predict_fn(x):

            smiles_index = list_smiles[list_smiles == smiles].index[0]
            maccs_array = X_test.iloc[smiles_index].values.reshape(1, -1)
            lables = X_test.columns
            print(f"SMILES: {smiles}, MACCS array: {maccs_array}")

            print(f"Prediction: {model.predict(maccs_array).flatten()}")
            return model.predict(maccs_array).flatten()

        for i, instance in X_test.iterrows():
            if 'OB(O)' in list_smiles.iloc[i]:
                continue
            if 'K' in list_smiles.iloc[i]:
                continue
            if 'Mg' in list_smiles.iloc[i]:
                continue
            if 'Na' in list_smiles.iloc[i]:
                continue

            # instance_array = instance.values.reshape(1, -1)
            smiles = list_smiles.iloc[i]
            # explainer = exmol.Explanation(
            #     predict_fn=local_predict_fn,
            #     method="lime",
            #     num_samples=1000,       # liczba wygenerowanych perturbacji
            #     kernel_width=0.25       # parametr jądra
            # )
            # explanation_result = explainer.explain_instance(instance_array)

            samples = exmol.sample_space(
                smiles, 
                local_predict_fn, 
                batched=False,
                #preset='chemed',
                num_samples=10,
                use_selfies=False)
            
            # if samples != None:
            print(f"Samples: {len(samples)}")
            # cfs = exmol.cf_explain(samples)
            # self.cfs.append(cfs)
            # exmol.plot_cf(cfs)

            beta=exmol.lime_explain(samples, descriptor_type='MACCS', return_beta=True)
            # plot_path = os.path.join(self.save_dir, f"lime_explanation_instance_{i}.png")
            # import skunk

            # svg = exmol.plot_descriptors(samples, return_svg=True)
            # skunk.display(svg)
            # svg = exmol.plot_utils.similarity_map_using_tstats(samples[0], return_svg=True)
            # skunk.display(svg)           
            # explanation_result = exmol.plot_descriptors(samples, output_file='MACCS.png')

            # plot_path = os.path.join(self.save_dir, f"lime_explanation_instance_{i}.png")
            # explanation_result.save_plot(plot_path)

            lime_explanations.append({"smiles": list_smiles.iloc[i], "explanation": beta})

        return lime_explanations

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> tuple:
        """
        Train the model.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: tuple with results, scores, and LIME results.
        """
        proper_model_name, model, param_grid = Models().get_model(model_name, model_path=model_path)
        self.init_scores_LIME()

        if self.verbose:
            print(f"Training model {proper_model_name}")
        i=0
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

            if i == 0:
                LIME_explanations = self.generate_lime_explanations(model, X_test, smiles['smiles'])
                self.LIME_results.append({"fold": fold, "explanations": LIME_explanations})
            i+=1

        results = self.aggregate_scores()

        if len(self.save_dir) > 0:
            self.save_results(results, proper_model_name, model.get_params())

        return results, self.scores, self.LIME_results
