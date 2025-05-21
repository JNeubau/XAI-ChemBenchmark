from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import copy
from typing import Any, Iterable
import numpy as np
import pandas as pd
import shapiq
import os

from datetime import datetime
from XAIFlow.AI_models.models import Models
from XAIFlow.AI_models.eval_metrics import EvalMetrics
from XAIFlow.utils.data_split import custom_data_split


class CrossValidationShapIqPipeline:
    """
    Cross-validation pipeline class for SHAP-IQ.
    """

    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            folds: list,
            metrics: list,
            save_dir: str,
            data_name: str,
            hyperparam_opt: bool = True,
            verbose: bool = False,
            iq_min_order: int = 1,
            iq_max_order: int = 1,
    ):
        """
        Initialize the cross-validation pipeline.
        :param X: dataframe with features.
        :param y: dataframe with target variable.
        :param numerical_features: list with numerical features.
        :param categorical_features: list with categorical features.
        :param folds: list with cross-validation folds.
        :param metrics: list with metrics to evaluate.
        :param save_dir: path to save scores.
        :param data_name: name of the dataset.
        :param verbose: whether to print model scores.
        """
        self.X = X
        self.y = y
        self.folds = folds
        self.data_name = data_name
        self.save_dir = save_dir
        self.verbose = verbose
        self.metrics = metrics
        self.hyperparam_opt = hyperparam_opt
        self.scores = None
        self.shap_values = None
        self.iq_min_order = iq_min_order
        self.iq_max_order = iq_max_order
        self.model = None

    def tune_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model: object,
                   param_grid: dict | None) -> object:
        """
        Perform model tuning.
        :param X_train: training data.
        :param y_train: target variable data.
        :param model: prediction model.
        :param param_grid: dictionary with parameter grid.
        :return: model with optimized parameters.
        """
        if self.hyperparam_opt:
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
            folds = custom_data_split(X_train, y_train, train_size=0.6)
            opt = GridSearchCV(estimator=model, param_grid=param_grid, cv=folds, scoring=scorer, refit=True, n_jobs=-1,
                               return_train_score=True)
            opt.fit(X_train.to_numpy(), y_train[y_train.columns[0]].to_numpy())
            best_score, best_params = opt.best_score_, opt.best_params_
            model.set_params(**best_params)
            if self.verbose:
                print(f"Best score: {best_score}\n Best params: {best_params}")
        return model

    @staticmethod
    def explain_model(model: object, X_test: pd.DataFrame, min_order, max_order) -> Iterable:
        """
        Explain the model.
        :param model: prediction model.
        :param X_test: test data.
        :return: shap values.
        """
        explainer = shapiq.TreeExplainer(model, index='SV', min_order=min_order, max_order=max_order)
        # explainer = shap.TreeExplainer(model)
        # print(X_test.head())
        shap_values = []
        new_X_test = X_test.to_numpy()
        for i in range(new_X_test.shape[0]):
            shap_value = explainer.explain(new_X_test[i])
            shap_values.append(shap_value.values)
        shap_values = np.array(shap_values)
        return shap_values

    def explain_model_interaction(self, model, X_test, min_order, max_order):
        """
        Explain the model interaction.
        :param model: prediction model.
        :param X_test: test data.
        :return: shap values.
        """
        explainer = shapiq.TreeExplainer(model, index='k-SII', min_order=min_order, max_order=max_order)
        # explainer = shapiq.TreeExplainer(model, index='SV', min_order=min_order, max_order=max_order)
        # print(X_test.head())
        shap_values = []
        new_X_test = X_test.to_numpy()
        for i in range(new_X_test.shape[0]):
            shap_value = explainer.explain(new_X_test[i])
            shap_values.append(shap_value)
        return shap_values

    def update_shap(self, model: object, X_test: pd.DataFrame):
        """
        Update shap values.
        :param model: prediction model.
        :param X_test: test data.
        """
        shap_values = self.explain_model_interaction(model, X_test, min_order=self.iq_min_order, max_order=self.iq_max_order)
        # shap_values = self.explain_model(model, X_test, min_order=self.iq_min_order, max_order=self.iq_max_order)
        self.shap_values.append(shap_values)

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

    def init_scores_shap(self):
        """
        Initialize scores dictionary and shap.
        """
        scores = {}
        for metric in self.metrics:
            scores[metric] = []
        self.scores = scores
        self.shap_values = []

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
            f"{model_name}\n" f"Model parameters: {model_params}\n",
            f"Training data: {self.data_name}\n",
            *metric_lines,
            "******************************************************\n",
        ]
        if self.verbose:
            for l in lines:
                print(l)
        with open(save_scores_path, "a", encoding="utf-8") as f:
            f.writelines(lines)

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> tuple:
        """
        Train the model.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: tuple with results, scores, explanations.
        """
        proper_model_name, model, param_grid = Models().get_model(model_name, model_path=model_path)
        self.init_scores_shap()

        if self.verbose:
            print(f"Training model {proper_model_name}")

        for fold in self.folds:
            train_idx, test_idx = fold

            # train-test split
            X_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)

            model = self.tune_model(X_train, y_train, model, param_grid)

            # model training
            model.fit(X_train, y_train[y_train.columns[0]])
            self.model = model

            y_pred = model.predict(X_test).flatten()

            # model eval
            y_test_numpy = y_test.to_numpy().flatten()
            y_pred_eval = self.eval_model(y_pred, y_test_numpy)

            self.update_scores(y_pred_eval)
            self.update_shap(model, X_test)

        results = self.aggregate_scores()

        if len(self.save_dir) > 0:
            self.save_results(results, proper_model_name, model.get_params())

        return results, self.scores, self.shap_values
    
    def predict_capacity(self, X_input):
        """
        Predict capacity using the trained model.
        :param X_input: input data for prediction.
        :return: predicted values.
        """        
        prediction = self.model.predict(X_input)
        return prediction