from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import copy
from typing import Any, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from XAIFlow.AI_models.models import Models
from XAIFlow.AI_models.eval_metrics import EvalMetrics, smape, rmse, mape
from XAIFlow.utils.data_split import custom_data_split
import lime
import lime.lime_tabular

class CrossValidationLimePipeline:
    """
    Cross-validation pipeline class for LIME.
    """

    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            z: pd.Series,  # SMILES data
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
        :param z: Series with SMILES strings.
        :param folds: list with cross-validation folds.
        :param metrics: list with metrics to evaluate.
        :param save_dir: path to save scores.
        :param data_name: name of the dataset.
        :param verbose: whether to print model scores.
        """
        self.X = X
        self.y = y
        self.z = z  # Store SMILES data
        self.folds = folds
        self.data_name = data_name
        self.save_dir = save_dir
        self.verbose = verbose
        self.metrics = metrics
        self.hyperparam_opt = hyperparam_opt
        self.scores = None
        self.lime_values = None
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
            scorer = make_scorer(smape, greater_is_better=False)
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
    def explain_model(model: object, X_test: pd.DataFrame, expainer: object, smiles_list: pd.Series,f: int) -> Iterable:
        """
        Explain the model and save explanation plots.
        :param model: prediction model.
        :param X_test: test data.
        :param expainer: LIME explainer object
        :param smiles_list: Series containing SMILES strings for molecules
        :return: lime values.
        """
        # from sklearn.linear_model import Lasso
        # from sklearn.ensemble import RandomForestRegressor

        # lasso = Lasso(alpha=0.01,random_state=42)
        # rf = RandomForestRegressor(n_estimators=100, random_state=42)

        lime_values = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parent_dir = os.path.dirname(os.getcwd())

        # Create plots directory
        plots_dir = os.path.join(parent_dir, 'results', 'plots', "LIME", datetime.today().strftime("%d-%m-%Y"))
        os.makedirs(plots_dir, exist_ok=True)

        for idx, (instance, smiles) in enumerate(zip(X_test.values, smiles_list)):
            print(f"Processing molecule {idx}, SMILES: {smiles}")
            # Get explanation
            lime_explanation = expainer.explain_instance(instance, model.predict, num_features=5)
            lime_value = lime_explanation.as_list()
            lime_values.append(lime_value)
            
            try:
                # Save explanation plot
                fig = lime_explanation.as_pyplot_figure()
                fig.suptitle(f'LIME Explanation for SMILES: {smiles}')
                
                # Create more detailed filename with timestamp and SMILES
                plot_path = os.path.join(
                    plots_dir,
                    f"lime_explanation_{f}_{idx}_{timestamp}.svg"
                )
                
                # Save high quality vector graphics
                fig.savefig(plot_path, bbox_inches='tight', dpi=300, format='svg')
                plt.close(fig)
                
                if lime_values:
                    print(f"Explanation saved for SMILES {smiles} at {plot_path}")
                
                # Save explanation to a html file
                lime_explanation.save_to_file(os.path.join(plots_dir, f"lime_explanation_{f}_{idx}_{timestamp}.html"))
                print(f"Explanation saved to HTML for SMILES {smiles} at {plot_path}")
                    
            except Exception as e:
                print(f"An error occurred while saving explanation for SMILES {smiles}: {e}")
        
        return lime_values

    def update_lime(self, model: object, X_test: pd.DataFrame, explainer: object, f: int, smiles_test: pd.Series):
        """
        Update lime values.
        :param model: prediction model.
        :param X_test: test data.
        """
        # Get SMILES from the test data index
        smiles_list = smiles_test
        lime_values = self.explain_model(model, X_test, explainer, smiles_list,f)
        self.lime_values.append(lime_values)

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

    def init_scores_lime(self):
        """
        Initialize scores dictionary and lime.
        """
        scores = {}
        for metric in self.metrics:
            scores[metric] = []
        self.scores = scores
        self.lime_values = []

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
        self.init_scores_lime()

        if self.verbose:
            print(f"Training model {proper_model_name}")
        f=0
        for fold in self.folds:
            train_idx, test_idx = fold

            # train-test split
            X_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)
            # SMILES data
            # smiles_train = copy.deepcopy(self.z.loc[train_idx]).reset_index(drop=True)
            smiles_test = copy.deepcopy(self.z.loc[test_idx]).reset_index(drop=True)
            model = self.tune_model(X_train, y_train, model, param_grid)

            # model training
            model.fit(X_train, y_train[y_train.columns[0]])
            self.model = model

            y_pred = model.predict(X_test).flatten()
            feature_names = list(X_train.columns)
            n_features = X_train.shape[1]

            # All features are categorical (binary)
            categorical_features = list(range(n_features))
            categorical_names = {i: [0, 1] for i in categorical_features}
            # Fit the Explainer on the training data set using the LimeTabularExplainer
            explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, 
                                        feature_names = feature_names,
                                        mode = 'regression',
                                        random_state=42,
                                        verbose=True,
                                        categorical_features = categorical_features,
                                        categorical_names = categorical_names,
                                        discretize_continuous = False,
                                        )

            # model eval
            y_test_numpy = y_test.to_numpy().flatten()
            y_pred_eval = self.eval_model(y_pred, y_test_numpy)

            self.update_scores(y_pred_eval)
            self.update_lime(model, X_test,explainer,f,smiles_test)
            f+=1

        results = self.aggregate_scores()

        if len(self.save_dir) > 0:
            self.save_results(results, proper_model_name, model.get_params())

        return results, self.scores, self.lime_values
    
    def predict_capacity(self, X_input):
        """
        Predict capacity using the trained model.
        :param X_input: input data for prediction.
        :return: predicted values.
        """        
        prediction = self.model.predict(X_input)
        return prediction