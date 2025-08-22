import warnings

from sklearn.metrics import root_mean_squared_error
import copy
import numpy as np
import pandas as pd
import joblib
import os
import optuna

from src.predictive_model.evaluation import EvalMetrics
from src.predictive_model.factory import Models
from src.predictive_model.utils import custom_data_split

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")


class PredictiveModelTrainingPipeline:
    """
    Cross-validation pipeline class.
    """

    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            folds: list,
            metrics: list,
            save_dir: str,
            data_name: str,
            num_bins: int,
            hyperparam_opt: bool = True,
            verbose: bool = False,
    ):
        """
        Initialize the cross-validation pipeline.
        :param X: dataframe with features.
        :param y: dataframe with target variable.
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
        self.num_bins = num_bins
        self.scores = None
        self.feature_importance = None

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
            folds = custom_data_split(X_train, y_train, num_bins=self.num_bins, train_size=0.7)
            x_train_opt, y_train_opt = X_train.iloc[folds[0][0], :], y_train.iloc[folds[0][0], :]
            x_test_opt, y_test_opt = X_train.iloc[folds[0][1], :], y_train.iloc[folds[0][1], :]

            def objective(trial):
                params = {}
                for key, value in param_grid.items():
                    if isinstance(value, list):
                        params[key] = trial.suggest_categorical(key, value)
                    elif isinstance(value, tuple):
                        if isinstance(value[0], int):
                            params[key] = trial.suggest_int(key, value[0], value[1])
                        else:
                            params[key] = trial.suggest_float(key, value[0], value[1])
                    else:
                        raise ValueError(f"Unsupported parameter type: {type(value)}")

                model.set_params(**params)
                model.fit(x_train_opt.to_numpy(), y_train_opt[y_train.columns[0]].to_numpy())
                y_pred = model.predict(x_test_opt.to_numpy()).flatten()
                score = root_mean_squared_error(y_test_opt.to_numpy(), y_pred)
                return score

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.QMCSampler(seed=42))

            study.optimize(objective, n_trials=100, show_progress_bar=False)
            best_params = study.best_params
            best_score = study.best_value
            model.set_params(**best_params)
            if self.verbose:
                print(f"Best score: {best_score}\n Best params: {best_params}")
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

    def init_scores(self):
        """
        Initialize scores dictionary and shap.
        """
        scores = {}
        for metric in self.metrics:
            scores[metric] = []
            scores[f'baseline_{metric}'] = []
        self.scores = scores
        self.feature_importance = []

    def update_scores(self, model_scores: dict, baselines: dict, f_importance: pd.DataFrame | None = None):
        """
        Update scores dictionary.
        :param model_scores: dictionary with model scores.
        :param baselines: dictionary with baseline scores.
        :param f_importance: feature importance dataframe.
        """
        for metric in self.metrics:
            self.scores[metric].append(model_scores[metric])
            self.scores[f'baseline_{metric}'].append(baselines[f'{metric}'])
        if f_importance is not None:
            self.feature_importance.append(f_importance)
        else:
            self.feature_importance.append(pd.DataFrame({'importance': [], 'feature': []}))

    def aggregate_scores(self):
        """
        Aggregate scores.
        :return: aggregated scores.
        """
        results = {}
        for metric in self.metrics:
            metric_scores = self.scores[metric]
            results[metric] = round(sum(metric_scores) / len(metric_scores), 4)
            results[f'baseline_{metric}'] = round(sum(self.scores[f'baseline_{metric}']) / len(self.scores[f'baseline_{metric}']), 4)
        return results

    def save_model(self, model: object, fold_num: int):
        """
        Save the trained model to a file.
        :param model: trained model.
        :param fold_num: fold number for saving.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        save_model_path = os.path.join(self.save_dir, f"model_{fold_num}.joblib")
        joblib.dump(model, save_model_path)
        if self.verbose:
            print(f"Model saved to {save_model_path}")

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> tuple:
        """
        Train the model.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: tuple with results, scores, explanations.
        """
        proper_model_name, model, param_grid, f_importance = Models().get_model(model_name, model_path=model_path)
        self.init_scores()

        if self.verbose:
            print(f"Training model {proper_model_name}")

        for i, fold in enumerate(self.folds):
            train_idx, test_idx = fold

            # train-test split
            X_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)

            model = self.tune_model(X_train, y_train, model, param_grid)

            # model training
            model.fit(X_train.to_numpy(), y_train[y_train.columns[0]].to_numpy())

            y_pred = model.predict(X_test.to_numpy()).flatten()

            f_imp = f_importance(model)
            f_imp = pd.DataFrame({'importance': f_imp, 'feature': list(X_train.columns)})

            # model eval
            y_test_numpy = y_test.to_numpy().flatten()
            y_pred_eval = self.eval_model(y_pred, y_test_numpy)

            baseline = np.median(y_train[y_train.columns[0]].to_numpy()) * np.ones_like(y_test_numpy)
            baselines = self.eval_model(baseline, y_test_numpy)

            self.update_scores(y_pred_eval, baselines, f_imp)
            if len(self.save_dir) > 0:
                self.save_model(model, fold_num=i)

        results = self.aggregate_scores()
        return results, self.scores, self.feature_importance
