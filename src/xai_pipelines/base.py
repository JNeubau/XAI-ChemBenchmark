from abc import abstractmethod, ABC

import copy
from typing import Iterable, Any
import pandas as pd
import os
import joblib


class BaseXAIPipeline(ABC):
    """
    Base cross-validation XAI pipeline.
    """

    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            z: pd.Series,  # SMILES data
            folds: list,
    ):
        """
        Initialize the cross-validation pipeline.
        :param X: dataframe with features.
        :param y: dataframe with target variable.
        :param z: Series with SMILES strings.
        :param folds: list with cross-validation folds.
        """
        self.X = X
        self.y = y
        self.z = z  # Store SMILES data
        self.folds = folds
        self.values = None
        self.filter_columns = X.columns.tolist()

    @abstractmethod
    def init_explainer(self, **kwargs) -> object:
        """
        Initialize the explainer.
        :return: explainer object.
        """
        pass

    @abstractmethod
    def explain_model(self, model: object, X_test: pd.DataFrame, explainer: Any, smiles_list: pd.Series,):
        pass

    @abstractmethod
    def init_values(self):
        """
        Initialize scores dictionary and values list.
        """
        pass

    def xai_pipeline(self, model_path: str | None = None, **kwargs) -> tuple:
        """
        Explain the model.
        :param model_path: path to saved model.
        :return: tuple with results, scores, explanations.
        """
        self.init_values()
        self.values['training_data'] = []
        self.values['test_data'] = []
        for i, fold in enumerate(self.folds):
            train_idx, test_idx = fold

            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            X_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)

            smiles_test = copy.deepcopy(self.z.loc[test_idx]).reset_index(drop=True)
            model = self.load_model(os.path.join(model_path, f"model_{i}.joblib"))

            kwargs['model'] = model
            kwargs['X_train'] = X_train
            kwargs['y_train'] = y_train
            kwargs['sample'] = i

            self.explain_model(model, X_test, self.init_explainer(**kwargs), smiles_test)
            self.values['training_data'].append(pd.concat([X_train, y_train], axis=1))
            self.values['test_data'].append(pd.concat([X_test, y_test], axis=1))

        return self.values

    @staticmethod
    def load_model(model_path: str) -> object:
        """
        Load a trained model from a file.
        :param model_path: path to the saved model.
        :return: loaded model.
        """
        if os.path.exists(model_path):
            loaded_model = joblib.load(model_path)
            return loaded_model
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
