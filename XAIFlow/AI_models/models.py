import pickle
from typing import Callable
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class Models:
    """
    Class for ML models.
    """

    _models: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Register an ML model with a given name.

        This method is used as a decorator to register an ML model
        under a specified name. The registered function can later be retrieved
        and used to retrieve and train the model.

        :param name: The name to register the ML model under.
        :return: A decorator that registers the ML model.
        """

        def decorator(func: Callable) -> Callable:
            cls._models[name] = func
            return func

        return decorator

    def get_model(self, name: str, model_path: str | None = None) -> tuple:
        """
        Loads a selected model.
        :param name: name of the model, registered in the class.
        :param model_path: path to the saved model, if necessary (if this path is given, only a trained model and a name of the model will be returned).
        :return: "proper" model name, model and a parameter grid
        """
        if name in self._models:
            if model_path is not None:
                model = get_trained_model(model_path)
                _, _, dict_name = self._models[name]()
                return dict_name, model, None
            model, params, dict_name = self._models[name]()
            return dict_name, model, params
        raise ValueError(f"Model '{name}' is not defined.")


@Models.register("XGBReg")
def _get_xgboost() -> tuple:
    """
    :return: xgboost model, its parameter grid and name
    """
    xgb = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        n_estimators=25,
        learning_rate=0.1,
        max_depth=10,
        min_child_weight=1,
        gamma=0.2,
        colsample_bytree=0.3,
    )
    xgb_params = _get_xgb_params()
    return xgb, xgb_params, "XGBoost Regressor"


@Models.register("RFReg")
def _get_rf() -> tuple:
    """
    :return: random forest model, its parameter grid and name
    """
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        max_depth=10,
        n_estimators=25,
        min_samples_leaf=1,
        min_samples_split=2
    )
    rf_params = _get_rf_params()
    return rf, rf_params, "Random_Forest_Regressor"


def get_trained_model(model_path: str) -> object:
    """
    Loads trained model saved as a pickle file.
    :param model_path: path to the saved model (pickle format)
    :return: trained model
    """
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


def _get_xgb_params():
    return {
        "n_estimators": [10, 15, 25, 40, 50, 75, 100],
        "learning_rate": [0.05, 0.10, 0.15],
        "max_depth": [3, 5, 8, 10, None],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.2],
        "colsample_bytree": [0.1, 0.2, 0.3, 0.4],
    }


def _get_rf_params():
    return {
        "n_estimators": [10, 15, 25, 40, 50, 75, 100],
        "max_depth": [None, 3, 5, 8, 10],
        "min_samples_split": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }