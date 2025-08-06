from typing import Any, Iterable

import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.xai_pipelines.base import BaseXAIPipeline


class LimePipeline(BaseXAIPipeline):
    """
    Cross-validation XAI pipeline using LIME.
    """

    def init_explainer(self, **kwargs) -> object:
        """
        Initialize the LIME explainer.
        :return: LIME explainer object.
        """
        X_train = kwargs['X_train']
        feature_names = list(X_train.columns)
        categorical_features = list(range(len(feature_names)))
        return LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            mode=kwargs['mode'],
            random_state=kwargs['random_state'],
            verbose=kwargs['verbose'],
            categorical_features=categorical_features,
            discretize_continuous=kwargs['discretize_continuous'],
        )

    def init_values(self):
        """
        Initialize scores dictionary and values list.
        """
        self.values = {
            'lime_values': [],
            'lime_explanations': [],
            'smiles': []
        }

    def explain_model(self, model: object, X_test: pd.DataFrame, explainer: Any, smiles_list: pd.Series):
        """
        Explain the model using LIME.
        :param model: trained model to explain.
        :param X_test: test dataset.
        :param explainer: LIME explainer object.
        :param smiles_list: Series with SMILES strings corresponding to the test dataset.
        """
        lime_values = []
        lime_explanations = []

        for idx, (instance, smiles) in enumerate(zip(X_test.values, smiles_list)):
            print(f"LIME: Processing molecule {idx}, SMILES: {smiles}, len: {len(instance)}")
            lime_explanation = explainer.explain_instance(instance, model.predict, num_features=len(instance))
            lime_value = lime_explanation.as_list()
            lime_values.append(lime_value)
            lime_explanation = {
                'intercept': lime_explanation.intercept[0],
                'prediction_local': lime_explanation.local_pred,
                'right': lime_explanation.predicted_value,
                'score': lime_explanation.score,
                'local_exp': lime_explanation.local_exp[0],
            }
            lime_explanations.append(lime_explanation)

        self.values['lime_values'].append(lime_values)
        self.values['lime_explanations'].append(lime_explanations)
        self.values['smiles'].append(smiles_list)
