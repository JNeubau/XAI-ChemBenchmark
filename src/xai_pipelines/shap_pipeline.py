from typing import Any

import pandas as pd
import shap

from src.xai_pipelines.base import BaseXAIPipeline


class ShapPipeline(BaseXAIPipeline):
    """
    SHAP cross-validation pipeline.
    """

    def init_explainer(self, **kwargs) -> object:
        model = kwargs['model']
        return shap.TreeExplainer(model)

    def explain_model(self, model: object, X_test: pd.DataFrame, explainer: Any, smiles_list: pd.Series):
        expected_value = explainer.expected_value
        shap_values = explainer.shap_values(X_test)
        self.values['shap_values'].append(shap_values)
        self.values['expected_values'].append(expected_value)
        self.values['smiles'].append(smiles_list)

    def init_values(self):
        self.values = {
            'shap_values': [],
            'expected_values': [],
            'smiles': []
        }