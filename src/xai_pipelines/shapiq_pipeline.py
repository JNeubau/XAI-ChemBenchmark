from typing import Any

import numpy as np
import pandas as pd
import shapiq

from src.xai_pipelines.base import BaseXAIPipeline


class ShapiqPipeline(BaseXAIPipeline):
    """SHAPiq cross-validation XAI pipeline."""

    def init_explainer(self, **kwargs) -> object:
        """
        Initialize the SHAPiq explainer.
        :param kwargs: additional parameters for the explainer.
        :return: SHAPiq explainer object.
        """
        model = kwargs['model']
        max_order = kwargs['max_order']
        shapiq_type = kwargs['shapiq_type']
        index_type = 'SV' if max_order == 1 else 'k-SII'
        if shapiq_type == 'tree':
            return shapiq.TreeExplainer(model, index=index_type, max_order=max_order), None
        else:
            return shapiq.TabularExplainer(model.predict, index=index_type, max_order=max_order, data=kwargs['X_train'].to_numpy()), 2 * len(kwargs['X_train'].columns) + 2048

    def explain_model(self, model: object, X_test: pd.DataFrame, explainer: Any, smiles_list: pd.Series):
        explainer, budget = explainer
        shap_values = []
        interaction_values = []
        new_X_test = np.array(X_test)
        for i in range(len(new_X_test)):
            if budget is None:
                shap_value = explainer.explain(new_X_test[i])
            else:
                shap_value = explainer.explain(new_X_test[i], budget=budget, random_state=42)
            if explainer.max_order == 1:
                vals = shap_value.get_n_order_values(1)
                shap_values.append(vals)
                interaction_values.append(shap_value.to_dict())
            else:
                interaction_values.append(shap_value.to_dict())
        self.values['shap_values'].append(shap_values)
        self.values['interactions'].append(interaction_values)
        self.values['features'].append(X_test)
        self.values['smiles'].append(smiles_list)

    def init_values(self):
        self.values = {
            'shap_values': [],
            'interactions': [],
            'features': [],
            'smiles': []
        }
