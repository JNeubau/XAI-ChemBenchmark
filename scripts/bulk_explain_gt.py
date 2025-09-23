import copy
import os
import sys
import pickle

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.preprocessing import StandardScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(1, project_root)
    
from src.predictive_model.utils import custom_data_kfold
from src.xai_pipelines.lime_pipeline import LimePipeline
from src.xai_pipelines.meg_pipeline import MegPipeline
from src.xai_pipelines.mmace_pipeline import MMacePipeline
from src.xai_pipelines.shap_pipeline import ShapPipeline
from src.xai_pipelines.shapiq_pipeline import ShapiqPipeline

class GtModel:
    def __init__(self, dataset_name, feature_order, X_train):
        self.dataset_name = dataset_name
        self.feature_order = feature_order
        self.selected_features_names = [30, 123, 10, 16, 81, 33]
        self.selected_features_positions = [self.feature_order[f'ecfp_feature_{i}'] for i in self.selected_features_names]
        self.choices = {
            'qm9_simple_linear6': self.linear_function,
            'qm9_piecewise_linear_6': self.piecewise_linear_function,
            'qm9_nonlinear_6': self.nonlinear_function,
        }
        self.scaler = StandardScaler().fit(X_train.to_numpy())
        self.X_train = X_train

    def linear_function(self, df, df_not_scaled):
        f_30 = df[:, self.selected_features_positions[0]]
        f_123 = df[:, self.selected_features_positions[1]]
        f_10 = df[:, self.selected_features_positions[2]]
        f_16 = df[:, self.selected_features_positions[3]]
        f_81 = df[:, self.selected_features_positions[4]]
        f_33 = df[:, self.selected_features_positions[5]]
        target = 8.5 * f_30 + 10.5 * f_123 - 3.5 * f_10 + 3 * f_16 - 2.5 * f_81 + 5.5 * f_33 + 30
        return target

    def piecewise_linear_function(self, df, df_not_scaled):
        f_30 = df[:, self.selected_features_positions[0]]
        f_123 = df[:, self.selected_features_positions[1]]
        f_16 = df[:, self.selected_features_positions[3]]
        f_81 = df[:, self.selected_features_positions[4]]
        f_33 = df[:, self.selected_features_positions[5]]

        f_10 = df_not_scaled[:, self.selected_features_positions[2]]
        f_10 = f_10.round(0)

        conditions = [
            f_10 < 1,
            (f_10 >= 1) & (f_10 < 2),
            f_10 >= 2
        ]

        choices = [
            10.5 * f_30 + 6.5 * f_123 - 1.5 * f_81 + 30,
            5 * f_30 + 13 * f_123 - 2.5 * f_16 + 30,
            -1.5 * f_30 + 3.5 * f_123 + 15.5 * f_33 + 30
        ]
        target = np.select(conditions, choices, default=10000)
        return target

    def nonlinear_function(self, df, df_not_scaled):
        f_30 = df[:, self.selected_features_positions[0]]
        f_123 = df[:, self.selected_features_positions[1]]
        f_10 = df[:, self.selected_features_positions[2]]
        f_16 = df[:, self.selected_features_positions[3]]
        f_81 = df[:, self.selected_features_positions[4]]
        f_33 = df[:, self.selected_features_positions[5]]

        component1 = -9.5 * f_10 + 2.5 * f_81 ** 2 - 3.5 * f_16
        component2 = 7.5 * f_30 * f_123
        component3 = 1.5 * f_33 ** 2 + 30
        return component1 + component2 + component3

    def predict(self, df):
        df_scaled = self.scaler.transform(df)
        if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
            df = df.to_numpy()
            df_scaled = df_scaled.to_numpy()
        return self.choices[self.dataset_name](df_scaled, df)


def save_results(results, save_path, method_name):
    """
    Save the results to a CSV file.
    """
    with open(os.path.join(save_path, f'{method_name}_results.pickle'), 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {os.path.join(save_path, f'{method_name}_results.pickle')}")

if __name__ == "__main__":
    explaining_config_path = '../config/gt_synthetic_data/explaining.yaml'

    with open(explaining_config_path, 'r') as f:
        config = yaml.safe_load(f)

    target_column = config['target_column']
    dataset_name = config['dataset_name']

    save_path = os.path.join(config['save_path'], dataset_name, 'explanations')
    os.makedirs(save_path, exist_ok=True)

    model_path = os.path.join(config['models_path'], dataset_name)

    data = pd.read_csv(os.path.join(config['data_path'], f'{dataset_name}.csv'))
    X = data.drop(columns=['smiles', target_column])
    y = data[[target_column]]
    z = data['smiles']
    folds = custom_data_kfold(X, y, num_splits=config['num_splits'], num_bins=config['num_bins'], random_state=42)

    for i, fold in enumerate(folds):
        train_idx, test_idx = fold

        X_test = copy.deepcopy(X.loc[test_idx, :]).reset_index(drop=True)
        X_train = copy.deepcopy(X.loc[train_idx, :]).reset_index(drop=True)
        y_test = copy.deepcopy(y.loc[test_idx, :]).reset_index(drop=True)
        y_train = copy.deepcopy(y.loc[train_idx, :]).reset_index(drop=True)


        feature_order = {f: i for i, f in enumerate(X_train.columns)}
        model = GtModel(dataset_name, feature_order, X_train)

        os.makedirs(model_path, exist_ok=True)
        save_model_path = os.path.join(model_path, f"model_{i}.joblib")
        joblib.dump(model, save_model_path)

    configs_path = config['configs_path']

    # region LIME
    with open(os.path.join(configs_path, 'lime.yaml'), 'r') as f:
        lime_config = yaml.safe_load(f)
    lime_piepline = LimePipeline(
        X=X,
        y=y,
        z=z,
        folds=folds,
    )
    results_lime = lime_piepline.xai_pipeline(model_path=model_path, **lime_config)
    save_results(results_lime, save_path, 'lime')
    
    # region SHAP
    with open(os.path.join(configs_path, 'shap.yaml'), 'r') as f:
        shap_config = yaml.safe_load(f)
    shap_piepline = ShapPipeline(
        X=X,
        y=y,
        z=z,  # Pass the SMILES data
        folds=folds,
    )
    results_shap = shap_piepline.xai_pipeline(model_path=model_path, **shap_config)
    save_results(results_shap, save_path, 'shap')
    
    # region SHAPIQ-1
    with open(os.path.join(configs_path, 'shapiq1.yaml'), 'r') as f:
        shapiq1_config = yaml.safe_load(f)
    shapiq1_piepline = ShapiqPipeline(
        X=X,
        y=y,
        z=z,  # Pass the SMILES data
        folds=folds,
    )
    results_shapiq1 = shapiq1_piepline.xai_pipeline(model_path=model_path, **shapiq1_config)
    save_results(results_shapiq1, save_path, 'shapiq1')
    
    # region SHAPIQ-2
    with open(os.path.join(configs_path, 'shapiq2.yaml'), 'r') as f:
        shapiq2_config = yaml.safe_load(f)
    shapiq2_piepline = ShapiqPipeline(
        X=X,
        y=y,
        z=z,  # Pass the SMILES data
        folds=folds,
    )
    results_shapiq2 = shapiq2_piepline.xai_pipeline(model_path=model_path, **shapiq2_config)
    save_results(results_shapiq2, save_path, 'shapiq2')
    
    # region MMACE
    with open(os.path.join(configs_path, 'mmace.yaml'), 'r') as f:
        mmace_config = yaml.safe_load(f)
    mmace_piepline = MMacePipeline(
        X=X,
        y=y,
        z=z,  # Pass the SMILES data
        folds=folds,
        fingerprint_type=mmace_config['fingerprint_type'],
        fp_params=mmace_config['fp_params'],
        num_samples=mmace_config['num_samples'],
        alphabet=mmace_config.get('alphabet', None),
        num_mutations=mmace_config['num_mutations'],
        delta=mmace_config['delta'],
        nmols=mmace_config['nmols'],
    )
    results_mmace = mmace_piepline.xai_pipeline(model_path=model_path, **mmace_config)
    save_results(results_mmace, save_path, 'mmace')

    # region MEG
    # with open(os.path.join(configs_path, 'meg.yaml'), 'r') as f:
    #     meg_config = yaml.safe_load(f)
    # meg_pipeline = MegPipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    #     fingerprint_type=meg_config['fingerprint_type'],
    #     fp_params=meg_config['fp_params'],
    #     delta=meg_config['delta'],
    #     samples=meg_config['samples'],
    #     epochs=meg_config['epochs']
    # )
    # results_meg = meg_pipeline.xai_pipeline(model_path=model_path, **meg_config)
    # save_results(results_meg, save_path, 'meg2')



