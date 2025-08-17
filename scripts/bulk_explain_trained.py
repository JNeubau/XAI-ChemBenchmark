import os
import pickle

import pandas as pd
import yaml

from src.predictive_model.utils import custom_data_kfold
from src.xai_pipelines.lime_pipeline import LimePipeline
from src.xai_pipelines.meg_pipeline import MegPipeline
from src.xai_pipelines.mmace_pipeline import MMacePipeline
from src.xai_pipelines.shap_pipeline import ShapPipeline
from src.xai_pipelines.shapiq_pipeline import ShapiqPipeline


def save_results(results, save_path, method_name):
    """
    Save the results to a CSV file.
    """
    with open(os.path.join(save_path, f'{method_name}_results.pickle'), 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {os.path.join(save_path, f'{method_name}_results.pickle')}")

if __name__ == "__main__":
    explaining_config_path = '../config/real_data/explaining.yaml'

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

    configs_path = config['configs_path']

    # #LIME
    # with open(os.path.join(configs_path, 'lime.yaml'), 'r') as f:
    #     lime_config = yaml.safe_load(f)
    # lime_piepline = LimePipeline(
    #     X=X,
    #     y=y,
    #     z=z,
    #     folds=folds,
    # )
    # results_lime = lime_piepline.xai_pipeline(model_path=model_path, **lime_config)
    # save_results(results_lime, save_path, 'lime')
    #
    # #SHAP
    # with open(os.path.join(configs_path, 'shap.yaml'), 'r') as f:
    #     shap_config = yaml.safe_load(f)
    # shap_piepline = ShapPipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    # )
    # results_shap = shap_piepline.xai_pipeline(model_path=model_path)
    # save_results(results_shap, save_path, 'shap')
    #
    # #SHAPIQ-1
    # with open(os.path.join(configs_path, 'shapiq1.yaml'), 'r') as f:
    #     shapiq1_config = yaml.safe_load(f)
    # shapiq1_piepline = ShapiqPipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    # )
    # results_shapiq1 = shapiq1_piepline.xai_pipeline(model_path=model_path, **shapiq1_config)
    # save_results(results_shapiq1, save_path, 'shapiq1')
    #
    # #SHAPIQ-2
    # with open(os.path.join(configs_path, 'shapiq2.yaml'), 'r') as f:
    #     shapiq2_config = yaml.safe_load(f)
    # shapiq2_piepline = ShapiqPipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    # )
    # results_shapiq2 = shapiq2_piepline.xai_pipeline(model_path=model_path, **shapiq2_config)
    # save_results(results_shapiq2, save_path, 'shapiq2')
    #
    # #MMACE
    # with open(os.path.join(configs_path, 'mmace.yaml'), 'r') as f:
    #     mmace_config = yaml.safe_load(f)
    # mmace_piepline = MMacePipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    #     fingerprint_type=mmace_config['fingerprint_type'],
    #     fp_params=mmace_config['fp_params'],
    #     num_samples=mmace_config['num_samples'],
    #     alphabet=mmace_config.get('alphabet', None),
    #     num_mutations=mmace_config['num_mutations'],
    #     delta=mmace_config['delta'],
    #     nmols=mmace_config['nmols'],
    # )
    # results_mmace = mmace_piepline.xai_pipeline(model_path=model_path, **mmace_config)
    # save_results(results_mmace, save_path, 'mmace')

    #MEG
    with open(os.path.join(configs_path, 'meg.yaml'), 'r') as f:
        meg_config = yaml.safe_load(f)
    meg_pipeline = MegPipeline(
        X=X,
        y=y,
        z=z,  # Pass the SMILES data
        folds=folds,
        fingerprint_type=meg_config['fingerprint_type'],
        fp_params=meg_config['fp_params'],
        delta=meg_config['delta'],
        samples=meg_config['samples'],
        epochs=meg_config['epochs']
    )
    results_meg = meg_pipeline.xai_pipeline(model_path=model_path, **meg_config)
    save_results(results_meg, save_path, 'meg1')



