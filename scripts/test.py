import os

import pandas as pd
import yaml

from src.predictive_model.training import PredictiveModelTrainingPipeline
from sklearn.model_selection import StratifiedKFold
from src.predictive_model.utils import custom_data_kfold
from src.xai_pipelines.lime_pipeline import LimePipeline
from src.xai_pipelines.meg_pipeline import MegPipeline
from src.xai_pipelines.mmace_pipeline import MMacePipeline
from src.xai_pipelines.shap_pipeline import ShapPipeline
from src.xai_pipelines.shapiq_pipeline import ShapiqPipeline

if __name__ == '__main__':
    data_path = '../data/cof_data/data_batteries_ecfp_descriptor.csv'
    data = pd.read_csv(data_path)
    X = data.drop(columns=['smiles', 'capacity_max'])
    y = data[['capacity_max']]
    z = data['smiles']
    folds = custom_data_kfold(X, y, num_splits=10, num_bins=10, random_state=42)
    model_path = '../results/real_data/data_batteries_ecfp_descriptor'
    #
    # metrics = ['smape', 'mape', 'rmse', 'pairwise_accuracy_score']
    # train_model_pipeline = PredictiveModelTrainingPipeline(
    #     X=X,
    #     y=y,
    #     folds=folds,
    #     metrics=metrics,
    #     save_dir='../results',
    #     data_name='redox',
    #     hyperparam_opt=True,
    #     verbose=True,
    # )
    # results, scores, f_imp = train_model_pipeline.train_pipeline("RFReg")
    # print("Training completed successfully.")
    # print(f"Results: {results}")
    # print(f"Scores: {scores}")
    # print(f"Feature Importance: {f_imp}")
    # with open('../config/lime.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # lime_piepline = LimePipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    # )
    # results = lime_piepline.xai_pipeline(model_path=model_path, **config)
    # print(results)
    # with open('../config/shap.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # shap_piepline = ShapPipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    # )
    # results = shap_piepline.xai_pipeline(model_path=model_path)
    # print(results)
    # with open('../config/shapiq1.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # shapiq_piepline = ShapiqPipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    # )
    # results = shapiq_piepline.xai_pipeline(model_path=model_path, **config)
    # print(results)
    # with open('../config/mmace.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # shapiq_piepline = MMacePipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    #     fingerprint_type=config['fingerprint_type'],
    #     fp_params=config['fp_params'],
    #     num_samples=config['num_samples'],
    #     alphabet=config.get('alphabet', None),
    #     num_mutations=config['num_mutations'],
    #     delta=config['delta'],
    #     nmols=config['nmols'],
    # )
    # results = shapiq_piepline.xai_pipeline(model_path=model_path, **config)
    # print(results['pred_original'])
    # print(results['pred_counterfactual'])
    # for c, t in zip(results['pred_counterfactual'], results['pred_original']):
    #     for i in range(len(c)):
    #         print(c[i], t[i])

    with open('../config/real_data/meg.yaml', 'r') as f:
        config = yaml.safe_load(f)
    meg_pipeline = MegPipeline(
        X=X,
        y=y,
        z=z,  # Pass the SMILES data
        folds=folds,
        fingerprint_type=config['fingerprint_type'],
        fp_params=config['fp_params'],
        delta=config['delta'],
        samples=config['samples'],
        epochs=config['epochs']
    )
    results = meg_pipeline.xai_pipeline(model_path=model_path, **config)
    print(results)

    for c, t, r in zip(results['pred_counterfactual'], results['pred_original'], results['counterfactuals_pred_reward']):
        for i in range(len(c)):
            print(t[i], c[i], r[i])
