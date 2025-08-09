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
    data_path = '../data/data_qm9_ecfp.csv'
    data = pd.read_csv(data_path)
    X = data.drop(columns=['smiles', 'target'])
    y = data[['target']]
    z = data['smiles']
    folds = custom_data_kfold(X, y, num_splits=10, num_bins=5, random_state=42)

    metrics = ['smape', 'mape', 'rmse', 'pairwise_accuracy_score']
    train_model_pipeline = PredictiveModelTrainingPipeline(
        X=X,
        y=y,
        folds=folds,
        metrics=metrics,
        save_dir='../results',
        data_name='redox',
        hyperparam_opt=True,
        verbose=True,
    )
    results, scores, f_imp = train_model_pipeline.train_pipeline("RFReg")
    print("Training completed successfully.")
    print(f"Results: {results}")
    print(f"Scores: {scores}")
    print(f"Feature Importance: {f_imp}")
    # with open('../config/lime.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # lime_piepline = LimePipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    # )
    # results = lime_piepline.xai_pipeline(model_path='../results/', **config)
    # print(results)
    # with open('../config/shap.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # shap_piepline = ShapPipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    # )
    # results = shap_piepline.xai_pipeline(model_path='../results/')
    # print(results)
    # with open('../config/shapiq.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # shapiq_piepline = ShapiqPipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    # )
    # results = shapiq_piepline.xai_pipeline(model_path='../results/', **config)
    # print(results)
    # with open('../config/mmace.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # shapiq_piepline = MMacePipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    #     fingerprint_type=config['fingerprint_type'],
    #     num_samples=config['num_samples'],
    #     alphabet=config.get('alphabet', None),
    #     num_mutations=config['num_mutations'],
    #     delta=config['delta'],
    #     nmols=config['nmols'],
    # )
    # results = shapiq_piepline.xai_pipeline(model_path='../results/', **config)
    # print(results)

    # with open('../config/meg.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # meg_pipeline = MegPipeline(
    #     X=X,
    #     y=y,
    #     z=z,  # Pass the SMILES data
    #     folds=folds,
    #     fingerprint_type=config['fingerprint_type'],
    #     delta=config['delta'],
    #     samples=config['samples'],
    #     epochs=config['epochs']
    # )
    # results = meg_pipeline.xai_pipeline(model_path='../results/', **config)
    # print(results)
