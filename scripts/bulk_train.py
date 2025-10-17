import os
import pickle

import pandas as pd
import yaml

from src.predictive_model.training import PredictiveModelTrainingPipeline
from src.predictive_model.utils import custom_data_kfold


def train_model(train_config: dict):
    data = pd.read_csv(train_config['data_path'])
    X = data.drop(columns=['smiles', train_config['target_column']])
    y = data[[train_config['target_column']]]
    folds = custom_data_kfold(X, y, num_splits=train_config['num_splits'], num_bins=train_config['num_bins'], random_state=42)

    metrics = train_config['metrics']
    train_model_pipeline = PredictiveModelTrainingPipeline(
        X=X,
        y=y,
        folds=folds,
        num_bins=train_config['num_bins'],
        metrics=metrics,
        save_dir=train_config['save_dir'],
        data_name=train_config['data_name'],
        hyperparam_opt=True,
        verbose=True,
    )
    results, scores, f_imp = train_model_pipeline.train_pipeline(train_config['model_type'])
    print("Training completed successfully.")
    print(f"Results: {results}")
    print(f"Scores: {scores}")
    with open(os.path.join(train_config['save_dir'], f'results.pickle'), 'wb') as f:
        pickle.dump({
            'results': results,
            'scores': scores,
            'feature_importance': f_imp
        }, f)
    print('=' * 200)

if __name__ == '__main__':
    with open('../config/training.yaml', 'r') as f:
        config = yaml.safe_load(f)

    results_dir = config['output_path']
    real_results_dir = os.path.join(results_dir, 'real_data')
    herg_results_dir = os.path.join(results_dir, 'herg_data')
    synthetic_results_dir = os.path.join(results_dir, 'synthetic_data')

    real = config['real_dataset']
    synthetic = config['synthetic_dataset']
    herg = config['herg_dataset']

    train = config['train']

    real_datasets = os.listdir(real['data_path'])
    herg_datasets = os.listdir(herg['data_path'])
    synthetic_datasets = os.listdir(synthetic['data_path'])

    for dataset in herg_datasets:
        dataset_name = dataset.split('.')[0]
        dataset_train = {
            'data_path': os.path.join(herg['data_path'], dataset),
            'target_column': herg['target_column'],
            'num_splits': herg['num_splits'],
            'num_bins': herg['num_bins'],
            'metrics': train['metrics'],
            'save_dir': os.path.join(herg_results_dir, dataset_name),
            'data_name': dataset_name,
            'model_type': train['model_type']
        }
        os.makedirs(dataset_train['save_dir'], exist_ok=True)
        print(f"Training model on real dataset: {dataset_name}")
        train_model(dataset_train)

    for dataset in real_datasets:
        dataset_name = dataset.split('.')[0]
        dataset_train = {
            'data_path': os.path.join(real['data_path'], dataset),
            'target_column': real['target_column'],
            'num_splits': real['num_splits'],
            'num_bins': real['num_bins'],
            'metrics': train['metrics'],
            'save_dir': os.path.join(real_results_dir, dataset_name),
            'data_name': dataset_name,
            'model_type': train['model_type']
        }
        os.makedirs(dataset_train['save_dir'], exist_ok=True)
        print(f"Training model on real dataset: {dataset_name}")
        train_model(dataset_train)

    for dataset in synthetic_datasets:
        dataset_name = dataset.split('.')[0]
        dataset_train = {
            'data_path': os.path.join(synthetic['data_path'], dataset),
            'target_column': synthetic['target_column'],
            'num_splits': synthetic['num_splits'],
            'num_bins': synthetic['num_bins'],
            'metrics': train['metrics'],
            'save_dir': os.path.join(synthetic_results_dir, dataset_name),
            'data_name': dataset_name,
            'model_type': train['model_type']
        }
        os.makedirs(dataset_train['save_dir'], exist_ok=True)
        print(f"Training model on synthetic dataset: {dataset_name}")
        train_model(dataset_train)