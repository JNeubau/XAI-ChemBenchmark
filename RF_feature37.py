import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance, plot_partial_dependence
# from sklearn.externals import joblib

import matplotlib.pyplot as plt
import os
import glob
import pickle

def load_model(model_path, verbose=False):
    if os.path.exists(model_path):
        print("Loading model from {}".format(model_path))
        # with open(model_path, "rb") as f:
        #     first_bytes = f.read(20)
        #     print(first_bytes)
        #     loaded_model = pickle.load(f)
        loaded_model = joblib.load(model_path) # TODO fix
        # with open(model_path, "rb") as f:
        #     loaded_model = pickle.load(f)
        if verbose:
            print("Model loaded from {}".format(model_path))
        return loaded_model
    else:
        raise FileNotFoundError("Model file not found at {}".format(model_path))

def load_data(data_path, verbose=False):
    print("Loading data from {}".format(data_path))
    data = pd.read_csv(data_path)
    X = data.drop(columns=['capacity_max', 'smiles'])
    y = data[['capacity_max']]
    if verbose:
        print("Loaded data from {}, shape: {}".format(data_path, data.shape))
        print("X shape: {}, y shape: {}".format(X.shape, y.shape))
    return X, y

def plot_feature_importances(importances, feature_names=None, save_path="feature_importances.png", top_n=10):
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.title("Top {} Mean Feature Importances (Aggregated)".format(top_n))
    if feature_names is not None:
        names = np.array(feature_names)[indices]
        plt.bar(range(top_n), importances[indices], align="center")
        plt.xticks(range(top_n), names, rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def aggregate_models(models_folder, X, y, feature_names=None, verbose=False):
    model_files = sorted(glob.glob(os.path.join(models_folder, "model_*.joblib")))
    model_files_other = [f for f in model_files if "_37.joblib" in f]
    XXmodel_files_other = [f for f in model_files if "_37.joblib" not in f]
    # print("Models with '_37': {}".format(model_files_37))
    print("Other models: {}".format(model_files_other))
    n_models = len(model_files_other)
    if n_models == 0:
        raise ValueError("No model files found in the specified folder.")
    print("Found {} model files.".format(n_models))

    metrics = []
    all_importances = []
    all_perm_importances = []

    for model_file in model_files_other:
        print("Processing model file: {}".format(model_file))
        model = load_model(model_file, verbose=verbose)
        y_pred = model.predict(X)
        rmse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        metrics.append([rmse, mae, r2])
        all_importances.append(model.feature_importances_)

        perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        all_perm_importances.append(perm.importances_mean)

    metrics = np.array(metrics)
    mean_metrics = metrics.mean(axis=0)
    std_metrics = metrics.std(axis=0)
    mean_importances = np.mean(all_importances, axis=0)
    mean_perm_importances = np.mean(all_perm_importances, axis=0)

    print("Aggregated Regression Metrics (mean ± std):")
    print("RMSE: {:.4f} ± {:.4f}".format(mean_metrics[0], std_metrics[0]))
    print("MAE: {:.4f} ± {:.4f}".format(mean_metrics[1], std_metrics[1]))
    print("R2: {:.4f} ± {:.4f}".format(mean_metrics[2], std_metrics[2]))

    print("Aggregated Feature Importances (mean):")
    sorted_idx = np.argsort(mean_importances)[::-1]
    for idx in sorted_idx[:10]:
        fname = feature_names[idx] if feature_names else str(idx)
        print("{}: {:.4f}".format(fname, mean_importances[idx]))

    plot_feature_importances(mean_importances, feature_names)

    print("Aggregated Permutation Importances (mean):")
    for idx in sorted_idx[:10]:
        fname = feature_names[idx] if feature_names else str(idx)
        print("{}: {:.4f}".format(fname, mean_perm_importances[idx]))

    # Partial Dependence Plot for top 2 features
    top_features = sorted_idx[:2]
    model = load_model(model_files_other[0])  # Use the first model for PDP
    # Use plot_partial_dependence for compatibility with older scikit-learn
    plot_partial_dependence(
        model, X, features=top_features, feature_names=feature_names
    )
    plt.tight_layout()
    plt.savefig("partial_dependence_aggregated.png")
    plt.close()

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)
    models_folder = 'path' #do zmiany
    X_path = os.path.join(parent_dir, 'XAI-experiments', 'data', 'new_maccs_merged_all.csv')
    X, y = load_data(X_path)
    feature_names = ["maccsfingerprint{}".format(i) for i in range(167)]
    aggregate_models(models_folder, X, y, feature_names=feature_names)