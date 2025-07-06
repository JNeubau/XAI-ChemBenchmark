import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt
import os
import glob
import json
from XAIFlow.utils.exportlib import save_data_to_excel_with_highlights

def export_rfreg_feature_importances_to_excel(model_files, all_importances, feature_names, smarts_mapping_path, save_path="rfreg_feature_importances.xlsx"):
    # Load SMARTS mapping
    with open(smarts_mapping_path, 'r') as f:
        smarts_mapping = json.load(f)

    excel_data = {
        "Fold_No": [],
        "Smiles_key": [],
        "Feature_key": [],
        "SMARTS": [],
        "Molecule": [],
        "number_of_molecules_where_fingerprint": [],
        "Number_where_important": [],
        'feature_in_smiles': [],
        "Explanation_value": [],
        "Explanation_sign": [],
        "Capacity_Max": [],
        "Capacity_Pred": [],
        "Model": [],
        'Positive_explanation_add_count': [],
        'Negative_explanation_add_count': [],
        'Positive_explanation_del_count': [],
        'Negative_explanation_del_count': []
    }

    for model_idx, (model_file, importances) in enumerate(zip(model_files, all_importances)):
        top_indices = np.argsort(np.abs(importances))[::-1][:10]
        for feat_idx in top_indices:
            importance = importances[feat_idx]
            fname = feature_names[feat_idx] if feature_names else str(feat_idx)
            smarts = smarts_mapping.get(f'maccsfingerprint{int(fname.replace("maccsfingerprint", "")) - 1}', [""])[0]
            fold_no = os.path.splitext(os.path.basename(model_file))[0].split("_")[-1]
            excel_data["Fold_No"].append(fold_no)
            excel_data["Smiles_key"].append("")
            excel_data["Feature_key"].append(fname)
            excel_data["SMARTS"].append(smarts)
            excel_data["Molecule"].append("")
            excel_data["number_of_molecules_where_fingerprint"].append("")
            excel_data["Number_where_important"].append("")
            excel_data["feature_in_smiles"].append("")
            excel_data["Explanation_value"].append(importance)
            excel_data["Explanation_sign"].append("Positive" if importance >= 0 else "Negative")
            excel_data["Capacity_Max"].append("")
            excel_data["Capacity_Pred"].append("")
            excel_data["Model"].append("RFReg310")
            excel_data['Positive_explanation_add_count'].append("")
            excel_data['Negative_explanation_add_count'].append("")
            excel_data['Positive_explanation_del_count'].append("")
            excel_data['Negative_explanation_del_count'].append("")

    save_data_to_excel_with_highlights(excel_data, save_path)
    print(f"RFReg feature importances exported to {save_path}")


def export_top10_rfreg_importances(models_folder, feature_names, smarts_mapping_path, timestamp):
    model_files = sorted(glob.glob(os.path.join(models_folder, "model_*.joblib")))
    model_files_other = [f for f in model_files if "_37.joblib" not in f]

    all_importances = []
    for model_file in model_files_other:
        model = joblib.load(model_file)
        all_importances.append(model.feature_importances_)
        print(f"Loaded importances from {model_file}")
        print(f"Feature importances shape: {model.feature_importances_}")
    export_rfreg_feature_importances_to_excel(
        model_files_other, all_importances, feature_names, smarts_mapping_path,
        save_path=f"rfreg_feature_molecule_results_with_highlights_importances_{timestamp}.xlsx"
    )


def load_model(model_path, verbose=False):
    if os.path.exists(model_path):
        loaded_model = joblib.load(model_path)
        if verbose:
            print(f"Model loaded from {model_path}")
        return loaded_model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

def load_data(data_path, verbose=False):
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    X = data.drop(columns=['capacity_max', 'smiles'])
    y = data[['capacity_max']]
    if verbose:
        print(f"Loaded data from {data_path}, shape: {data.shape}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"First few rows of X:\n{X.head()}")
    print(f"First few rows of X:\n{X.head()}")
    return X, y

def plot_feature_importances(importances, feature_names=None, save_path="feature_importances.png", top_n=10):
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Mean Feature Importances")
    if feature_names is not None:
        names = np.array(feature_names)[indices]
        plt.bar(range(top_n), importances[indices], align="center")
        plt.xticks(range(top_n), names, rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def aggregate_models(models_folder, X, y, feature_names=None, verbose=False):
    model_files = sorted(glob.glob(os.path.join(models_folder, "model_*.joblib")))
    XXmodel_files_other = [f for f in model_files if "_37.joblib" in f]
    model_files_other = [f for f in model_files if "_37.joblib" not in f]
    # print(f"Models with '_37': {model_files_37}")
    print(f"Other models: {model_files_other}")
    n_models = len(model_files_other)
    if n_models == 0:
        raise ValueError("No model files found in the specified folder.")
    print(f"Found {n_models} model files.")

    metrics = []
    all_importances = []
    all_perm_importances = []

    for model_file in model_files_other:
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
    print(f"RMSE: {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"MAE: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"R2: {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")

    print("Aggregated Feature Importances (mean):")
    sorted_idx = np.argsort(mean_importances)[::-1]
    for idx in sorted_idx[:10]:
        fname = feature_names[idx] if feature_names else str(idx)
        print(f"{fname}: {mean_importances[idx]:.4f}")

    plot_feature_importances(mean_importances, feature_names)

if __name__ == "__main__":
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)
    models_folder = os.path.join(parent_dir, 'XAI-experiments', 'RFReg', 'final', 'ckpt')
    X_path = os.path.join(parent_dir, 'XAI-experiments', 'data', 'new_maccs_merged_all.csv')
    X, y = load_data(X_path)
    feature_names = [f"maccsfingerprint{i}" for i in range(1,167)]
    print(f"Feature names: {feature_names[:10]}... end {feature_names[-10:]}")  # Print first 10 feature names for brevity
    # aggregate_models(models_folder, X, y, feature_names=feature_names,verbose=True)

    smarts_mapping_path = os.path.join(parent_dir, 'XAI-experiments', 'data', 'maccs_smarts_mapping.json')
    
    export_top10_rfreg_importances(models_folder, feature_names, smarts_mapping_path, timestamp)