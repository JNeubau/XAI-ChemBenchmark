import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
    
from AI_models.rfreg_cross_validation import CrossValidationRFRegPipeline
from utils.data_split import custom_data_kfold, save_fold_indices, load_fold_indices
from utils.exportlib import save_data_to_excel_with_highlights, save_scores_to_excel_new_sheet

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

from LIME.lime_tabular_explainer import CrossValidationLimePipeline

def mainXaiFlow(model, local_explanation=True, experiment_name='rf_test', folds=5, seed=42, train_rfreg=False, data_file=None):
    print("Model: ", model)
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)
    
    if data_file is None:
        maccs_fingerprints = os.path.join(parent_dir, 'data', 'new_maccs_merged.csv')
    else:
        maccs_fingerprints =  os.path.join(parent_dir, 'data', data_file)
    
    smarts_mapping_path = os.path.join(parent_dir, 'data', 'maccs_smarts_mapping.json')
    if local_explanation:
        explenation_type = 'local'    
    else:
        explenation_type = 'global'
    results_dir = os.path.join(parent_dir, 'results', experiment_name, model, explenation_type)
    folds_dir = os.path.join(parent_dir, 'RFReg', experiment_name, 'folds')
    
    data = pd.read_csv(maccs_fingerprints)
    print(data.head())
    os.makedirs(results_dir, exist_ok=True)

    # folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 5)

    # Train the model
    if train_rfreg:
        folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], num_splits=folds, random_state=seed)
        save_fold_indices(folds, folds_dir)
        train_RFReg(experiment_name, folds, data, parent_dir)
    else:
        folds = load_fold_indices(folds_dir)
        
    cv_pipeline = select_pipeline(model, data, folds)
    lime_values = cv_pipeline.load_pipeline(os.path.join(parent_dir, 'RFReg', experiment_name, 'ckpt'))

    smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds(
        folds, data, lime_values, smarts_mapping_path, local_explanation, cv_pipeline)
    molecules_statistics_all = count_molecules_with_fingerprint(data, molecules_statistics_all)
    molecules_statistics_all = count_important_features(data, molecules_statistics_all)

    #remove molecule_statistic_all row where Explanation_value is 0
    # molecules_statistics_all = {k: v for k, v in molecules_statistics_all.items() if v["lime_value"] != 0}
    
    excel_data = prepare_data_for_excel_export(match_molecules_all, smarts_top_all, molecules_statistics_all)
    save_molecules_to_excel(excel_data, results_dir)
    
    # scores_data = create_dataframe_from_scores(scores, results)
    # save_scores_to_excel(scores_data, results_dir)
    
    
def train_RFReg(experiment_name, folds, data, parent_dir):
    RFReg_cv_pipeline = select_pipeline('RFReg', data, folds, save_dir=os.path.join(parent_dir, 'RFReg', experiment_name, 'ckpt'))
    results, scores = RFReg_cv_pipeline.train_pipeline('RFReg')

    scores_data = create_dataframe_from_scores(scores, results)
    save_scores_to_excel(scores_data, os.path.join(parent_dir, 'RFReg', experiment_name, 'ckpt'))
    return RFReg_cv_pipeline


def create_dataframe_from_scores(scores, results):
    df_scores = pd.DataFrame(scores)
    for key, value in results.items():
        df_scores.loc["Final", key] = value
    return df_scores


def count_molecules_with_fingerprint(maccs_fingerprints_data, molecules_statistics_all):
    column_ones_count = maccs_fingerprints_data.sum(axis=0).to_dict()
    for key in molecules_statistics_all.keys():
        column_name = key[2]
        if column_name in column_ones_count:
            molecules_statistics_all[key]["number_of_molecules_where_fingerprint"] = column_ones_count[column_name]
    return molecules_statistics_all


def count_important_features(data, molecules_statistics_all):
    feature_importance_count = {col: 0 for col in data.columns}

    for key in molecules_statistics_all.keys():
        feature_key = key[2]
        if feature_key in feature_importance_count:
            feature_importance_count[feature_key] += 1

    for key in molecules_statistics_all.keys():
        feature_key = key[2]
        molecules_statistics_all[key]["number_where_important"] = feature_importance_count.get(feature_key, 0)
    return molecules_statistics_all


def predict_capacity(pipeline, data, molecules_statistics_all):
    for row in data.iterrows():
        molecules_statistics_all["capacity_pred"] = pipeline.predict_capacity(row.drop(columns=['capacity_max', 'smiles']).to_frame().T)
    return molecules_statistics_all


def select_pipeline(model, data, folds, save_dir=''):
    match model:
        case 'RFReg':
            return CrossValidationRFRegPipeline(
                X=data.drop(columns=['capacity_max', 'smiles']),
                y=data[['capacity_max']],
                folds=folds,
                metrics=['smape', 'pairwise_accuracy_score', 'rmse', 'ndcg_score'],
                save_dir=save_dir,
                data_name='battery',
                verbose=True
            )
        case 'LIME':
            return CrossValidationLimePipeline(
                X=data.drop(columns=['capacity_max', 'smiles']),
                y=data[['capacity_max']],
                z=data['smiles'],
                folds=folds,
                metrics=['smape', 'pairwise_accuracy_score', 'rmse', 'ndcg_score'],
                save_dir='',
                data_name='battery',
                verbose=True
            )
        case default:
            raise ValueError("Model not selected.")


def prepare_data_for_excel_export(match_molecules, smarts_top, molecules_statistics_all):
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
        "positive_changes": [],
        "negative_changes": [],
        "Positive_explanation_add_count": [],
        "Negative_explanation_add_count": [],
        "Positive_explanation_del_count": [],
        "Negative_explanation_del_count": []
    }
            
    for key, smarts in smarts_top.items():
        excel_data["Fold_No"].append(key[0])
        excel_data["Smiles_key"].append(key[1])
        excel_data["Feature_key"].append(f'maccsfingerprint{int(key[2].replace("maccsfingerprint", "")) + 1}')
        # excel_data["Feature_key"].append(key[2])
        excel_data["SMARTS"].append(smarts)
        excel_data["Molecule"].append(key[1])
        excel_data["number_of_molecules_where_fingerprint"].append(molecules_statistics_all[key]["number_of_molecules_where_fingerprint"])
        excel_data["Number_where_important"].append(molecules_statistics_all[key]["number_where_important"])
        excel_data["feature_in_smiles"].append(molecules_statistics_all[key]["feature_in_smiles"])
        excel_data["Explanation_value"].append(molecules_statistics_all[key]["lime_value"])
        excel_data["Explanation_sign"].append(molecules_statistics_all[key]["lime_sign"])
        excel_data["Capacity_Max"].append(molecules_statistics_all[key]["capacity_max"])
        excel_data["Capacity_Pred"].append(molecules_statistics_all[key]["capacity_pred"])
        excel_data["Model"].append("LIME")
        excel_data["positive_changes"].append(molecules_statistics_all[key]["positive_changes"])
        excel_data["negative_changes"].append(molecules_statistics_all[key]["negative_changes"])
        excel_data["Positive_explanation_add_count"].append(molecules_statistics_all[key]["Positive_explanation_add_count"])
        excel_data["Negative_explanation_add_count"].append(molecules_statistics_all[key]["Negative_explanation_add_count"])
        excel_data["Positive_explanation_del_count"].append(molecules_statistics_all[key]["Positive_explanation_del_count"])
        excel_data["Negative_explanation_del_count"].append(molecules_statistics_all[key]["Negative_explanation_del_count"])
    
    return excel_data


def save_molecules_to_excel(excel_data, results_dir):
    results_dir = results_dir + f'\\molecule_results_with_highlights_{datetime.now().strftime("%H-%M-%S")}.xlsx'
    save_data_to_excel_with_highlights(excel_data, results_dir,with_images=False)
    print(f"Molecule results with highlights saved to {results_dir}")


def save_scores_to_excel(scores_data, results_dir):
    results_dir = results_dir + f'\\molecule_scores_{datetime.now().strftime("%H-%M-%S")}.xlsx'
    save_scores_to_excel_new_sheet(scores_data, results_dir)
    print(f"Scores saved to {results_dir}")
    

def process_folds_local(folds, data, lime_values, smarts_mapping_path, top_i=5):
    smarts_top_all = {}
    match_molecules_all = {}
    molecules_statistics_all = {}
    
    # Create plots directory
    parent_dir = os.path.dirname(os.getcwd())
    plots_dir = os.path.join(parent_dir, 'results', 'plots', "LIME", datetime.today().strftime("%d-%m-%Y"))
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load SMARTS mapping
    with open(smarts_mapping_path, 'r') as f:
        smarts_mapping = json.load(f)
    
    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        lime_fold_values, lime_fold_explanations = lime_values[i]

        # print("Fold:", i)
        for molecule_idx, (lime_array, lime_explanation) in enumerate(zip(lime_fold_values, lime_fold_explanations)):
            # Save explanation plots
            try:
                smiles = test_f.iloc[molecule_idx]['smiles']
                
                # Save PyPlot figure
                fig = lime_explanation.as_pyplot_figure()
                fig.suptitle(f'LIME Explanation for SMILES: {smiles}')
                plot_path = os.path.join(plots_dir, f"lime_explanation_{i}_{molecule_idx}_{timestamp}.svg")
                fig.savefig(plot_path, bbox_inches='tight', dpi=300, format='svg')
                plt.close(fig)
                
                # Save HTML explanation
                html_path = os.path.join(plots_dir, f"lime_explanation_{i}_{molecule_idx}_{timestamp}.html")
                lime_explanation.save_to_file(html_path)
                # print(f"Explanation saved for SMILES {smiles}")
                
            except Exception as e:
                print(f"An error occurred while saving explanation: {e}")

            # Process LIME values
            feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()
            lime_dict = {item[0].split('=')[0]: item[1] for item in lime_array}
            
            # Get top features
            # top_features = sorted(lime_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_i]
            top_features = sorted(lime_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            feature_names_only = [feature for feature, _ in top_features]

            # Map features to SMARTS and collect statistics
            for feature in feature_names_only:
                if lime_dict[feature] ==0:
                    continue
                key = (i, test_f.iloc[molecule_idx]['smiles'], feature)
                maccs_idx = int(feature.replace("maccsfingerprint", "")) #+ 1
                smarts_top_all[key] = smarts_mapping[f'maccsfingerprint{maccs_idx}'][0]
                
                # Initialize molecules statistics
                molecules_statistics_all[key] = {
                    "number_of_molecules_where_fingerprint": 0,
                    "number_where_important": 0,
                    "lime_value": np.abs(lime_dict[feature]),
                    "lime_sign": 'Positive' if lime_dict[feature] >= 0 else 'Negative',
                    "feature_in_smiles": bool(data.loc[data['smiles'] == key[1], feature].values[0] == 1),
                    "capacity_max": test_f.iloc[molecule_idx]['capacity_max'],
                    "capacity_pred": 0,
                    'Positive_explanation_add_count':  +1 if lime_dict[feature] >= 0 and data.loc[data['smiles'] == key[1], feature].values[0] == 1 else +0, 
                    'Negative_explanation_add_count':  +1 if lime_dict[feature] < 0 and data.loc[data['smiles'] == key[1], feature].values[0] == 1 else +0,
                    'Positive_explanation_del_count': +1 if lime_dict[feature] >= 0 and data.loc[data['smiles'] == key[1], feature].values[0] == 0 else +0,
                    'Negative_explanation_del_count': +1 if lime_dict[feature] < 0 and data.loc[data['smiles'] == key[1], feature].values[0] == 0 else +0,
                    "positive_changes": +1 if lime_dict[feature] >= 0 else +0,
                    "negative_changes": +1 if lime_dict[feature] < 0 else +0,
                }
                
                # Collect matching molecules
                non_zero_molecules = test_f[test_f[feature] == 1]['smiles'].tolist()
                match_molecules_all[key] = non_zero_molecules

    return smarts_top_all, match_molecules_all, molecules_statistics_all


def process_folds_global(folds, data, lime_values, smarts_mapping_path, top_i=10, cv_pipeline=None):
    smarts_top_all = {}
    match_molecules_all = {}
    molecules_statistics_all = {}
    
    # Create plots directory
    parent_dir = os.path.dirname(os.getcwd())
    plots_dir = os.path.join(parent_dir, 'results', 'plots', "LIME", datetime.today().strftime("%d-%m-%Y"))
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Prepare log file path
    log_dir = os.path.join(parent_dir, 'results', 'logs', "LIME", datetime.today().strftime("%d-%m-%Y"))
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'lime_global_fold_logs_{timestamp}.csv')

    all_logs = []

    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        lime_fold_values, lime_fold_explanations = lime_values[i]
        # Collect logs for this fold
        fold_logs = []
        fold_logs.append([f"Fold {i}"])
        fold_logs.append([f"LIME values for fold {i}: {lime_fold_values}"])
        # fold_logs.append([f"LIME explanations for fold {i}: {lime_fold_explanations}"])
        feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()
        lime_dicts = []
        for lime_array in lime_fold_values:
            lime_dict = {item[0].split('=')[0]: item[1] for item in lime_array}
            # print(f"LIME DICT:{lime_dict}")
            lime_dicts.append(lime_dict)
            feature_values = {feature: [] for feature in feature_names}
            sign_counts = {feature: {'Positive': 0, 'Negative': 0} for feature in feature_names}
            # New keys for explanation add/del counts
            positive_explanation_add_count = {feature: 0 for feature in feature_names}
            negative_explanation_add_count = {feature: 0 for feature in feature_names}
            positive_explanation_del_count = {feature: 0 for feature in feature_names}
            negative_explanation_del_count = {feature: 0 for feature in feature_names}

            for idx, lime_dict in enumerate(lime_dicts):
                for feature in feature_names:
                    val = lime_dict.get(feature, 0)
                    feature_values[feature].append(np.abs(val))
                    # Count sign
                    if val >= 0:
                        sign_counts[feature]['Positive'] += 1
                    else:
                        sign_counts[feature]['Negative'] += 1

                    # Check if feature exists in molecule (1 or 0)
                    # Use test_f.iloc[idx] to get the current molecule in the fold
                    if idx < len(test_f):
                        smiles = test_f.iloc[idx]['smiles']
                        # print(f"Processing feature: {feature} for SMILES: {smiles}")
                        feature_exists = int(data.loc[data['smiles'] == smiles, feature].values[0]) if feature in data.columns else 0
                        # print(f"Feature {feature} exists in SMILES {smiles}: {feature_exists}")
                        if val >= 0 and feature_exists == 1:
                            positive_explanation_add_count[feature] += 1
                        if val < 0 and feature_exists == 1:
                            negative_explanation_add_count[feature] += 1
                        if val >= 0 and feature_exists == 0:
                            positive_explanation_del_count[feature] += 1
                        if val < 0 and feature_exists == 0:
                            negative_explanation_del_count[feature] += 1

                # Save the print output to logs
                fold_logs.append([f"Feature: {feature}, Values: {feature_values[feature]}"])
            # Add sign counts summary to logs
            for feature in feature_names:
                fold_logs.append([
                f"Feature: {feature}, Positive count: {sign_counts[feature]['Positive']}, "
                f"Negative count: {sign_counts[feature]['Negative']}, "
                f"Total: {sign_counts[feature]['Positive'] + sign_counts[feature]['Negative']}, "
                f"Positive_explanation_add_count: {positive_explanation_add_count[feature]}, "
                f"Negative_explanation_add_count: {negative_explanation_add_count[feature]}, "
                f"Positive_explanation_del_count: {positive_explanation_del_count[feature]}, "
                f"Negative_explanation_del_count: {negative_explanation_del_count[feature]}"
                ])
        mean_abs_lime_values = np.array([np.mean(feature_values[feature]) for feature in feature_names])
        fold_logs.append([f"Mean absolute LIME values for fold {i}: {mean_abs_lime_values}"])

        # Get top features
        top_i_indices = np.argsort(mean_abs_lime_values)[-top_i:][::-1]
        fold_logs.append([f"Top {top_i} feature indices for fold {i}: {top_i_indices}"])
        top_i_indices = [idx for idx in top_i_indices if mean_abs_lime_values[idx] != 0]
        top_i_feature_names = [feature_names[i] for i in top_i_indices]
        fold_logs.append([f"Top {top_i} feature names for fold {i}: {top_i_feature_names}"])

        with open(smarts_mapping_path, 'r') as f:
            smarts_mapping = json.load(f)

        smarts_topi = {
            (i, match_molecule_global(feature, test_f, data), feature): smarts_mapping[f'maccsfingerprint{int(feature.replace("maccsfingerprint", ""))}'][0]
            for feature in top_i_feature_names
        }
        fold_logs.append([f"SMARTS for top features in fold {i}: {smarts_topi}"])
        match_molecules = {key: [] for key in smarts_topi.keys()}
        molecules_statistics = {s: {
            "number_of_molecules_where_fingerprint": 0,
            "number_where_important": 0,
            "lime_value": mean_abs_lime_values[feature_names.index(s[2])],
            "lime_sign": '',
            "feature_in_smiles": True,
            "capacity_max": 0,
            "capacity_pred": 0,
            "Positive_explanation_add_count": 0,
            "Negative_explanation_add_count": 0,
            "Positive_explanation_del_count": 0,
            "Negative_explanation_del_count": 0,
            "positive_changes": 0,
            "negative_changes": 0
        } for s in smarts_topi.keys()}

        for key in molecules_statistics.keys():
            feature = key[2]
            lime_values_for_feature = [lime_dict.get(feature, 0) for lime_dict in lime_dicts]
            X_test = test_f.drop(columns=['capacity_max', 'smiles'])
            capacity_pred = cv_pipeline.predict_capacity(X_test)
            df = pd.DataFrame({
                'lime_values': lime_values_for_feature,
                'capacity_values': capacity_pred
            })
            correlation = df.corr(method='spearman').loc['lime_values', 'capacity_values']
            molecules_statistics[key]["lime_sign"] = f'Positive|{correlation}' if correlation > 0 else f'Negative|{correlation}'
            # Use sign_counts to fill positive_count and negative_count
            # print(f"Feature: {feature}, Positive count: {sign_counts[feature]['Positive']}, Negative count: {sign_counts[feature]['Negative']}")
            molecules_statistics[key]["positive_changes"] = sign_counts[feature]['Positive']
            molecules_statistics[key]["negative_changes"] = sign_counts[feature]['Negative']
            molecules_statistics[key]["Positive_explanation_add_count"] = positive_explanation_add_count[feature]
            molecules_statistics[key]["Negative_explanation_add_count"] = negative_explanation_add_count[feature]
            molecules_statistics[key]["Positive_explanation_del_count"] = positive_explanation_del_count[feature]
            molecules_statistics[key]["Negative_explanation_del_count"] = negative_explanation_del_count[feature]

        fold_logs.append([f"Molecules correlation statistics for fold {i}: {molecules_statistics}"])
        smarts_top_all.update(smarts_topi)
        match_molecules_all.update(match_molecules)
        molecules_statistics_all.update(molecules_statistics)

        fold_logs.append([f"Top features indices for fold {i}: {top_i_indices}"])
        all_logs.extend(fold_logs)

        # Generate global importance plot for the fold
        plt.figure(figsize=(12, 6))
        plt.barh([feature_names[idx] for idx in top_i_indices],
                [mean_abs_lime_values[idx] for idx in top_i_indices])
        plt.title(f'Global LIME Feature Importance - Fold {i}')
        plt.xlabel('Mean |LIME value|')
        plot_path = os.path.join(plots_dir, f"lime_global_importance_fold_{i}_{timestamp}.svg")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300, format='svg')
        plt.close()

        # Generate correlation plot
        plt.figure(figsize=(12, 6))
        correlations = []
        for feat_idx in top_i_indices:
            feature = feature_names[feat_idx]
            lime_values_for_feature = [lime_dict.get(feature, 0) for lime_dict in lime_dicts]
            X_test = test_f.drop(columns=['capacity_max', 'smiles'])
            capacity_pred = cv_pipeline.predict_capacity(X_test)
            correlation = np.corrcoef(lime_values_for_feature, capacity_pred)[0, 1]
            correlations.append(correlation)
        
        plt.barh([feature_names[idx] for idx in top_i_indices], correlations)
        plt.title(f'LIME Values vs Predicted Capacity Correlation - Fold {i}')
        plt.xlabel('Correlation coefficient')
        plot_path = os.path.join(plots_dir, f"lime_correlation_fold_{i}_{timestamp}.svg")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300, format='svg')
        plt.close()


    # Save all logs to CSV
    pd.DataFrame(all_logs, columns=["log"]).to_csv(log_file_path, index=False)

    molecules_statistics_all = number_where_important_global(molecules_statistics_all, match_molecules_all)
    return smarts_top_all, match_molecules_all, molecules_statistics_all


def match_molecule_global(feature, test_f, full_data):
    """
    Match a molecule based on the feature. If no match is found in the test fold,
    search the entire dataset for a matching SMILES.

    Parameters:
    - feature: The feature to match.
    - test_f: The test fold data.
    - full_data: The entire dataset.

    Returns:
    - A matching SMILES string or 'C' if no match is found.
    """
    non_zero_molecules = test_f[test_f[feature] == 1]
    non_zero_molecules = non_zero_molecules['smiles'].tolist()

    if non_zero_molecules:
        return non_zero_molecules[0]

    non_zero_molecules_full = full_data[full_data[feature] == 1]
    non_zero_molecules_full = non_zero_molecules_full['smiles'].tolist()

    return non_zero_molecules_full[0] if non_zero_molecules_full else 'C'

def number_where_important_global(molecules_statistics_all, match_molecules_all):
    feature_importance_count = {}

    # Count how many times each feature is important
    for key in molecules_statistics_all.keys():
        feature = key[2]
        if feature not in feature_importance_count:
            feature_importance_count[feature] = 0
        feature_importance_count[feature] += 1

    # Update all keys with the count of how many times their feature was important
    for key in molecules_statistics_all.keys():
        feature = key[2]
        molecules_statistics_all[key]["number_where_important"] = feature_importance_count[feature]

    return molecules_statistics_all

def process_folds(folds, data, lime_values, smarts_mapping_path, local_explanation=True, cv_pipeline=None):
    if local_explanation:
        return process_folds_local(folds, data, lime_values, smarts_mapping_path)
    else:
        return process_folds_global(folds, data, lime_values, smarts_mapping_path, 10, cv_pipeline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run XAI Flow with specified parameters')
    parser.add_argument('--experiment_name', type=str, default='test', help='Name of the experiment (default: test)')
    parser.add_argument('--fold', type=int, default=5, help='Number of folds to process (default: 5)')
    parser.add_argument('--model', type=str, default='LIME', help='Model to use (default: LIME)')
    parser.add_argument('--local', action='store_true', default=False, help='Use local explanations (default: False)')
    parser.add_argument('--train_rfreg', action='store_true', default=False, help='Train new model (default: False)')
    parser.add_argument('--seed', type=int, default=42, help='Set seed value (default: 42)')
    parser.add_argument('--data_file', type=str, default=None, help='Path to the data file (default: None)')
    args = parser.parse_args()
    experiment_name = args.experiment_name
    model = args.model
    
    print(f"\n=== Running {args.model} ===\n")
    print("Arguments:", vars(args))
    mainXaiFlow(
        model,
        local_explanation=args.local,
        experiment_name=args.experiment_name,
        folds=args.fold,
        seed=args.seed,
        train_rfreg=args.train_rfreg,
        data_file=args.data_file
    )