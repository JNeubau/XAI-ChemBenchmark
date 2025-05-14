import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
    
from AI_models.models import Models
from AI_models.eval_metrics import EvalMetrics
from utils.data_split import custom_data_kfold
from utils.exportlib import save_data_to_excel_with_highlights_lime, save_scores_to_excel_new_sheet

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

from LIME.lime_tabular_explainer import CrossValidationLimePipeline

def mainXaiFlow(model, local_explanation=True,experiment_name='battery'):
    print("Model: ", model)
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)
    
    maccs_fingerprints = os.path.join(parent_dir, 'data', 'new_maccs_merged.csv')
    smarts_mapping_path = os.path.join(parent_dir, 'data', 'maccs_smarts_mapping.json')
    explenation_type = 'local'    
    results_dir = os.path.join(parent_dir, 'results', experiment_name, model, explenation_type, datetime.today().strftime("%d-%m-%Y"))
    
    data = pd.read_csv(maccs_fingerprints)
    print(data.head())
    os.makedirs(results_dir, exist_ok=True)

    folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 5)

    # Train the model
    cv_pipeline = select_pipeline(model, data, folds)
    results, scores, lime_values = cv_pipeline.train_pipeline('RFReg')

    smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds(folds, data, lime_values, smarts_mapping_path, local_explanation)
    molecules_statistics_all = count_molecules_with_fingerprint(data, molecules_statistics_all)
    molecules_statistics_all = count_important_features(data, molecules_statistics_all)
    
    excel_data = prepare_data_for_excel_export(match_molecules_all, smarts_top_all, molecules_statistics_all)
    save_molecules_to_excel(excel_data, results_dir)
    
    scores_data = create_dataframe_from_scores(scores, results)
    save_scores_to_excel(scores_data, results_dir)
    
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


def select_pipeline(model, data, folds):
    match model:
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
        "Fold_No": [],
        "Smiles_key": [],
        "Feature_key": [],
        "SMARTS": [],
        "Molecule": [],
        "number_of_molecules_where_fingerprint": [],
        "Number_where_important": [],
        'feature_in_smiles': [],
        "lime_value": [],
        "lime_sign": [],
        "Capacity Max": [],
        "Capacity Pred": [],
    }
            
    bbbb=0
    for key, smarts in smarts_top.items():
        # print("=============molecule===============")
        # print("key:", key)
        # print("smarts:", smarts)
        excel_data["Fold_No"].append(key[0])
        excel_data["Smiles_key"].append(key[1])
        excel_data["Feature_key"].append(key[2])
        excel_data["SMARTS"].append(smarts)
        excel_data["Molecule"].append(key[1])
        excel_data["number_of_molecules_where_fingerprint"].append(molecules_statistics_all[key]["number_of_molecules_where_fingerprint"])
        excel_data["Number_where_important"].append(molecules_statistics_all[key]["number_where_important"])
        excel_data["feature_in_smiles"].append(molecules_statistics_all[key]["feature_in_smiles"])
        excel_data["lime_value"].append(molecules_statistics_all[key]["lime_value"])
        excel_data["lime_sign"].append(molecules_statistics_all[key]["lime_sign"])
        excel_data["Capacity Max"].append(molecules_statistics_all[key]["capacity_max"])
        excel_data["Capacity Pred"].append(molecules_statistics_all[key]["capacity_pred"])
        bbbb+=1
    # print("bbbb:", bbbb)
    return excel_data


def save_molecules_to_excel(excel_data, results_dir):
    results_dir = results_dir + f'\\molecule_results_with_highlights_{datetime.now().strftime("%H-%M-%S")}.xlsx'
    save_data_to_excel_with_highlights_lime(excel_data, results_dir)
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

        print("Fold:", i)
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
                print(f"Explanation saved for SMILES {smiles}")
                
            except Exception as e:
                print(f"An error occurred while saving explanation: {e}")

            # Process LIME values
            feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()
            lime_dict = {item[0].split('=')[0]: item[1] for item in lime_array}
            
            # Get top features
            top_features = sorted(lime_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_i]
            feature_names_only = [feature for feature, _ in top_features]

            # Map features to SMARTS and collect statistics
            for feature in feature_names_only:
                key = (i, test_f.iloc[molecule_idx]['smiles'], feature)
                maccs_idx = int(feature.replace("maccsfingerprint", "")) + 1
                smarts_top_all[key] = smarts_mapping[f'maccsfingerprint{maccs_idx}'][0]
                
                # Initialize molecules statistics
                molecules_statistics_all[key] = {
                    "number_of_molecules_where_fingerprint": 0,
                    "number_where_important": 0,
                    "lime_value": lime_dict[feature],
                    "lime_sign": 'Positive' if lime_dict[feature] >= 0 else 'Negative',
                    "feature_in_smiles": bool(data.loc[data['smiles'] == key[1], feature].values[0] == 1),
                    "capacity_max": test_f.iloc[molecule_idx]['capacity_max'],
                    "capacity_pred": 0
                }
                
                # Collect matching molecules
                non_zero_molecules = test_f[test_f[feature] == 1]['smiles'].tolist()
                match_molecules_all[key] = non_zero_molecules

    return smarts_top_all, match_molecules_all, molecules_statistics_all


def process_folds_global(folds, data, lime_values, smarts_mapping_path, top_i=5):
    smarts_top_all = {}
    match_molecules_all = {}
    molecules_statistics_all = {}
    
    # Create plots directory
    parent_dir = os.path.dirname(os.getcwd())
    plots_dir = os.path.join(parent_dir, 'results', 'plots', "LIME", datetime.today().strftime("%d-%m-%Y"))
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        lime_fold_values, lime_fold_explanations = lime_values[i]
        feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()

        # Calculate mean absolute LIME values across all molecules in the fold
        lime_arrays = [np.array([item[1] for item in array]) for array in lime_fold_values]
        mean_abs_lime_values = np.mean(np.abs(lime_arrays), axis=0)
        
        print("Fold:", i)
        print("Mean absolute LIME values:", mean_abs_lime_values)
        print("LIME arrays:", lime_arrays)
        # # Generate global importance plot for the fold
        # plt.figure(figsize=(12, 6))
        # top_features_idx = np.argsort(mean_abs_lime_values)[-top_i:]
        # plt.barh([feature_names[idx] for idx in top_features_idx],
        #         [mean_abs_lime_values[idx] for idx in top_features_idx])
        # plt.title(f'Global LIME Feature Importance - Fold {i}')
        # plt.xlabel('Mean |LIME value|')
        # plot_path = os.path.join(plots_dir, f"lime_global_importance_fold_{i}_{timestamp}.svg")
        # plt.savefig(plot_path, bbox_inches='tight', dpi=300, format='svg')
        # plt.close()

        # # Generate correlation plot
        # plt.figure(figsize=(12, 6))
        # correlations = []
        # for feat_idx in top_features_idx:
        #     feature = feature_names[feat_idx]
        #     lime_values_for_feature = [array[feat_idx] for array in lime_arrays]
        #     capacity_values = test_f['capacity_max'].values
        #     correlation = np.corrcoef(lime_values_for_feature, capacity_values)[0, 1]
        #     correlations.append(correlation)
        
        # plt.barh([feature_names[idx] for idx in top_features_idx], correlations)
        # plt.title(f'LIME Values vs Capacity Correlation - Fold {i}')
        # plt.xlabel('Correlation coefficient')
        # plot_path = os.path.join(plots_dir, f"lime_correlation_fold_{i}_{timestamp}.svg")
        # plt.savefig(plot_path, bbox_inches='tight', dpi=300, format='svg')
        # plt.close()

        top_i_indices = np.argsort(mean_abs_lime_values)[-top_i:][::-1]
        top_i_indices = [idx for idx in top_i_indices if mean_abs_lime_values[idx] != 0]
        top_i_feature_names = [feature_names[i] for i in top_i_indices]

        print("\n==================================\nTop features:", top_i_feature_names)
        print("Mean absolute LIME values for top features:", mean_abs_lime_values[top_i_indices])

        with open(smarts_mapping_path, 'r') as f:
            smarts_mapping = json.load(f)

        smarts_topi = {
            (i, match_molecule_global(feature, test_f, data), feature): smarts_mapping[f'maccsfingerprint{int(feature.replace("maccsfingerprint", ""))+1}'][0]
            for feature in top_i_feature_names
        }

        match_molecules = {key: [] for key in smarts_topi.keys()}
        molecules_statistics = {s: {
            "number_of_molecules_where_fingerprint": 0,
            "number_where_important": 0,
            "lime_value": mean_abs_lime_values[feature_names.index(s[2])],
            "lime_sign": '',
            "feature_in_smiles": True,
            "capacity_max": 0,
            "capacity_pred": 0
        } for s in smarts_topi.keys()}

        for key in molecules_statistics.keys():
            feature = key[2]
            lime_values_for_feature = [array[feature_names.index(feature)] for array in lime_arrays]
            capacity_values = test_f['capacity_max'].values
            df = pd.DataFrame({
                'lime_values': lime_values_for_feature,
                'capacity_values': capacity_values
            })
            correlation = df.corr(method='spearman').loc['lime_values', 'capacity_values']
            molecules_statistics[key]["lime_sign"] = f'Positive|{correlation}' if correlation > 0 else f'Negative|{correlation}'

        smarts_top_all.update(smarts_topi)
        match_molecules_all.update(match_molecules)
        molecules_statistics_all.update(molecules_statistics)

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

def process_folds(folds, data, lime_values, smarts_mapping_path, local_explanation=True):
    if local_explanation:
        return process_folds_local(folds, data, lime_values, smarts_mapping_path)
    else:
        return process_folds_global(folds, data, lime_values, smarts_mapping_path)


if __name__ == '__main__':
    model = ['LIME'] 
    local_explanation = False
    experiment_name = 'global'
    [mainXaiFlow(m, local_explanation,experiment_name) for m in model]