import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import json
import argparse
import joblib
    
from AI_models.models import Models
from AI_models.eval_metrics import EvalMetrics
from utils.data_split import custom_data_kfold
from utils.exportlib import save_data_to_excel_with_highlights, save_scores_to_excel_new_sheet, save_interactions_to_excel_with_highlights

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

from SHAP.shap_cross_validation import CrossValidationShapPipeline
import SHAP.shapplot as plot_shap
from SHAP_IQ.shapiq_cross_validation import CrossValidationShapIqPipeline
import SHAP_IQ.shapiqplot as plot_iq


def mainXaiFlow(model, local_explanation=True, max_order_iq=1,experiment_name='battery'):
    if model != 'SHAP_IQ':
        max_order_iq = 1
    print("Model: ", model)
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)
    
    maccs_fingerprints = os.path.join(parent_dir, 'data', 'new_maccs_merged.csv')
    smarts_mapping_path = os.path.join(parent_dir, 'data', 'maccs_smarts_mapping.json')
    if max_order_iq > 1 and local_explanation:
        explenation_type = 'local_interactions'
    elif max_order_iq > 1 and not local_explanation:
        explenation_type = 'global_interactions'
    elif local_explanation:
        explenation_type = 'local'
    else:
        explenation_type = 'global'
    results_dir = os.path.join(parent_dir, 'results', experiment_name, model, explenation_type, datetime.today().strftime("%d-%m-%Y"))
    
    data = pd.read_csv(maccs_fingerprints)
    print(data.head())
    os.makedirs(results_dir, exist_ok=True)

    folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 5)

    # Train the model
    cv_pipeline = select_pipeline(model, data, folds, max_order_iq)
    results, scores, shap_values = cv_pipeline.train_pipeline('RFReg')
    
    plots_dir = os.path.join(parent_dir, 'results', 'plots', model, explenation_type, datetime.today().strftime("%d-%m-%Y"))
    create_plots(plots_dir, data, folds,model, shap_values, max_order_iq)   
    
    if max_order_iq > 1 and local_explanation: 
        smarts_top_all, molecules_statistics_all = process_folds_local_interactions(folds, data, shap_values, smarts_mapping_path, 10)
        match_molecules_all = {}
    elif max_order_iq > 1 and not local_explanation:
        smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds_global_interactions(folds, data, shap_values, smarts_mapping_path, 10)
    elif local_explanation:
        smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds(folds, data, shap_values, smarts_mapping_path, local_explanation)
    else:
        if model == 'SHAP_IQ':
            shap_values = [np.array(shap_values[i]) for i in range(len(shap_values))]
        smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds(folds, data, shap_values, smarts_mapping_path, local_explanation)
    # molecules_statistics_all = predict_capacity(cv_pipeline, smarts_top_all, molecules_statistics_all)
    molecules_statistics_all = count_molecules_with_fingerprint(data, molecules_statistics_all)
    molecules_statistics_all = count_important_features(data, molecules_statistics_all)
    
    excel_data = prepare_data_for_excel_export(match_molecules_all, smarts_top_all, molecules_statistics_all)
    if model == 'SHAP_IQ' and max_order_iq > 1:
        save_interactions_to_excel(excel_data, results_dir)
    else:
        save_molecules_to_excel(excel_data, results_dir)
    
    scores_data = create_dataframe_from_scores(scores, results)
    save_scores_to_excel(scores_data, results_dir)
    
    
            
def mainXaiFlow_all(model, local_explanation=True, max_order_iq=1, dataset_name='battery', experiment_name='test', folds=5):
    if model != 'SHAP_IQ':
        max_order_iq = 1
        
    print("Model: ", model)
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)
    
    maccs_fingerprints = os.path.join(parent_dir, 'data', 'new_maccs_merged.csv')
    smarts_mapping_path = os.path.join(parent_dir, 'data', 'maccs_smarts_mapping.json')
    if max_order_iq > 1 and local_explanation:
        explenation_type = 'local_interactions'
    elif max_order_iq > 1 and not local_explanation:
        explenation_type = 'global_interactions'
    elif local_explanation:
        explenation_type = 'local'
    else:
        explenation_type = 'global'
    results_dir = os.path.join(parent_dir, 'results', experiment_name, model, explenation_type, datetime.today().strftime("%d-%m-%Y"))
    
    data = pd.read_csv(maccs_fingerprints)
    print(data.head())
    os.makedirs(results_dir, exist_ok=True)

    folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 5)

    # Train the model
    cv_pipeline = select_pipeline(model, data, folds, max_order_iq)
    results, scores, shap_values = cv_pipeline.train_pipeline('RFReg')
    
    plots_dir = os.path.join(parent_dir, 'results', 'plots', model, explenation_type, datetime.today().strftime("%d-%m-%Y"))
    create_plots(plots_dir, data, folds,model, shap_values, max_order_iq)   
    
    if max_order_iq > 1 and local_explanation: 
        smarts_top_all, molecules_statistics_all = process_folds_local_interactions(folds, data, shap_values, smarts_mapping_path, 10)
        match_molecules_all = {}
    elif max_order_iq > 1 and not local_explanation:
        smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds_global_interactions(folds, data, shap_values, smarts_mapping_path, 10)
    elif local_explanation:
        smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds(folds, data, shap_values, smarts_mapping_path, local_explanation)
    else:
        if model == 'SHAP_IQ':
            shap_values = [np.array(shap_values[i]) for i in range(len(shap_values))]
        smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds(folds, data, shap_values, smarts_mapping_path, local_explanation, max_order_iq)
    # molecules_statistics_all = predict_capacity(cv_pipeline, smarts_top_all, molecules_statistics_all)
    molecules_statistics_all = count_molecules_with_fingerprint(data, molecules_statistics_all)
    molecules_statistics_all = count_important_features(data, molecules_statistics_all)
    
    excel_data = prepare_data_for_excel_export(match_molecules_all, smarts_top_all, molecules_statistics_all)
    if model == 'SHAP_IQ' and max_order_iq > 1:
        save_interactions_to_excel(excel_data, results_dir)
    else:
        save_molecules_to_excel(excel_data, results_dir)
    
    scores_data = create_dataframe_from_scores(scores, results)
    save_scores_to_excel(scores_data, results_dir)
    

def create_plots(plots_dir, data,folds, model, shap_values, max_order_iq=1):
    if model == 'SHAP':
        plot_shap.generate_shap_plots_folds(data,shap_values, plots_dir,['all'])
        plot_shap.generate_shap_plots_local(data, shap_values,folds, plots_dir, ['force', 'waterfall'])
    if model == 'SHAP_IQ':
        if max_order_iq > 1:
            plot_iq.plot_shapiq_local(data, shap_values,folds, plots_dir, ['all'])
            plot_iq.plot_shapiq_fold(data, shap_values, plots_dir, ['bar'])
        else:
            plot_iq.plot_shapiq_local(data, shap_values,folds, plots_dir, ['force', 'waterfall'])
            plot_iq.plot_shapiq_fold(data, shap_values, plots_dir, ['bar'])
    print("Plots saved to: ", plots_dir)


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


def select_pipeline(model, data, folds, max_order_iq=1):
    match model:
        case 'SHAP':
            return CrossValidationShapPipeline(
                X=data.drop(columns=['capacity_max', 'smiles']),
                y=data[['capacity_max']],
                folds=folds,
                metrics=['smape', 'pairwise_accuracy_score', 'rmse', 'ndcg_score'],
                save_dir='',
                data_name='battery',
                verbose=True
            )
        case 'SHAP_IQ':
            return CrossValidationShapIqPipeline(
                X=data.drop(columns=['capacity_max', 'smiles']),
                y=data[['capacity_max']],
                folds=folds,
                metrics=['smape', 'pairwise_accuracy_score', 'rmse', 'ndcg_score'],
                save_dir='',
                data_name='battery',
                verbose=True,
                iq_min_order=1,
                iq_max_order=max_order_iq
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
        "Shap_value": [],
        "shap_sign": [],
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
        excel_data["Shap_value"].append(molecules_statistics_all[key]["shap_value"])
        excel_data["shap_sign"].append(molecules_statistics_all[key]["shap_sign"])
        excel_data["Capacity Max"].append(molecules_statistics_all[key]["capacity_max"])
        excel_data["Capacity Pred"].append(molecules_statistics_all[key]["capacity_pred"])
        bbbb+=1
    print("bbbb:", bbbb)
    return excel_data


def save_molecules_to_excel(excel_data, results_dir):
    results_dir = results_dir + f'\\molecule_results_with_highlights_{datetime.now().strftime("%H-%M-%S")}.xlsx'
    save_data_to_excel_with_highlights(excel_data, results_dir)
    print(f"Molecule results with highlights saved to {results_dir}")
    
    
def save_interactions_to_excel(excel_data, results_dir):
    results_dir = results_dir + f'\\interactions_results_{datetime.now().strftime("%H-%M-%S")}.xlsx'
    save_interactions_to_excel_with_highlights(excel_data, results_dir)
    print(f"Interaction results saved to {results_dir}")


def save_scores_to_excel(scores_data, results_dir):
    results_dir = results_dir + f'\\molecule_scores_{datetime.now().strftime("%H-%M-%S")}.xlsx'
    save_scores_to_excel_new_sheet(scores_data, results_dir)
    print(f"Scores saved to {results_dir}")
    

def process_folds_local(folds, data, shap_values, smarts_mapping_path, top_i=5):
    smarts_top_all = {}
    match_molecules_all = {}
    molecules_statistics_all = {}
    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        shap_f = shap_values[i]

        for molecule_idx, shap_array in enumerate(shap_f):
            feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()
            abs_shap_values = np.abs(shap_array)
            top_10_indices = np.argsort(abs_shap_values)[-top_i:][::-1]
            top_10_indices = [idx for idx in top_10_indices if abs_shap_values[idx] != 0]
            top_10_feature_names = [feature_names[i] for i in top_10_indices]

            with open(smarts_mapping_path, 'r') as f:
                smarts_mapping = json.load(f)

            smarts_top10 = {
                (i, test_f.iloc[molecule_idx]['smiles'], feature): smarts_mapping[f'maccsfingerprint{int(feature.replace("maccsfingerprint", ""))+1}'][0]
                for feature in top_10_feature_names
            }

            match_molecules = {s: [] for s in smarts_top10.keys()}
            molecules_statistics = {s: {
                "number_of_molecules_where_fingerprint": 0,
                "number_where_important": 0,
                "shap_value": 0,
                "shap_sign": '',
                "feature_in_smiles": False,
                "capacity_max": test_f.iloc[molecule_idx]['capacity_max'],
                "capacity_pred": 0
            } for s in smarts_top10.keys()}
            
            for key, value in smarts_top10.items():
                non_zero_molecules = test_f[test_f[key[2]] == 1]
                non_zero_molecules = non_zero_molecules['smiles'].tolist()
                match_molecules[key].extend(non_zero_molecules)
                # count_mol_with_fingerprint = len(non_zero_molecules)
                # if key not in molecules_statistics_all:
                #     molecules_statistics[key]["number_where_important"] = count_mol_with_fingerprint
                # else:
                #     molecules_statistics[key]["number_where_important"] = molecules_statistics_all[key]["number_where_important"] + count_mol_with_fingerprint
                molecules_statistics[key]["shap_value"] = abs(shap_array[feature_names.index(key[2])])
                molecules_statistics[key]["shap_sign"] = 'Positive' if shap_array[feature_names.index(key[2])] >= 0 else 'Negative'
                molecules_statistics[key]["feature_in_smiles"] = bool(data.loc[data['smiles'] == key[1], key[2]].values[0] == 1)

            smarts_top_all.update(smarts_top10)
            match_molecules_all.update(match_molecules)
            molecules_statistics_all.update(molecules_statistics)

    return smarts_top_all, match_molecules_all, molecules_statistics_all


def process_folds_local_interactions(folds, data, shap_values, smarts_mapping_path, top_i=10):
    smarts_top_all = {}
    match_molecules_all = {}
    molecules_statistics_all = {}
    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        shap_f = shap_values[i]

        for molecule_idx, shap_array in enumerate(shap_f):
            # dictionary, sorted list of tuples
            _, sorted_top_list_interaction = shap_array.get_top_k(top_i + 1, as_interaction_values=False)
            sorted_top_list_interaction = sorted_top_list_interaction[1:]
            print(sorted_top_list_interaction)
            feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()
            top_10_feature_names = [
                ([feature_names[j] for j in i[0]], i[1]) for i in sorted_top_list_interaction
            ]

            with open(smarts_mapping_path, 'r') as f:
                smarts_mapping = json.load(f)

            smarts_top10 = {
                (i, test_f.iloc[molecule_idx]['smiles'], tuple(feature)): [
                    smarts_mapping[f'maccsfingerprint{int(f.replace("maccsfingerprint", "")) + 1}'][0]
                    for f in feature
                ]
                for feature, _ in top_10_feature_names
            }

            molecules_statistics = {s: {
                "number_of_molecules_where_fingerprint": 0,
                "number_where_important": 0,
                "shap_value": 0,
                "shap_sign": '',
                "feature_in_smiles": False ,
                "capacity_max": test_f.iloc[molecule_idx]['capacity_max'],
                "capacity_pred": 0
            } for s in smarts_top10.keys()}
            
            for key, value in smarts_top10.items():
                for feature, shap_value in top_10_feature_names:
                    if list(key[2]) == feature:
                        molecules_statistics[key]["shap_value"] = abs(shap_value)
                        molecules_statistics[key]["shap_sign"] = 'Positive' if shap_value >= 0 else 'Negative'
                        molecules_statistics[key]["feature_in_smiles"] = [
                            bool(data.loc[data['smiles'] == key[1], f].values[0] == 1) for f in key[2]
                        ]
                        break

            smarts_top_all.update(smarts_top10)
            molecules_statistics_all.update(molecules_statistics)

    return smarts_top_all, molecules_statistics_all


def process_folds_global(folds, data, shap_values, smarts_mapping_path, top_i=10):
    smarts_top_all = {}
    match_molecules_all = {}
    molecules_statistics_all = {}  # Initialize molecules_statistics_all

    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        shap_f = shap_values[i]
        feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()
        mean_abs_shap_values = np.mean(np.abs(shap_f), axis=0)
        top_i_indices = np.argsort(mean_abs_shap_values)[-top_i:][::-1]
        top_i_indices = [idx for idx in top_i_indices if mean_abs_shap_values[idx] != 0]
        top_i_feature_names = [feature_names[i] for i in top_i_indices]

        with open(smarts_mapping_path, 'r') as f:
            smarts_mapping = json.load(f)

        smarts_topi = {
            (i, match_molecule_global(feature,test_f,data), feature): smarts_mapping[f'maccsfingerprint{int(feature.replace("maccsfingerprint", ""))+1}'][0]
            for feature in top_i_feature_names
        }
        # print("SMARTS Top 10:", smarts_topi)

        match_molecules = {key: [] for key in smarts_topi.keys()}
        molecules_statistics = {s: {
            "number_of_molecules_where_fingerprint": 0,
            "number_where_important": 0,
            "shap_value": mean_abs_shap_values[feature_names.index(s[2])],
            "shap_sign": '',
            "feature_in_smiles": True,
            "capacity_max": 0,
            "capacity_pred": 0
        } for s in smarts_topi.keys()}

        for key in molecules_statistics.keys():
            feature = key[2]
            shap_values_for_feature = shap_f[:, feature_names.index(feature)]
            capacity_values = test_f['capacity_max'].values
            df = pd.DataFrame({
                'shap_values': shap_values_for_feature,
                'capacity_values': capacity_values
            })
            correlation = df.corr(method='spearman').loc['shap_values', 'capacity_values']
            molecules_statistics[key]["shap_sign"] = f'Positive|{correlation}' if correlation > 0 else f'Negative|{correlation}'

        smarts_top_all.update(smarts_topi)
        match_molecules_all.update(match_molecules)
        molecules_statistics_all.update(molecules_statistics)  # Update molecules_statistics_all

    molecules_statistics_all = number_where_important_global(molecules_statistics_all,match_molecules_all)
    return smarts_top_all, match_molecules_all, molecules_statistics_all


def process_folds_global_interactions(folds, data, shap_values, smarts_mapping_path, top_i=10):
    smarts_top_all = {}
    molecules_statistics_all = {}
    match_molecules_all = {}

    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        shap_f = shap_values[i]
        # print("Fold number:", i)
        # print("Test data:", test_f)
        # print("SHAP values:", shap_f)
        # print("\n================================================\n")
        # Calculate mean interaction values across all molecules in the fold
        mean_interactions = {}
        for molecule_idx, shap_array in enumerate(shap_f):
            _, interactions = shap_array.get_top_k(top_i + 1, as_interaction_values=False)
            interactions = interactions[1:] 
            
            for interaction in interactions:
                key = tuple(sorted(interaction[0]))  # Sort indices to ensure consistent key
                value = abs(interaction[1])
                if key not in mean_interactions:
                    mean_interactions[key] = []
                mean_interactions[key].append(value)

        # Calculate means and sort to get top interactions
        mean_interaction_values = {k: np.mean(np.abs(v)) for k, v in mean_interactions.items()}
        top_interactions = sorted(mean_interaction_values.items(), key=lambda x: x[1], reverse=True)[:top_i]

        feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()


        # print("mean_interaction_values:", mean_interaction_values)
        # print("Top interactions:", top_interactions)
        # # print("Feature names:", feature_names)
        # print("\n================================================\n")
        
        with open(smarts_mapping_path, 'r') as f:
            smarts_mapping = json.load(f)

        # Create SMARTS dictionary for top interactions
        smarts_topi = {}
        match_topi = {}
        mol_stat = {}

        for indices, mean_value in top_interactions:
            print("Indices:", indices)
            features = [feature_names[idx] for idx in indices]
            print("features:", features)
            smiles = match_molecule_global(features[0], test_f, data)  # Use first feature to match molecule
            
            key = (i, smiles, tuple(features))
            smarts = [
                smarts_mapping[f'maccsfingerprint{int(f.replace("maccsfingerprint", ""))+1}'][0]
                for f in features
            ]
            smarts_topi[key] = smarts

            match_molecules = {key: [] for key in smarts_topi.keys()}
            match_topi[key] = match_molecules
            # Initialize statistics for this interaction
            molecules_statistics = {
                "number_of_molecules_where_fingerprint": 0,
                "number_where_important": 0,
                "shap_value": mean_value,
                "shap_sign": '',
                "feature_in_smiles": False,
                "capacity_max": 0,
                "capacity_pred": 0
            }

            # Calculate correlation for interaction
            interaction_values = []
            capacity_values = []
            for mol_idx, mol_shap in enumerate(shap_f):
                _, mol_interactions = mol_shap.get_top_k(len(mol_shap), as_interaction_values=False)
                for mol_int in mol_interactions:
                    if tuple(sorted(mol_int[0])) == indices:
                        interaction_values.append(mol_int[1])
                        capacity_values.append(test_f.iloc[mol_idx]['capacity_max'])
                        break

            if interaction_values:
                df = pd.DataFrame({
                    'interaction_values': interaction_values,
                    'capacity_values': capacity_values
                })
                correlation = df.corr(method='spearman').loc['interaction_values', 'capacity_values']
                molecules_statistics["shap_sign"] = f'Positive|{correlation}' if correlation > 0 else f'Negative|{correlation}'
            mol_stat[key] = molecules_statistics

        match_molecules_all.update(match_topi)
        molecules_statistics_all.update(mol_stat)
        smarts_top_all.update(smarts_topi)

    molecules_statistics_all = count_interaction_features(data, molecules_statistics_all)
    return smarts_top_all, match_molecules_all, molecules_statistics_all

def count_interaction_features(data, molecules_statistics_all):
    interaction_count = {}
    
    # Count occurrences of each feature in interactions
    for key in molecules_statistics_all.keys():
        features = key[2]
        for feature in features:
            if feature not in interaction_count:
                interaction_count[feature] = 0
            interaction_count[feature] += 1

    # Update statistics for each interaction
    for key in molecules_statistics_all.keys():
        features = key[2]
        # Sum the counts of all features in this interaction
        total_importance = sum(interaction_count.get(feature, 0) for feature in features)
        molecules_statistics_all[key]["number_where_important"] = total_importance

    return molecules_statistics_all


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

def process_folds(folds, data, shap_values, smarts_mapping_path, local_explanation=True):
    if local_explanation:
        return process_folds_local(folds, data, shap_values, smarts_mapping_path)
    else:
        return process_folds_global(folds, data, shap_values, smarts_mapping_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run XAI Flow with specified parameters')
    parser.add_argument('--dataset_name', type=str, default='battery', help='Name of the dataset (default: battery)')
    parser.add_argument('--experiment_name', type=str, default='test', help='Name of the experiment (default: test)')
    parser.add_argument('--fold', type=int, default=5, help='Number of folds to process (default: 5)')
    parser.add_argument('--model', type=str, choices=['SHAP', 'SHAP_IQ', 'both'], default='both', help='Model to use (default: both)')
    parser.add_argument('--local', action='store_true', default=False, help='Use local explanations (default: False)')
    parser.add_argument('--max_order', type=int, default=1, help='Maximum interaction order for SHAP_IQ (default: 1)')
    
    args = parser.parse_args()
    
    # Determine which models to run
    models_to_run = []
    if args.model == 'both':
        models_to_run = ['SHAP', 'SHAP_IQ']
    else:
        models_to_run = [args.model]
    
    # Run the main flow with the specified arguments
    for model in models_to_run:
        print(f"\n=== Running {model} ===\n")
        # mainXaiFlow_all(
        #     model=model,
        #     local_explanation=args.local,
        #     max_order_iq=args.max_order,
        #     dataset_name=args.dataset_name,
        #     experiment_name=args.experiment_name,
        #     fold=args.fold
        # )
    
    model = ['SHAP','SHAP_IQ'] # 'SHAP' or 'SHAP_IQ' - in the future it should be a list of models to run 
    local_explanation = False
    experiment_name= 'battery_test'
    max_order_iq = 2
    [mainXaiFlow(m, local_explanation, max_order_iq,experiment_name) for m in model]