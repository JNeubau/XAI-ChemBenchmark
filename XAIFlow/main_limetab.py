import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import json
    
from AI_models.models import Models
from AI_models.eval_metrics import EvalMetrics
from utils.data_split import custom_data_kfold
from utils.exportlib import save_data_to_excel_with_highlights, save_scores_to_excel_new_sheet, save_interactions_to_excel_with_highlights

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

from LIME.lime_tabular_explainer import CrossValidationLimePipeline
# from LIME.lime_cross_validation import CrossValidationLimePipeline
# # from LIME.limeplot import generate_lime_plots
# import LIME.limeplot as plot_lime
# from LIME_IQ.limeiq_cross_validation import CrossValidationLimeIqPipeline
# import LIME_IQ.limeiqplot as plot_iq


def mainXaiFlow(model, local_explanation=True, max_order_iq=1):
    if model != 'LIME':
        max_order_iq = 1
    print("Model: ", model)
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)
    
    maccs_fingerprints = os.path.join(parent_dir, 'data', 'maccs_merged.csv')
    smarts_mapping_path = os.path.join(parent_dir, 'data', 'maccs_smarts_mapping.json')
    explenation_type = 'local'    
    results_dir = os.path.join(parent_dir, 'results', 'battery', model, explenation_type, datetime.today().strftime("%d-%m-%Y"))
    
    data = pd.read_csv(maccs_fingerprints, index_col=0)
    print(data.head())
    os.makedirs(results_dir, exist_ok=True)

    folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 5)

    # Train the model
    cv_pipeline = select_pipeline(model, data, folds, max_order_iq)
    results, scores, lime_values = cv_pipeline.train_pipeline('RFReg')
    
    # plots_dir = os.path.join(parent_dir, 'results', 'plots', model, explenation_type, datetime.today().strftime("%d-%m-%Y"))
    # create_plots(plots_dir, data, model, lime_values, max_order_iq)   
    
    # if max_order_iq > 1: 
    #     smarts_top_all, molecules_statistics_all = process_folds_local_interactions(folds, data, lime_values, smarts_mapping_path, 10)
    #     match_molecules_all = {}
    # else:
    #     if model == 'LIME_IQ':
    #         lime_values = [np.array(lime_values[i]) for i in range(len(lime_values))]
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


def select_pipeline(model, data, folds, max_order_iq=1):
    match model:
        # case 'LIME':
        #     return CrossValidationLimePipeline(
        #         X=data.drop(columns=['capacity_max', 'smiles']),
        #         y=data[['capacity_max']],
        #         folds=folds,
        #         metrics=['smape', 'pairwise_accuracy_score', 'rmse', 'ndcg_score'],
        #         save_dir='',
        #         data_name='battery',
        #         verbose=True
        #     )
        # case 'LIME_IQ':
        #     return CrossValidationLimeIqPipeline(
        #         X=data.drop(columns=['capacity_max', 'smiles']),
        #         y=data[['capacity_max']],
        #         folds=folds,
        #         metrics=['smape', 'pairwise_accuracy_score', 'rmse', 'ndcg_score'],
        #         save_dir='',
        #         data_name='battery',
        #         verbose=True,
        #         iq_min_order=1,
        #         iq_max_order=max_order_iq
        #     )
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
    

def process_folds_local(folds, data, lime_values, smarts_mapping_path, top_i=5):
    smarts_top_all = {}
    match_molecules_all = {}
    molecules_statistics_all = {}
    
    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        lime_f = lime_values[i]

        print("Fold:", i)
        for molecule_idx, lime_array in enumerate(lime_f):
            feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()
            # Convert LIME explanation list to dict
            print("molecule_idx:", test_f.iloc[molecule_idx]['smiles'])
            lime_dict = {item[0].split('=')[0]: item[1] for item in lime_array}
            print("LIME dict:", lime_dict)
            # abs_lime_values = lime_dict
            top_features = lime_dict.items()
            # top_features = sorted_features
            feature_names_only = [feature for feature, _ in top_features]
            for feature in feature_names_only:
                if data.loc[data['smiles'] == test_f.iloc[molecule_idx]['smiles'], feature].values[0] == 1:
                    # molecules_statistics[(i, test_f.iloc[molecule_idx]['smiles'], feature)]["feature_in_smiles"] = True
                    print("Feature in SMILES:", feature, "True")
                else:
                    # molecules_statistics[(i, test_f.iloc[molecule_idx]['smiles'], feature)]["feature_in_smiles"] = False
                    print("Feature in SMILES:", feature, "False")
            # print("Test fold:", test_f.iloc[molecule_idx]['smiles'])
            # print("LIME dict:", lime_dict)
            # print("Top features:", top_features)
            # print("Feature names only:", feature_names_only)

            with open(smarts_mapping_path, 'r') as f:
                smarts_mapping = json.load(f)

            smarts_top10 = {
                (i, test_f.iloc[molecule_idx]['smiles'], feature): smarts_mapping[f'maccsfingerprint{int(feature.replace("maccsfingerprint", ""))+1}'][0]
                for feature in feature_names_only
            }

            match_molecules = {s: [] for s in smarts_top10.keys()}
            molecules_statistics = {s: {
                "number_of_molecules_where_fingerprint": 0,
                "number_where_important": 0,
                "lime_value": 0,
                "lime_sign": '',
                "feature_in_smiles": False,
                "capacity_max": test_f.iloc[molecule_idx]['capacity_max'],
                "capacity_pred": 0
            } for s in smarts_top10.keys()}
            
            for key, value in smarts_top10.items():
                non_zero_molecules = test_f[test_f[key[2]] == 1]
                non_zero_molecules = non_zero_molecules['smiles'].tolist()
                match_molecules[key].extend(non_zero_molecules)
                # print("Lime value:", lime_dict[key[2]])
                molecules_statistics[key]["lime_value"] = lime_dict[key[2]]
                molecules_statistics[key]["lime_sign"] = 'Positive' if lime_dict[key[2]] >= 0 else 'Negative'
                molecules_statistics[key]["feature_in_smiles"] = bool(data.loc[data['smiles'] == key[1], key[2]].values[0] == 1)

            smarts_top_all.update(smarts_top10)
            match_molecules_all.update(match_molecules)
            molecules_statistics_all.update(molecules_statistics)

    return smarts_top_all, match_molecules_all, molecules_statistics_all

def process_folds_global(folds, data, lime_values, smarts_mapping_path, top_i=10):
   
    return 0


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
    local_explanation = True
    [mainXaiFlow(m, local_explanation) for m in model]