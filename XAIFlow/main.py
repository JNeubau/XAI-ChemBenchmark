import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import json
    
from AI_models.models import Models
from AI_models.eval_metrics import EvalMetrics
from utils.data_split import custom_data_kfold
from utils.exportlib import save_data_to_excel_with_highlights,save_data_to_excel_with_highlights_no_sort

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

from SHAP.shap_cross_validation import CrossValidationShapPipeline
from SHAP_IQ.shapiq_cross_validation import CrossValidationShapIqPipeline


def mainXaiFlow(model, local_explanation=True):
    print("Model: ", model)
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)
    
    maccs_fingerprints = os.path.join(parent_dir, 'data', 'maccs_merged.csv')
    smarts_mapping_path = os.path.join(parent_dir, 'data', 'maccs_smarts_mapping.json')
    if local_explanation:
        explenation_type = 'local'
    else:
        explenation_type = 'global'
    results_dir = os.path.join(parent_dir, 'results', 'battery', model, explenation_type, datetime.today().strftime("%d-%m-%Y"))
    
    data = pd.read_csv(maccs_fingerprints, index_col=0)
    print(data.head())
    os.makedirs(results_dir, exist_ok=True)

    folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 5)

    # Train the model
    cv_pipeline = select_pipeline(model, data, folds)
    results, scores, shap_values = cv_pipeline.train_pipeline('RFReg')
    print("Results:", results)

    smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds(folds, data, shap_values, smarts_mapping_path, local_explanation)
    molecules_statistics_all = count_molecules_with_fingerprint(data, molecules_statistics_all)
    
    excel_data = prepare_data_for_excel_export(match_molecules_all, smarts_top_all, molecules_statistics_all)
    save_to_excel(excel_data, results_dir)

def count_molecules_with_fingerprint(maccs_fingerprints_data, molecules_statistics_all):
    column_ones_count = maccs_fingerprints_data.sum(axis=0).to_dict()

    for key in molecules_statistics_all.keys():
        column_name = key[2]
        if column_name in column_ones_count:
            molecules_statistics_all[key]["number_of_molecules_where_fingerprint"] = column_ones_count[column_name]
    return molecules_statistics_all

# def count_ones_in_columns(maccs_fingerprints_data, molecules_statistics_all):
#     for key in molecules_statistics_all.keys():
#         column_name = key[2]
#         if column_name in maccs_fingerprints_data.columns:
#             count_ones = maccs_fingerprints_data[column_name].sum()
#             molecules_statistics_all[key]["number_of_molecules_where_fingerprint"] = count_ones
#     return molecules_statistics_all

def select_pipeline(model, data, folds):
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
        "Shap_value": [],
    }

    # bbbb=0
    # for key, smarts in smarts_top.items():
    #     # molecule = match_molecules[key]
    #     print("=============smarts===============")
    #     for molecule in match_molecules[key]:
    #         print("=============molecule===============")
    #         print("key:", key)
    #         print("molecule:", molecule)
    #         print("smarts:", smarts)
    #         excel_data["Fold_No"].append(key[0])
    #         excel_data["Smiles_key"].append(key[1])
    #         excel_data["Feature_key"].append(key[2])
    #         excel_data["SMARTS"].append(smarts)
    #         excel_data["Molecule"].append(molecule)
    #         excel_data["number_of_molecules_where_fingerprint"].append(molecules_statistics_all[key]["number_of_molecules_where_fingerprint"])
    #         excel_data["Number_where_important"].append(molecules_statistics_all[key]["number_where_important"])
    #         bbbb+=1
            
    bbbb=0
    for key, smarts in smarts_top.items():
        # molecule = match_molecules[key]
        # print("=============smarts===============")
        # for molecule in match_molecules[key]:
        print("=============molecule===============")
        print("key:", key)
        # print("molecule:", molecule)
        print("smarts:", smarts)
        excel_data["Fold_No"].append(key[0])
        excel_data["Smiles_key"].append(key[1])
        excel_data["Feature_key"].append(key[2])
        excel_data["SMARTS"].append(smarts)
        excel_data["Molecule"].append(key[1])
        excel_data["number_of_molecules_where_fingerprint"].append(molecules_statistics_all[key]["number_of_molecules_where_fingerprint"])
        excel_data["Number_where_important"].append(molecules_statistics_all[key]["number_where_important"])
        excel_data["feature_in_smiles"].append(molecules_statistics_all[key]["feature_in_smiles"])
        excel_data["Shap_value"].append(molecules_statistics_all[key]["shap_value"])
        bbbb+=1

    # for key, molecule in match_molecules.items():
    #     excel_data["Feature"].append(key)
    #     # excel_data["SMARTS"].append(smarts)
    #     excel_data["Molecule"].append(molecule)
    #     smiles_list.append(molecule)
    #     smarts_list.append(smarts)

    print("bbbb:", bbbb)
    return excel_data


def save_to_excel(excel_data, results_dir):
    excel_output_path = results_dir + f'\\molecule_results_with_highlights_{datetime.now().strftime("%H-%M-%S")}.xlsx'
    save_data_to_excel_with_highlights(excel_data, excel_output_path)
    print(f"Molecule results with highlights saved to {excel_output_path}")


def process_folds_local(folds, data, shap_values, smarts_mapping_path, top_i=5):
    smarts_top_all = {}
    match_molecules_all = {}
    molecules_statistics_all = {}
    # aaaa=0
    for i, fold in enumerate(folds):
        # print("=====================================")
        # print("Fold:", i)
        # print("=====================================")
        test_f = data.loc[fold[1]]
        shap_f = shap_values[i]
        # print("SHAP values shape:", shap_f)

        for molecule_idx, shap_array in enumerate(shap_f):
            # print("Molecule index:", molecule_idx)
            # print("SHAP array shape:", shap_array.shape)
            # print("SHAP array:", shap_array)
            feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()
            abs_shap_values = np.abs(shap_array)
            top_10_indices = np.argsort(abs_shap_values)[-top_i:][::-1]
            top_10_indices = [idx for idx in top_10_indices if abs_shap_values[idx] != 0]
            top_10_feature_names = [feature_names[i] for i in top_10_indices]

            # print("mol idx:", molecule_idx)

            with open(smarts_mapping_path, 'r') as f:
                smarts_mapping = json.load(f)

            smarts_top10 = {
                (i, test_f.iloc[molecule_idx]['smiles'], feature): smarts_mapping[f'maccsfingerprint{int(feature.replace("maccsfingerprint", "")) + 1}'][0]
                for feature in top_10_feature_names
            }

            match_molecules = {s: [] for s in smarts_top10.keys()}
            molecules_statistics = {s: {
                "number_of_molecules_where_fingerprint": 0,
                "number_where_important": 0,
                "shap_value": 0,
                "feature_in_smiles": False 
            } for s in smarts_top10.keys()}
            
            for key, value in smarts_top10.items():
                # print("key:", key)
                # print("key:", key[2])
                # print("value:", value)
                # print("test_f:", test_f[test_f[key[2]] == 1]) 
                # non_zero_molecules <- how many finderprints the molecule contains
                # print(test_f)
                # print("test_f:", test_f[key[2]])
                non_zero_molecules = test_f[test_f[key[2]] == 1]
                non_zero_molecules = non_zero_molecules['smiles'].tolist()
                match_molecules[key].extend(non_zero_molecules)
                count_mol_with_fingerprint = len(non_zero_molecules)
                # count_mol_with_fingerprint = test_f[key[2]].sum()
                if key not in molecules_statistics_all:
                    molecules_statistics[key]["number_where_important"] = count_mol_with_fingerprint
                else:
                    molecules_statistics[key]["number_where_important"] = molecules_statistics_all[key]["number_where_important"] + count_mol_with_fingerprint
                molecules_statistics[key]["shap_value"] = shap_array[feature_names.index(key[2])]
                molecules_statistics[key]["feature_in_smiles"] = data.loc[data['smiles'] == key[1], key[2]].values[0] == 1
                # print("non_zero_molecules:", non_zero_molecules)
                # print("match_molecules:", match_molecules)

            smarts_top_all.update(smarts_top10)
            match_molecules_all.update(match_molecules)
            molecules_statistics_all.update(molecules_statistics)
    # print("aaaa:", aaaa)

    return smarts_top_all, match_molecules_all, molecules_statistics_all

def process_folds_local_top_i(folds, data, shap_values, smarts_mapping_path, top_i=3):
    smarts_top_all = {}
    match_molecules_all = {}
    molecules_statistics_all = {}
    
    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        shap_f = shap_values[i]

        for molecule_idx, shap_array in enumerate(shap_f):
            feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()
            best_shap_values = np.argsort(shap_array)[-top_i:][::-1]
            worst_shap_values = np.argsort(shap_array)[:top_i]
            # abs_shap_values = np.abs(shap_array)
            # top_10_indices = np.argsort(abs_shap_values)[-10:][::-1]
            top_indices = [idx for idx in np.concatenate((best_shap_values, worst_shap_values))]
            # top_10_indices = [idx for idx in top_10_indices if abs_shap_values[idx] != 0]
            top_10_feature_names = [feature_names[i] for i in top_indices]
            # top_10_feature_names = [feature_names[i] for i in top_10_indices]

            with open(smarts_mapping_path, 'r') as f:
                smarts_mapping = json.load(f)

            smarts_top10 = {
                (i, test_f.iloc[molecule_idx]['smiles'], feature): smarts_mapping[f'maccsfingerprint{int(feature.replace("maccsfingerprint", "")) + 1}'][0]
                for feature in top_10_feature_names
            }

            match_molecules = {s: [] for s in smarts_top10.keys()}
            molecules_statistics = {s: {"number_of_molecules_where_fingerprint": 0, "number_where_important": 0, "shap_value": 0} 
                                    for s in smarts_top10.keys()}
            
            for key, value in smarts_top10.items():
                non_zero_molecules = test_f[test_f[key[2]] == 1]
                non_zero_molecules = non_zero_molecules['smiles'].tolist()
                match_molecules[key].extend(non_zero_molecules)
                count_mol_with_fingerprint = len(non_zero_molecules)
                if key not in molecules_statistics_all:
                    molecules_statistics[key]["number_where_important"] = count_mol_with_fingerprint
                else:
                    molecules_statistics[key]["number_where_important"] = molecules_statistics_all[key]["number_where_important"] + count_mol_with_fingerprint
                molecules_statistics[key]["shap_value"] = shap_array[feature_names.index(key[2])]

            smarts_top_all.update(smarts_top10)
            match_molecules_all.update(match_molecules)
            molecules_statistics_all.update(molecules_statistics)

    return smarts_top_all, match_molecules_all, molecules_statistics_all


def process_folds_global(folds, data, shap_values, smarts_mapping_path, top_i=10):
    smarts_top_all = {}
    match_molecules_all = {}
    molecules_statistics_all = {}  # Initialize molecules_statistics_all

    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        shap_f = shap_values[i]

        feature_names = test_f.drop(columns=['capacity_max', 'smiles']).columns.tolist()
        mean_abs_shap_values = np.mean(np.abs(shap_f), axis=0)
        top_10_indices = np.argsort(mean_abs_shap_values)[-top_i:][::-1]
        top_10_indices = [idx for idx in top_10_indices if mean_abs_shap_values[idx] != 0]
        top_10_feature_names = [feature_names[i] for i in top_10_indices]

        with open(smarts_mapping_path, 'r') as f:
            smarts_mapping = json.load(f)

        smarts_top10 = {
            (i, match_molecule(feature,test_f), feature): smarts_mapping[f'maccsfingerprint{int(feature.replace("maccsfingerprint", "")) + 1}'][0]
            for feature in top_10_feature_names
        }

        match_molecules = {key: [] for key in smarts_top10.keys()}
        molecules_statistics = {s: {
            "number_of_molecules_where_fingerprint": 0,
            "number_where_important": 0,
            "shap_value": 0,
            "feature_in_smiles": True 
        } for s in smarts_top10.keys()}

        # for key, value in smarts_top10.items():
            # non_zero_molecules = test_f[test_f[key[2]] == 1]
            # non_zero_molecules = non_zero_molecules['smiles'].tolist()
            # # match_molecules[key].extend(non_zero_molecules)
            # count_mol_with_fingerprint = len(non_zero_molecules)
            # molecules_statistics[key]["number_where_important"] = len(non_zero_molecules)
            # if key not in molecules_statistics_all:
            #     molecules_statistics[key]["number_where_important"] = count_mol_with_fingerprint
            # else:
            #     molecules_statistics[key]["number_where_important"] = molecules_statistics_all[key]["number_where_important"] + count_mol_with_fingerprint
            
            # molecules_statistics[key]["shap_value"] = mean_abs_shap_values[feature_names.index(key[2])]
            # molecules_statistics[key]["feature_in_smiles"] = ''

        smarts_top_all.update(smarts_top10)
        match_molecules_all.update(match_molecules)
        molecules_statistics_all.update(molecules_statistics)  # Update molecules_statistics_all

    return smarts_top_all, match_molecules_all, molecules_statistics_all

def match_molecule(feature,test_f):
        non_zero_molecules = test_f[test_f[feature] == 1]
        non_zero_molecules = non_zero_molecules['smiles'].tolist()

        return non_zero_molecules[0] if non_zero_molecules else 'C'


def process_folds(folds, data, shap_values, smarts_mapping_path, local_explanation=True):
    if local_explanation:
        return process_folds_local(folds, data, shap_values, smarts_mapping_path)
    else:
        return process_folds_global(folds, data, shap_values, smarts_mapping_path)


if __name__ == '__main__':
    model = ['SHAP']#, 'SHAP_IQ'] # 'SHAP' or 'SHAP_IQ' - in the future it should be a list of models to run 
    # local_explanation = True
    # [mainXaiFlow(m, local_explanation) for m in model]
    local_explanation = False
    [mainXaiFlow(m, local_explanation) for m in model]