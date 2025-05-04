import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import exmol
import random
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

from utils.data_split import custom_data_kfold
from utils.exportlib import save_data_to_excel_with_highlights, save_scores_to_excel_new_sheet
from MMACE.mmace_cross_validation_pipeline import CrossValidationMMACEPipeline


def mainMMACEFlow():
    np.random.seed(0)
    random.seed(0)
    print("Running MMACE explanation pipeline...")
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)

    maccs_fingerprints = os.path.join(parent_dir, 'data', 'maccs_merged.csv')
    results_dir = os.path.join(parent_dir, 'results', 'battery', 'MMACE', 'local', datetime.today().strftime("%d-%m-%Y"))

    data = pd.read_csv(maccs_fingerprints, index_col=0)
    print(data.head())
    os.makedirs(results_dir, exist_ok=True)

    folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 5)

    print(f"Number of folds: {len(folds)}")
    print(f"Number of molecules: {len(data)}")

    cv_pipeline = CrossValidationMMACEPipeline(
        X=data.drop(columns=['capacity_max', 'smiles']),
        y=data[['capacity_max']],
        z=data[['smiles']],
        folds=folds,
        metrics=['smape', 'pairwise_accuracy_score', 'rmse', 'ndcg_score'],
        save_dir=results_dir,
        data_name='battery',
        verbose=True
    )

    results, scores, MMACE_explanations,cfs,samples = cv_pipeline.train_pipeline('RFReg')
    print("Results:", results)
    print("Scores:", scores)
    print("MMACE explanations:", MMACE_explanations)

    process_folds_local(folds, data, MMACE_explanations, samples)
    # smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds_local_MMACE(
    #     folds, data, MMACE_explanations, top_i=5
    # )
    #
    # scores_data = create_dataframe_from_scores(scores, results)
    # save_scores_to_excel(scores_data, results_dir)
    #
    # excel_data = prepare_data_for_excel_export(match_molecules_all, smarts_top_all, molecules_statistics_all)
    # save_molecules_to_excel(excel_data, results_dir)

def process_folds_local(folds,data, MMACE_explanation,samples):
    print("Processing folds for local MMACE...")

    export_plots_exmol(samples)
    print("Exported plots for exmol explanations.")

def export_plots_exmol(sample_space):
    # exmol.plot_descriptors(sample_space)
    # plt.savefig("my_descriptor_plot.png", bbox_inches="tight")
    # plt.close()
    return 0

# def process_folds_local_MMACE(folds, data, MMACE_explanations, top_i=5):
    # smarts_top_all = {}
    # match_molecules_all = {}
    # molecules_statistics_all = {}

    # for i, fold in enumerate(folds):
    #     test_f = data.loc[fold[1]]
    #     MMACE_fold_explanations = MMACE_explanations[i]["explanations"]

    #     for molecule_idx, explanation_dict in enumerate(MMACE_fold_explanations):
    #         feature_importances = explanation_dict["explanation"].as_list()
    #         top_features = sorted(feature_importances, key=lambda x: abs(x[1]), reverse=True)[:top_i]

    #         smarts_top10 = {
    #             (i, test_f.iloc[molecule_idx]['smiles'], feature[0]): feature[1]
    #             for feature in top_features
    #         }

    #         match_molecules = {s: [] for s in smarts_top10.keys()}
    #         molecules_statistics = {s: {
    #             "number_of_molecules_where_fingerprint": 0,
    #             "number_where_important": 0,
    #             "exmol_explanations": explanation_dict["explanation"],
    #             "feature_in_smiles": False,
    #             "capacity_max": test_f.iloc[molecule_idx]['capacity_max'],
    #             "capacity_pred": 0
    #         } for s in smarts_top10.keys()}

    #         for key in smarts_top10.keys():
    #             feature_name = key[2]
    #             non_zero_molecules = test_f[test_f[feature_name] == 1]
    #             non_zero_molecules = non_zero_molecules['smiles'].tolist()
    #             match_molecules[key].extend(non_zero_molecules)
    #             molecules_statistics[key]["feature_in_smiles"] = bool(
    #                 data.loc[data['smiles'] == key[1], feature_name].values[0] == 1
    #             )

    #         smarts_top_all.update(smarts_top10)
    #         match_molecules_all.update(match_molecules)
    #         molecules_statistics_all.update(molecules_statistics)

    # return smarts_top_all, match_molecules_all, molecules_statistics_all


# def count_molecules_with_fingerprint(maccs_fingerprints_data, molecules_statistics_all):
#     column_ones_count = maccs_fingerprints_data.sum(axis=0).to_dict()
#     for key in molecules_statistics_all.keys():
#         column_name = key[2]
#         if column_name in column_ones_count:
#             molecules_statistics_all[key]["number_of_molecules_where_fingerprint"] = column_ones_count[column_name]
#     return molecules_statistics_all


# def count_important_features(data, molecules_statistics_all):
#     feature_importance_count = {col: 0 for col in data.columns}

#     for key in molecules_statistics_all.keys():
#         feature_key = key[2]
#         if feature_key in feature_importance_count:
#             feature_importance_count[feature_key] += 1

#     for key in molecules_statistics_all.keys():
#         feature_key = key[2]
#         molecules_statistics_all[key]["number_where_important"] = feature_importance_count.get(feature_key, 0)
#     return molecules_statistics_all


# def prepare_data_for_excel_export(match_molecules, smarts_top, molecules_statistics_all):
#     excel_data = {
#         "Fold_No": [],
#         "Smiles_key": [],
#         "Feature_key": [],
#         "SMARTS": [],
#         "Molecule": [],
#         "number_of_molecules_where_fingerprint": [],
#         "Number_where_important": [],
#         'feature_in_smiles': [],
#         "ExMol_Explanations": [],
#         "Capacity Max": [],
#         "Capacity Pred": [],
#     }

#     for key, smarts in smarts_top.items():
#         excel_data["Fold_No"].append(key[0])
#         excel_data["Smiles_key"].append(key[1])
#         excel_data["Feature_key"].append(key[2])
#         excel_data["SMARTS"].append(smarts)
#         excel_data["Molecule"].append(key[1])
#         excel_data["number_of_molecules_where_fingerprint"].append(molecules_statistics_all[key]["number_of_molecules_where_fingerprint"])
#         excel_data["Number_where_important"].append(molecules_statistics_all[key]["number_where_important"])
#         excel_data["feature_in_smiles"].append(molecules_statistics_all[key]["feature_in_smiles"])
#         excel_data["ExMol_Explanations"].append(molecules_statistics_all[key]["exmol_explanations"])
#         excel_data["Capacity Max"].append(molecules_statistics_all[key]["capacity_max"])
#         excel_data["Capacity Pred"].append(molecules_statistics_all[key]["capacity_pred"])

#     return excel_data


# def save_molecules_to_excel(excel_data, results_dir):
#     results_path = os.path.join(results_dir, f'molecule_results_with_highlights_{datetime.now().strftime("%H-%M-%S")}.xlsx')
#     save_data_to_excel_with_highlights(excel_data, results_path)
#     print(f"Molecule results with highlights saved to {results_path}")


# def save_scores_to_excel(scores_data, results_dir):
#     results_path = os.path.join(results_dir, f'molecule_scores_{datetime.now().strftime("%H-%M-%S")}.xlsx')
#     save_scores_to_excel_new_sheet(scores_data, results_path)
#     print(f"Scores saved to {results_path}")


# def create_dataframe_from_scores(scores, results):
#     df_scores = pd.DataFrame(scores)
#     for key, value in results.items():
#         df_scores.loc["Final", key] = value
#     return df_scores


if __name__ == '__main__':
    mainMMACEFlow()
