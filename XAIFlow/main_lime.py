import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import exmol
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

from utils.data_split import custom_data_kfold
from utils.exportlib import save_data_to_excel_with_highlights, save_scores_to_excel_new_sheet
from LIME.lime_cross_validation import CrossValidationLIMEPipeline


def mainLimeFlow():
    print("Running LIME explanation pipeline...")
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)

    maccs_fingerprints = os.path.join(parent_dir, 'data', 'maccs_merged.csv')
    results_dir = os.path.join(parent_dir, 'results', 'battery', 'LIME', 'local', datetime.today().strftime("%d-%m-%Y"))

    data = pd.read_csv(maccs_fingerprints, index_col=0)
    print(data.head())
    os.makedirs(results_dir, exist_ok=True)

    folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 5)

    print(f"Number of folds: {len(folds)}")
    print(f"Number of molecules: {len(data)}")

    cv_pipeline = CrossValidationLIMEPipeline(
        X=data.drop(columns=['capacity_max', 'smiles']),
        y=data[['capacity_max']],
        z=data[['smiles']],
        folds=folds,
        metrics=['smape', 'pairwise_accuracy_score', 'rmse', 'ndcg_score'],
        save_dir=results_dir,
        data_name='battery',
        verbose=True
    )

    results, scores, lime_explanations,cfs,samples = cv_pipeline.train_pipeline('RFReg')
    # print("Results:", results)
    # print("Scores:", scores)
    # print("LIME explanations:", lime_explanations)

    process_folds_local(folds, data, lime_explanations, samples)
    # smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds_local_lime(
    #     folds, data, lime_explanations, top_i=5
    # )
    #
    # scores_data = create_dataframe_from_scores(scores, results)
    # save_scores_to_excel(scores_data, results_dir)
    #
    # excel_data = prepare_data_for_excel_export(match_molecules_all, smarts_top_all, molecules_statistics_all)
    # save_molecules_to_excel(excel_data, results_dir)

def process_folds_local(folds, data, lime_explanations, samples):
    print("Processing folds for local LIME...")

    # Export plots for ExMol explanations
    export_plots_exmol(samples)
    print("Exported plots for ExMol explanations.")

    # Save LIME explanations to Excel
    save_lime_explanations_to_excel(folds, data, lime_explanations)
    print("Saved LIME explanations to Excel.")


def save_lime_explanations_to_excel(folds, data, lime_explanations):
    """
    Save LIME explanations to an Excel file for the given SMILES.
    :param folds: List of cross-validation folds.
    :param data: Original dataset.
    :param lime_explanations: LIME explanations for each fold.
    """
    results_dir = os.path.join(os.getcwd(), "lime_results")
    os.makedirs(results_dir, exist_ok=True)

    excel_data = {
        "Fold": [],
        "SMILES": [],
        "Descriptor Type": [],
        "Beta Coefficients": [],
    }

    for fold_idx, fold_explanations in enumerate(lime_explanations):
        test_indices = folds[fold_idx][1]
        test_smiles = data.iloc[test_indices]["smiles"].values

        # Ensure the number of explanations matches the number of test SMILES
        num_explanations = len(fold_explanations["explanations"])
        num_test_smiles = len(test_smiles)

        if num_explanations != num_test_smiles:
            print(f"Warning: Mismatch in explanations and test SMILES for fold {fold_idx}.")
            min_length = min(num_explanations, num_test_smiles)
        else:
            min_length = num_explanations

        for i in range(min_length):
            explanation = fold_explanations["explanations"][i]
            excel_data["Fold"].append(fold_idx)
            excel_data["SMILES"].append(test_smiles[i])
            excel_data["Descriptor Type"].append(explanation["descriptor_type"])
            excel_data["Beta Coefficients"].append(explanation["beta"])

    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(excel_data)
    excel_path = os.path.join(results_dir, f"lime_explanations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"LIME explanations saved to {excel_path}")

def export_plots_exmol(sample_space):
    # exmol.plot_descriptors(sample_space)
    # plt.savefig("my_descriptor_plot.png", bbox_inches="tight")
    # plt.close()
    return 0

# def process_folds_local_lime(folds, data, lime_explanations, top_i=5):
    # smarts_top_all = {}
    # match_molecules_all = {}
    # molecules_statistics_all = {}

    # for i, fold in enumerate(folds):
    #     test_f = data.loc[fold[1]]
    #     lime_fold_explanations = lime_explanations[i]["explanations"]

    #     for molecule_idx, explanation_dict in enumerate(lime_fold_explanations):
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
    mainLimeFlow()
