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
from MMACE.timeoutexception import timeout
from MMACE.savemmacecfexcel import save_mmace_explanations_to_excel,create_mmace_pdf


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
    custom_alphabet = get_custom_alphabet(data)


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
        verbose=True,
        custom_alphabet=custom_alphabet  
    )

    results, scores,cfs,samples,MMACE_Explanations = cv_pipeline.train_pipeline('RFReg')
    print("Results:", results)
    print("Scores:", scores)

    save_mmace_explanations_to_excel(MMACE_Explanations, results_dir=results_dir)
    # print("MMACE Explanations:", MMACE_Explanations)
   
    process_folds_local(folds, data, samples, cfs,MMACE_Explanations,results_dir)


    # print("MMACE explanations:", MMACE_explanations)

    # process_folds_local(folds, data, samples,cfs)
    # smarts_top_all, match_molecules_all, molecules_statistics_all = process_folds_local_MMACE(
    #     folds, data, MMACE_explanations, top_i=5
    # )
    #
    # scores_data = create_dataframe_from_scores(scores, results)
    # save_scores_to_excel(scores_data, results_dir)
    #
    # excel_data = prepare_data_for_excel_export(match_molecules_all, smarts_top_all, molecules_statistics_all)
    # save_molecules_to_excel(excel_data, results_dir)
def get_custom_alphabet(data):
    """
    Generate custom alphabet from SMILES data.
    :param data: DataFrame containing SMILES strings.
    """
    import selfies as sf
    import exmol
    
    selfies_list = []
    for s in data.smiles:  # Changed from SMILES to smiles to match your DataFrame
        try:
            selfies_list.append(sf.encoder(exmol.sanitize_smiles(s)[1]))
        except sf.EncoderError:
            selfies_list.append(None)
    
    custom_alphabet = sf.get_alphabet_from_selfies([s for s in selfies_list if s is not None])
    return custom_alphabet

def process_folds_local(folds,data,samples,cfs,MMACE_Explanations,results_dir):
    """
    Process folds for local MMACE explanations.
    :param folds: List of folds for cross-validation.
    :param data: DataFrame containing the dataset.
    :param samples: List of samples for each fold.
    :param cfs: List of counterfactuals for each fold.
    :param MMACE_Explanations: List of MMACE explanations for each fold.
    """
    print("Processing folds for local MMACE...")
    smarts_top_all = {}
    match_molecules_all = {}
    molecules_statistics_all = {}
    # print(f"Number of folds: {len(folds)}")
    # print(f"Number of molecules: {len(data)}")
    # print(f"Number of samples: {len(samples)}")
    # print(f"Number of MMACE explanations: {len(cfs)}")
    # print(f"MMACE explanations: {cfs}") 
    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        mmace_cf = MMACE_Explanations[i]["explanations"]
        pdf_path = create_mmace_pdf(MMACE_Explanations, i, results_dir)
        print(f"Created PDF report for fold {i} at: {pdf_path}")
        # samples_fold = samples[i]
        # # print(f"test_f :{test_f}")
        # print(F"samples_fold :{samples_fold}")
        # print(f"mmace_cf :{mmace_cf}")
        
        # print(f"length of mmace_cf: {len(mmace_cf)}")
        # print(f"length of samples_fold: {len(samples_fold)}")
        # for molecule_idx, cf_array in enumerate(mmace_cf):
            # print(f"=============================\n Fold {i}, Molecule {molecule_idx}")
            # print(f"SMILES: {test_f.iloc[molecule_idx]['smiles']}")
            # print(f"cfs_array: {cf_array}")
            # print("CFS array:", cfs_array)
            # if i == 0 and molecule_idx==0:
                # print("MMACE CF:", cfs_array)
            # if len(cf_array) == 0:
            #     print("No CFS found for this molecule.")
            #     continue
            # if cfs_array is not None:
            # export_plots_exmol(cfs_array,i,molecule_idx)
            # print("Exported plots for exmol explanations.")

# @timeout(60)
# def plot_with_timeout(cfs):
#     fkw = {"figsize": (10, 3)}
#     exmol.plot_cf(cfs, figure_kwargs=fkw, mol_size=(450, 400), nrows=1)

# def export_plots_exmol(cfs, fold, i):
    
#     datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     try:
#         # plot_path = os.path.join(os.path.join(
#         #         os.path.dirname(os.getcwd())), 'results', 'plots', 'MMACE', "local", datetime.today().strftime("%d-%m-%Y"))
#         plot_dir = os.path.join(
#                 os.path.dirname(os.getcwd()), 'results', 'plots', 'MMACE', "local", datetime.today().strftime("%d-%m-%Y")
#             )
#         os.makedirs(plot_dir, exist_ok=True)
#         plot_path = os.path.join(
#                 plot_dir, f"explanation_fold_{fold}_instance_{i}_{datetime_now}.png"
#             )
#         # fkw = {"figsize": (10, 3)}
#         print(f"fkw")
#         plot_with_timeout(cfs)
#         print(f"exmol plot_cf")
#         plt.savefig(plot_path, bbox_inches="tight", dpi=180)
#         print(f"Plot saved to {plot_path}")
#     except Exception as e:
#         print(f"An error occurred while plotting CFS: {e}")
#     finally:
#         plt.close('all')
    
#     # exmol.plot_descriptors(sample_space)
#     # plt.savefig("my_descriptor_plot.png", bbox_inches="tight")
#     # plt.close()
#     return 0

# def export_plots_exmol_space(space,cfs, fold, i):
#     datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     try:
#         plot_path = os.path.join(
#                 os.path.dirname(os.getcwd()),
#                 f"explanation_space_fold_{fold}_instance_{i}_{datetime_now}.png"
#             )
#         # fkw = {"figsize": (10, 3)}
#         print(f"fkw")
#         fkw = {"figsize": (8, 6)}
#         font = {"family": "normal", "weight": "normal", "size": 22}


#         exmol.plot_space(space, cfs, figure_kwargs=fkw, mol_size=(200, 200), offset=1)
#         ax = plt.gca()
#         plt.colorbar(ax.get_children()[1], ax=[ax], location="left", label="Solubility [Log M]")
#         plt.savefig(plot_path, bbox_inches="tight", dpi=180)
#         print(f"Plot saved to {plot_path}")
#     except Exception as e:
#         print(f"An error occurred while plotting CFS: {e}")
#     finally:
#         plt.close('all')
#     return 0

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
