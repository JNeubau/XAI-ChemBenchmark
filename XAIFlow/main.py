import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import json
    
from AI_models.models import Models
from AI_models.eval_metrics import EvalMetrics
from utils.data_split import custom_data_kfold
from utils.exportlib import save_data_to_excel_with_highlights

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

from SHAP.shap_cross_validation import CrossValidationShapPipeline
from SHAP_IQ.shapiq_cross_validation import CrossValidationShapIqPipeline


def mainXaiFlow():
    model = 'SHAP_IQ'
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)
    
    maccs_fingerprints = os.path.join(parent_dir, 'data', 'maccs_merged.csv')
    smarts_mapping_path = os.path.join(parent_dir, 'data', 'maccs_smarts_mapping.json')
    results_dir = os.path.join(parent_dir, 'results', 'battery', model, datetime.today().strftime("%d-%m-%Y"))
    
    data = pd.read_csv(maccs_fingerprints, index_col=0)
    print(data.head())
    os.makedirs(results_dir, exist_ok=True)

    folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 10)
    # cv_pipeline = CrossValidationShapPipeline(
    cv_pipeline = CrossValidationShapIqPipeline(
        X=data.drop(columns=['capacity_max', 'smiles']),
        y=data[['capacity_max']],
        folds=folds,
        metrics=['smape', 'pairwise_accuracy_score', 'rmse', 'ndcg_score'],
        save_dir='',
        data_name='battery',
        verbose=True
    )

    # Train the model
    results, scores, shap_values = cv_pipeline.train_pipeline('XGBReg')
    print("Results:", results)

    # Explanations for the first fold
    test_f0 = data.loc[folds[0][1]]
    shap_f0 = shap_values[0]

    # Identify top 10 features
    feature_names = test_f0.drop(columns=['capacity_max', 'smiles']).columns.tolist()
    mean_abs_shap_values = np.mean(np.abs(shap_f0), axis=0)
    top_10_indices = np.argsort(mean_abs_shap_values)[-10:][::-1]
    top_10_feature_names = [feature_names[i] for i in top_10_indices]
    print("Top 10 Features:", top_10_feature_names)

    # Map features to SMARTS patterns
    with open(smarts_mapping_path, 'r') as f:
        smarts_mapping = json.load(f)

    smarts_top10 = {
        feature: smarts_mapping[f'maccsfingerprint{int(feature.replace("maccsfingerprint", "")) + 1}'][0]
        for feature in top_10_feature_names
    }
    print("SMARTS Top 10:", smarts_top10)

    # Match molecules to SMARTS patterns
    match_molecules = {s: [] for s in smarts_top10.keys()}
    for key, value in smarts_top10.items():
        non_zero_molecules = test_f0[test_f0[key] == 1]
        non_zero_molecules = non_zero_molecules['smiles'].tolist()
        match_molecules[key].extend(non_zero_molecules)
        
    excel_data, smiles_list, smarts_list = prepare_data_for_excel_export(match_molecules, smarts_top10)
    save_to_excel(excel_data, smiles_list, smarts_list, results_dir)


def prepare_data_for_excel_export(match_molecules, smarts_top10):
    excel_data = {
        "Feature": [],
        "SMARTS": [],
        "Molecule": []
    }
    smiles_list = []
    smarts_list = []

    for key, smarts in smarts_top10.items():
        for molecule in match_molecules[key]:
            excel_data["Feature"].append(key)
            excel_data["SMARTS"].append(smarts)
            excel_data["Molecule"].append(molecule)
            smiles_list.append(molecule)
            smarts_list.append(smarts)
    return excel_data, smiles_list, smarts_list


def save_to_excel(excel_data, smiles_list, smarts_list, results_dir):
    excel_output_path = results_dir + f'\\molecule_results_with_highlights_{datetime.now().strftime("%H-%M-%S")}.xlsx'
    save_data_to_excel_with_highlights(excel_data, smiles_list, smarts_list, excel_output_path)
    print(f"Molecule results with highlights saved to {excel_output_path}")


if __name__ == '__main__':
    mainXaiFlow()