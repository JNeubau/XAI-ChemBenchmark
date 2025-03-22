import os
from datetime import datetime
import pandas as pd
import numpy as np
# import shap
from models import Models
from eval_metrics import EvalMetrics
from data_split import custom_data_kfold
from cross_validation import CrossValidationPipeline
import shap
from exportlib import save_data_to_excel

def main ():
    print('Hello, world!')
    # Load data
    data = pd.read_csv('./data/maccs_merged.csv', index_col=0)
    print(data.head())

    date = datetime.today().strftime("%d-%m-%Y")
    os.makedirs(f'./results/battery/{date}', exist_ok=True)
    print(date)
    folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 10)
    cv_pipeline = CrossValidationPipeline(
        X=data.drop(columns=['capacity_max', 'smiles']),
        y=data[['capacity_max']],
        folds=folds,
        metrics=['smape', 'pairwise_accuracy_score', 'rmse', 'ndcg_score'],
        save_dir='',
        data_name='battery',
        verbose=True
    )

    # # Create results directory
    # date = datetime.today().strftime("%d-%m-%Y")
    # os.makedirs(f'./results/battery/{date}', exist_ok=True)

    # # Prepare data splits
    # folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 10)

    # # Initialize cross-validation pipeline
    # cv_pipeline = CrossValidationPipeline(
    #     X=data.drop(columns=['capacity_max', 'smiles']),
    #     y=data[['capacity_max']],
    #     folds=folds,
    #     metrics=['smape', 'pairwise_accuracy_score', 'rmse', 'ndcg_score'],
    #     save_dir='',
    #     data_name='battery',
    #     verbose=True
    # )

    # Train the model
    results, scores, shap_values = cv_pipeline.train_pipeline('XGBReg')

    # Display results
    print("Results:", results)

    # Explanations for the first fold
    test_f0 = data.loc[folds[0][1]]
    shap_f0 = shap_values[0]

    # # Visualize SHAP summary plot
    # shap.summary_plot(
    #     shap_f0,
    #     test_f0.drop(columns=['capacity_max', 'smiles']),
    #     max_display=10,
    #     show=True
    # )

    # Identify top 10 features
    feature_names = test_f0.drop(columns=['capacity_max', 'smiles']).columns.tolist()
    mean_abs_shap_values = np.mean(np.abs(shap_f0), axis=0)
    top_10_indices = np.argsort(mean_abs_shap_values)[-10:][::-1]
    top_10_feature_names = [feature_names[i] for i in top_10_indices]
    print("Top 10 Features:", top_10_feature_names)

    # Map features to SMARTS patterns
    import json
    with open('data/maccs_smarts_mapping.json', 'r') as f:
        smarts_mapping = json.load(f)

    smarts_top10 = {
        feature: smarts_mapping[f'maccsfingerprint{int(feature.replace("maccsfingerprint", "")) + 1}'][0]
        for feature in top_10_feature_names
    }
    print("SMARTS Top 10:", smarts_top10)

    # Match molecules to SMARTS patterns
    from rdkit import Chem
    from rdkit.Chem import Draw

    match_molecules = {s: [] for s in smarts_top10.keys()}
    for key, value in smarts_top10.items():
        non_zero_molecules = test_f0[test_f0[key] == 1]
        non_zero_molecules = non_zero_molecules['smiles'].tolist()
        match_molecules[key].extend(non_zero_molecules)

    # Prepare data for Excel export
    excel_data = {
        "Feature": [],
        "SMARTS": [],
        "Molecule": []
    }
    for key, smarts in smarts_top10.items():
        for molecule in match_molecules[key]:
            excel_data["Feature"].append(key)
            excel_data["SMARTS"].append(smarts)
            excel_data["Molecule"].append(molecule)

    # Save to Excel using exportlib
    excel_output_path = f'./results/battery/{date}/molecule_results.xlsx'
    save_data_to_excel(excel_data, excel_data["Molecule"], excel_output_path)
    print(f"Molecule results saved to {excel_output_path}")



if __name__ == '__main__':
    main()