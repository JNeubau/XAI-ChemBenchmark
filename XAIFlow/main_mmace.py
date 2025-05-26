import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import selfies as sf
import exmol
import random
import matplotlib.pyplot as plt
from rdkit import Chem
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

from utils.data_split import load_fold_indices
from MMACE.mmace_cross_validation_pipeline import CrossValidationMMACEPipeline
from MMACE.savemmacecfexcel import save_mmace_explanations_to_excel, create_mmace_pdf #,save_mmace_explanations_to_excel_new


def mainMMACEFlow(experiment_name = 'battery', seed=42):
    np.random.seed(seed)
    random.seed(seed)
    print("Running MMACE explanation pipeline...")
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)
    

    maccs_fingerprints = os.path.join(parent_dir, 'data', 'new_maccs_merged.csv')
    # maccs_fingerprints = os.path.join(parent_dir, 'data', 'maccs_merged.csv')
    results_dir = os.path.join(parent_dir, 'results', experiment_name, 'MMACE', 'local', datetime.today().strftime("%d-%m-%Y"))
    folds_dir = os.path.join(parent_dir, 'RFReg', experiment_name, 'folds')

    data = pd.read_csv(maccs_fingerprints)
    print(data.head())
    os.makedirs(results_dir, exist_ok=True)
    # custom_alphabet = get_custom_alphabet(data)
    custom_alphabet = get_basic_alphabet()
    print(f"Custom alphabet: {custom_alphabet}")

    folds = load_fold_indices(folds_dir)
    # folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 5)

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

    cfs, samples, MMACE_Explanations = cv_pipeline.load_pipeline(os.path.join(parent_dir, 'RFReg', experiment_name, 'ckpt'))
    # results, scores,cfs,samples,MMACE_Explanations = cv_pipeline.train_pipeline('RFReg')
    # print("Results:", results)
    # print("Scores:", scores)

    save_mmace_explanations_to_excel(MMACE_Explanations, results_dir=results_dir)
   
    process_folds(folds, data, samples, cfs,MMACE_Explanations,results_dir)

def get_custom_alphabet(data):
    """
    Generate custom alphabet from SMILES data.
    :param data: DataFrame containing SMILES strings.
    """
    
    selfies_list = []
    for s in data.smiles:  # Changed from SMILES to smiles to match your DataFrame
        try:
            selfies_list.append(sf.encoder(exmol.sanitize_smiles(s)[1]))
        except sf.EncoderError:
            selfies_list.append(None)
    
    custom_alphabet = sf.get_alphabet_from_selfies([s for s in selfies_list if s is not None])
    return custom_alphabet

def get_basic_alphabet():
    """
    Get basic alphabet for SELFIES.
    """

    return exmol.get_basic_alphabet()

def process_folds(folds, data, samples, cfs, MMACE_Explanations, results_dir):
    """Process folds for local MMACE explanations."""
    print("Processing folds for local MMACE...")
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    feature_importance_per_fold = []
    
    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        mmace_cf = MMACE_Explanations[i]["explanations"]
        
        # Analyze feature importance for each instance in the fold
        fold_importance = {}
        for idx, counterfactuals in enumerate(mmace_cf):
            if not counterfactuals:  # Skip if no counterfactuals
                continue
                
            original = test_f.iloc[idx]
            original_features = original.drop(['capacity_max', 'smiles'])
            
            # Calculate feature importance for each counterfactual
            feature_impacts = []
            for cf in counterfactuals:
                if not hasattr(cf, 'smiles'):  # Skip invalid counterfactuals
                    continue
                    
                # Get MACCS fingerprints for counterfactual
                cf_mol = Chem.MolFromSmiles(cf.smiles)
                if cf_mol is None:
                    continue
                # Generate MACCS fingerprint DataFrame for the counterfactual molecule
                fps = [list(Chem.MACCSkeys.GenMACCSKeys(cf_mol).ToBitString())]
                fps = np.array(fps)[:, 1:]
                fps_df = pd.DataFrame(fps, columns=[f'maccsfingerprint{i}' for i in range(fps.shape[1])])

                parent_dir = os.path.dirname(os.getcwd())
                maccs_merge_path = os.path.join(parent_dir, 'data', 'new_maccs_merged.csv')
                # print(f"Using MACCS merge file: {maccs_merge_path}")

                if not os.path.exists(maccs_merge_path):
                    raise FileNotFoundError(f"MACCS merge file not found: {maccs_merge_path}")

                maccs_merge = pd.read_csv(maccs_merge_path)
                maccs_merge = maccs_merge.loc[:, maccs_merge.columns.str.contains('maccs', case=False)]
                selected_keys = maccs_merge.columns.tolist()
                selected_keys = [key for key in selected_keys if key in fps_df.columns]
                filtered_fps = fps_df[selected_keys]

                # Align cf_features with original_features
                cf_features = filtered_fps.iloc[0].astype(int)
                # print(f"Counterfactual features: {cf_features}")
                # print(f"Original features: {original_features}")

                # Calculate feature differences and their impact
                feature_diff = cf_features - original_features
                prediction_diff = cf.yhat - original['capacity_max']

                # print(f"Prediction difference: {prediction_diff}")
                # print(f"Feature differences: {feature_diff}")
                
                # Record impact for changed features
                for feat, diff in feature_diff.items():
                    if diff != 0:
                        impact = prediction_diff #* diff
                        feature_impacts.append((feat, impact))
            
            # Aggregate feature impacts
            if feature_impacts:
                for feat, impact in feature_impacts:
                    if feat not in fold_importance:
                        fold_importance[feat] = []
                    fold_importance[feat].append(impact)
        
        # Calculate mean impact for each feature
        mean_importance = {feat: np.mean(impacts) for feat, impacts in fold_importance.items()}
        
        # Get top features
        top_features = sorted(mean_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        feature_importance_per_fold.append(top_features)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        features, importances = zip(*top_features)
        plt.barh([f"{f} ({i:.3f})" for f, i in zip(features, importances)], 
                [abs(i) for i in importances])
        plt.title(f'Top Feature Importance - Fold {i}')
        plt.xlabel('|Impact on Prediction|')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'feature_importance_fold_{i}.png'))
        plt.close()
        
        # # Create PDF report
        pdf_path = create_mmace_pdf(MMACE_Explanations, i, results_dir)
        print(f"Created PDF report for fold {i} at: {pdf_path}")
    
    # Save overall feature importance to CSV
    importance_df = pd.DataFrame([{f: i for f, i in fold} 
                                for fold in feature_importance_per_fold])
    importance_df.to_csv(os.path.join(results_dir, 'feature_importance.csv'))
    return feature_importance_per_fold
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run XAI Flow with specified parameters')
    parser.add_argument('--experiment_name', type=str, default='test', help='Name of the experiment (default: test)')
    parser.add_argument('--model', type=str, default='MMACE', help='Model to use (default: MMACE)')
    parser.add_argument('--seed', type=int, default=42, help='Set seed value (default: 42)')
    
    args = parser.parse_args()
    
    print(f"\n=== Running {args.model} ===\n")
    print("Arguments:", vars(args))
    # experiment_name = 'rf_test'
    mainMMACEFlow(experiment_name=args.experiment_name,
                  seed=args.seed)
