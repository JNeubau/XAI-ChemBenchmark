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


def mainMMACEFlow(experiment_name = 'battery', seed=42, explanation_value_mode="per_feature"):
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
   
    process_folds(folds, data, samples, cfs, MMACE_Explanations, results_dir, explanation_value_mode=explanation_value_mode)

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

def export_mmace_feature_changes_to_excel(cf_change_rows, results_dir):
    """
    Export MMACE counterfactual feature changes to an Excel file with a similar format to exportlib.
    Each row contains: original/cf SMILES, changed feature, original/cf prediction, prediction difference, and all features.
    """
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Draw
    from io import BytesIO

    if not cf_change_rows:
        print("No counterfactual feature changes to export.")
        return

    df = pd.DataFrame(cf_change_rows)
    excel_file = os.path.join(results_dir, "mmace_cf_molecule_results_with_highlights_all_features.xlsx")

    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data", startrow=0, startcol=6)
        worksheet = writer.sheets["Data"]

        # for i, row in df.iterrows():
        #     smiles_orig = row['SMILES_original']
        #     smiles_cf = row['SMILES_cf']
        #     # Original molecule image
        #     mol_orig = Chem.MolFromSmiles(smiles_orig)
        #     if mol_orig:
        #         img_orig = Draw.MolToImage(mol_orig)
        #         img_buffer_orig = BytesIO()
        #         img_orig.save(img_buffer_orig, format='PNG')
        #         img_buffer_orig.seek(0)
        #         worksheet.insert_image(i + 1, 0, '', {'image_data': img_buffer_orig})
        #     # Counterfactual molecule image
        #     mol_cf = Chem.MolFromSmiles(smiles_cf)
        #     if mol_cf:
        #         img_cf = Draw.MolToImage(mol_cf)
        #         img_buffer_cf = BytesIO()
        #         img_cf.save(img_buffer_cf, format='PNG')
        #         img_buffer_cf.seek(0)
        #         worksheet.insert_image(i + 1, 3, '', {'image_data': img_buffer_cf})
        #     worksheet.set_row(i + 1, 250)

        # worksheet.set_column('A:A', 20)  # Original molecule column
        # worksheet.set_column('D:D', 20)  # Counterfactual molecule column
        # worksheet.set_column('G:ZZ', 15)  # Data columns

    print(f"MMACE counterfactual feature changes exported to: {excel_file}")

def process_folds(folds, data, samples, cfs, MMACE_Explanations, results_dir, explanation_value_mode="per_feature"):
    """Process folds for local MMACE explanations."""
    print("Processing folds for local MMACE...")
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    feature_importance_per_fold = []
    
    # Collect per-counterfactual feature change info for CSV
    cf_change_rows = []

    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        mmace_cf = MMACE_Explanations[i]["explanations"]
        
        # Analyze feature importance for each instance in the fold
        fold_importance = {}
        for idx, counterfactuals in enumerate(mmace_cf):
            if not counterfactuals:  # Skip if no counterfactuals
                continue
                
            original = test_f.iloc[idx]
            original_smmiles = Chem.MolFromSmiles(original['smiles'])

            # Generate MACCS fingerprint DataFrame for the original molecule
            original_fps = [list(Chem.MACCSkeys.GenMACCSKeys(original_smmiles).ToBitString())]
            original_fps = np.array(original_fps)[:, 1:]
            original_fps_df = pd.DataFrame(original_fps, columns=[f'maccsfingerprint{i}' for i in range(original_fps.shape[1])])
            original_features = original_fps_df.iloc[0].astype(int)

            # Calculate feature importance for each counterfactual
            feature_impacts = []
            # print(f"==================\nProcessing fold {i}, instance {idx} with {len(counterfactuals)} counterfactuals\n============")
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


                # Align cf_features with original_features
                cf_features = fps_df.iloc[0].astype(int)

                # Calculate feature differences and their impact
                feature_diff = cf_features - original_features
                prediction_org = float(original['capacity_max'])
                prediction_cf = float(cf.yhat)
                prediction_diff = prediction_cf - prediction_org

                # get number of features changed
                num_features_changed = sum(feature_diff != 0)
                print(num_features_changed)

                # Magnitude-based attribution
                if explanation_value_mode == "magnitude":
                    abs_deltas = {feat: abs(diff) for feat, diff in feature_diff.items() if diff != 0}
                    total_change = sum(abs_deltas.values())
                    for feat, diff in feature_diff.items():
                        # print(f"Feature: {feat}, Diff: {diff}, Total Change: {total_change}")
                        if diff != 0:
                            weight = abs(diff) / total_change if total_change != 0 else 0
                            explanation_value = weight * prediction_diff
                            feature_impacts.append((feat, explanation_value))
                            cf_change_rows.append({
                                "Fold_no": i,
                                "SMILES_original": original['smiles'],
                                "SMILES": str(original['smiles']),
                                "SMILES_cf": str(cf.smiles),
                                "Feature_key": feat,
                                "Prediction_original": prediction_org,
                                "Prediction_cf": prediction_cf,
                                'Prediction_difference': prediction_diff,
                                "Explanation_value": np.abs(explanation_value),
                                'Explanation_sign': 'Positive' if explanation_value > 0 else 'Negative',
                                'AddedRemoved': diff,
                                'Model': 'MMACE',
                                'features_original': original_features.to_dict(),
                                'features_cf': cf_features.to_dict()
                            })
                else:
                    # Per-feature method
                    for feat, diff in feature_diff.items():
                        if diff != 0:
                            impact = prediction_diff #* diff
                            feature_impacts.append((feat, impact))
                            cf_change_rows.append({
                                "Fold_no": i,
                                "SMILES_original": original['smiles'],
                                "SMILES": str(original['smiles']),
                                "SMILES_cf": str(cf.smiles),
                                "Feature_key": feat,
                                "Prediction_original": prediction_org,
                                "Prediction_cf": prediction_cf,
                                "Prediction_difference": prediction_diff,
                                'Explanation_value': np.abs(prediction_cf/num_features_changed) if num_features_changed > 0 else 0, 
                                'Explanation_sign': 'positive' if explanation_value > 0 else 'negative',
                                'AddedRemoved': diff,
                                'Model': 'MMACE',
                                'features_original': original_features.to_dict(),
                                'features_cf': cf_features.to_dict()
                            })
            
            # Aggregate feature impacts
            if feature_impacts:
                for feat, impact in feature_impacts:
                    if feat not in fold_importance:
                        fold_importance[feat] = []
                    fold_importance[feat].append(impact)
        
        # Calculate mean impact for each feature
        mean_importance = {feat: np.mean(impacts) for feat, impacts in fold_importance.items()}
        # print(f"Mean importance for fold {i}: {mean_importance}")
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
        # pdf_path = create_mmace_pdf(MMACE_Explanations, i, results_dir)
        # print(f"Created PDF report for fold {i} at: {pdf_path}")
    
    # Save overall feature importance to CSV
    importance_df = pd.DataFrame([{f: i for f, i in fold} 
                                for fold in feature_importance_per_fold])
    importance_df.to_csv(os.path.join(results_dir, 'feature_importance.csv'))
    
    # Save the CSV with per-counterfactual feature changes
    if cf_change_rows:
        cf_change_df = pd.DataFrame(cf_change_rows)
        csv_path = os.path.join(results_dir, "mmace_cf_feature_changes.csv")
        cf_change_df.to_csv(csv_path, index=False)
        print(f"Counterfactual feature change CSV saved to {csv_path}")
        # Export to Excel with images
        # export_mmace_feature_changes_to_excel(cf_change_rows, results_dir)
        
        # --- Aggregate data in cf_change_rows for each fold ---
        # Aggregate to match the structure of cf_change_rows for Excel export
        agg_cols = [
            'Fold_no', 'Feature_key', 'Model'
        ]
        agg_df = (
            cf_change_df
            .groupby(agg_cols)
            .agg({
                'Prediction_difference': 'mean',
                'Explanation_value': 'mean',
                # 'AddedRemoved': 'sum',
                # 'Prediction_original': 'mean',
                # 'Prediction_cf': 'mean',
                'Explanation_sign': lambda x: x.mode()[0] if not x.mode().empty else '',
                'SMILES_original': lambda x: '',
                'SMILES': lambda x: '',
                'SMILES_cf': lambda x: '',
                'features_original': lambda x: {},
                'features_cf': lambda x: {},
            })
            .reset_index()
        )
        # Add count column for number of changes per group
        agg_df['count_changes'] = cf_change_df.groupby(agg_cols)['Prediction_difference'].count().values

        # Take only top 10 features per fold by absolute value of Explanation_value
        top10_agg_df = (
            agg_df
            .sort_values(['Fold_no', 'Explanation_value'], key=lambda x: x.abs(), ascending=[True, False])
            .groupby('Fold_no')
            .head(10)
            .reset_index(drop=True)
        )

        agg_csv_path = os.path.join(results_dir, "mmace_cf_feature_changes_aggregated_by_fold.csv")
        top10_agg_df.to_csv(agg_csv_path, index=False)
        print(f"Aggregated counterfactual feature change CSV saved to {agg_csv_path}")
        # Export aggregated data to Excel using the same function
        export_mmace_feature_changes_to_excel(top10_agg_df.to_dict(orient='records'), results_dir)
    return feature_importance_per_fold
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run XAI Flow with specified parameters')
    parser.add_argument('--experiment_name', type=str, default='test', help='Name of the experiment (default: test)')
    parser.add_argument('--model', type=str, default='MMACE', help='Model to use (default: MMACE)')
    parser.add_argument('--seed', type=int, default=42, help='Set seed value (default: 42)')
    parser.add_argument('--explanation_value_mode', type=str, default='per_feature', choices=['per_feature', 'magnitude'],
                        help='Explanation value calculation mode: per_feature or magnitude (default: per_feature)')
    
    args = parser.parse_args()
    
    print(f"\n=== Running {args.model} ===\n")
    print("Arguments:", vars(args))
    # experiment_name = 'rf_test'
    mainMMACEFlow(experiment_name=args.experiment_name,
                  seed=args.seed,explanation_value_mode=args.explanation_value_mode)
