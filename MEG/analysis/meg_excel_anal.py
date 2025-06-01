import os
import sys
import json
import pandas as pd
from glob import glob
from datetime import datetime

from utils import get_fingerprints, get_smarts, tanimoto_similarity
import joblib

parent_dir = os.getcwd()
maccs_merge_path = os.path.join(parent_dir, 'data', 'new_maccs_merged.csv')
# maccs_merge_path = os.path.join(parent_dir, 'data', 'new_maccs_merged_all.csv')
smarts_mapping_path = os.path.join(parent_dir, 'data', 'new_maccs_smarts_mapping.json')

experiment_name = 'full_test' 
    
def find_json_files(base_dir):
    search_pattern = os.path.join(
        base_dir, 'meg_output', "*_*", "data.json"
    )
    return glob(search_pattern)

def parse_json_file(json_file):
    folder_name = os.path.basename(os.path.dirname(json_file))
    fold_no, mol_id = folder_name.split("_", 1)

    with open(json_file, "r") as f:
        data = json.load(f)

    rows = []
    if isinstance(data, dict):
        data_row = {"fold_no": fold_no, "mol_id": mol_id}
        data_row.update(data)
        rows.append(data_row)
    elif isinstance(data, list):
        for entry in data:
            data_row = {"fold_no": fold_no, "mol_id": mol_id}
            if isinstance(entry, dict):
                data_row.update(entry)
            else:
                data_row["value"] = entry
            rows.append(data_row)
    return rows

def collect_all_data(json_files):
    all_data = []
    for json_file in json_files:
        all_data.extend(parse_json_file(json_file))
    return all_data

def convert_json_to_dataframel(data):
    df = pd.DataFrame(data)    
    if 'prediction' in df.columns:
        if not df['prediction'].isna().all():
            # Extract nested values
            df['prediction_type'] = df['prediction'].apply(
                lambda x: x.get('type') if isinstance(x, dict) else None)
            df['prediction_output'] = df['prediction'].apply(
                lambda x: x.get('output') if isinstance(x, dict) else None)
            df['prediction_for_explanation'] = df['prediction'].apply(
                lambda x: x.get('for_explanation') if isinstance(x, dict) else None)
            df['prediction_class'] = df['prediction'].apply(
                lambda x: x.get('class') if isinstance(x, dict) else None)
            df['prediction_difference'] = df['prediction'].apply(
                lambda x: x.get('difference') if isinstance(x, dict) else None)
            df['prediction_original'] = df['prediction'].apply(
                lambda x: x.get('original') if isinstance(x, dict) else None)
            df.drop(['prediction'], axis=1, inplace=True)
    
    original_df = df[df['marker'] == 'og'].copy()
    original_df.drop(['reward', 'reward_pred', 'reward_sim', 'features', 'id',
                      'prediction_difference', 'prediction_original'], 
                    axis=1, inplace=True)
    counterfactual_df = df[df['marker'] == 'cf'].copy()
    counterfactual_df.drop(['prediction_for_explanation', 'prediction_class'], 
                    axis=1, inplace=True)
    return original_df, counterfactual_df

def get_rf_model(model_path):
    return joblib.load(model_path)

def process_dataframes(og_df, cf_df):
    result_df = pd.DataFrame(columns=['Fold', 'Instance', 'Feature_key', 'SMARTS', 'SMILES_original', 'SMILES', 'Original_Prediction', 
                                      'SMILES_cf', 'Counterfactual_Prediction', 'Similarity', 'Model',
                                      'Prediction_difference', 'Explanation_sign', 'Explanation_value',
                                      'Pred_Original_with_feature_change', 'count_changes'])
    
    for index, cf_row in cf_df.iterrows():
        fold_no = cf_row['fold_no']
        mol_id = cf_row['mol_id']
        
        og_row = og_df[(og_df['fold_no'] == fold_no) & (og_df['mol_id'] == mol_id)].iloc[0]
        og_smiles = og_row['smiles']
        og_fp_df = get_fingerprints(og_smiles, maccs_merge_path)
        og_fp = [float(val) for val in og_fp_df.values.flatten().tolist()]
        
        cf_fp = cf_row.get('features')
        
        
        changed = sum([abs(og_fp[j] - cf_fp[j]) for j in range(len(og_fp))])
        for i in range(len(og_fp)):
            if og_fp[i] == cf_fp[i]:
                continue
            
            feature_key = og_fp_df.columns[i]
            smarts = get_smarts(feature_key, smarts_mapping_path)
            similarity = tanimoto_similarity(og_smiles, cf_row['smiles'])
            
            pred_diff = cf_row['prediction_difference']
            
            rf_model = get_rf_model(os.path.join(os.getcwd(), 'RFReg', experiment_name, 'ckpt', f'model_{fold_no}.joblib'))
            input_df = og_fp_df.copy()
            input_df.iloc[0, i] = abs(int(input_df.iloc[0, i]) - 1)
            pred_reverted = rf_model.predict(input_df)[0]
            explanation_value = cf_row['prediction_output'] - pred_reverted
        
            new_row = {
                'Fold': fold_no,
                'Instance': mol_id,
                'Feature_key': feature_key,
                'SMARTS': smarts,
                'SMILES_original': og_smiles,
                'SMILES': og_smiles,
                'Original_Prediction': og_row['prediction_output'],
                'SMILES_cf': cf_row['smiles'],
                'Counterfactual_Prediction': cf_row['prediction_output'],
                'Similarity': similarity,
                'Model': 'MEG',
                'Prediction_difference': pred_diff,       
                'Explanation_value': abs(explanation_value),
                "Explanation_sign": 'Positive' if explanation_value > 0 else 'Negative',  
                'Pred_Original_with_feature_change': pred_reverted,
                'count_changes': changed
            }
            result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

    return 

def aggregate_to_global(df: pd.DataFrame):
    return

def save_to_excel(df: pd.DataFrame, output_dir, output_excel):
    os.makedirs(output_dir, exist_ok=True)
    df.to_excel(os.path.join(output_dir, output_excel), index=False)
    print(f"Data successfully saved to {os.path.join(output_dir, output_excel)}")
    return df

def anal_global():
    workdir = os.path.join(os.getcwd(), 'results', experiment_name, 'MEG')
    output_excel = f'molecule_results_with_highlights_{datetime.now().strftime("%H-%M-%S")}.xlsx'

    json_files = find_json_files(workdir)
    all_data = collect_all_data(json_files)
    og_df, cf_df = convert_json_to_dataframel(all_data)
    save_to_excel(og_df, os.path.join(workdir, 'local'), 'raw_og_' + output_excel)
    save_to_excel(cf_df, os.path.join(workdir, 'local'), 'raw_cf_' + output_excel)
    
    local_result_df = process_dataframes(og_df, cf_df)
    save_to_excel(local_result_df, os.path.join(workdir, 'local'), output_excel)
    
    # global_result_df = aggregate_to_global(local_result_df)
    # save_to_excel(global_result_df, os.path.join(workdir, 'global'), output_excel)
    

if __name__ == "__main__":
    anal_global()