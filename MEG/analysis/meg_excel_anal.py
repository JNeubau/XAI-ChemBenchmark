import os
import json
import pandas as pd
from glob import glob
from datetime import datetime

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

def process_dataframes(og_df, cf_df):
    result_df = pd.DataFrame(columns=['fold_no', 'mol_id', 'og_prediction_type'])
    return result_df

def save_to_excel(df: pd.DataFrame, output_dir, output_excel):
    df.to_excel(os.path.join(output_dir, output_excel), index=False)
    print(f"Data successfully saved to {os.path.join(output_dir, output_excel)}")
    return df

def anal_global():
    experiment_name = "new_test" 
    workdir = os.path.join(os.getcwd(), 'results', experiment_name, 'MEG')
    output_excel = f'molecule_results_with_highlights_{datetime.now().strftime("%H-%M-%S")}.xlsx'

    json_files = find_json_files(workdir)
    all_data = collect_all_data(json_files)
    og_df, cf_df = convert_json_to_dataframel(all_data)
    print(og_df.T)
    print(cf_df.T)
    save_to_excel(og_df, os.path.join(workdir, 'global'), 'raw_og_' + output_excel)
    save_to_excel(cf_df, os.path.join(workdir, 'global'), 'raw_cf_' + output_excel)
    
    result_df = process_dataframes(og_df, cf_df)
    print(result_df)
    print(result_df.T)
    # save_to_excel(result_df, os.path.join(workdir, 'global'), output_excel)

if __name__ == "__main__":
    anal_global()