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
    try:
        fold_no, mol_id = folder_name.split("_", 1)
    except ValueError:
        fold_no, mol_id = None, None

    with open(json_file, "r") as f:
        data = json.load(f)

    print(data)
    rows = []
    if isinstance(data, dict):
        data_row = {"fold_no": fold_no, "mol_id": mol_id}
        data_row.update(data)
        rows.append(data_row)
    elif isinstance(data, list):
        for entry in data:
            # print('ent', entry)
            data_row = {"fold_no": fold_no, "mol_id": mol_id}
            if isinstance(entry, dict):
                data_row.update(entry)
            else:
                data_row["value"] = entry
            rows.append(data_row)
    return rows

def collect_all_data(json_files):
    all_data = []
    for json_file in json_files[:1]:
        all_data.extend(parse_json_file(json_file))
    return all_data

def save_to_excel(data, output_excel):
    df = pd.DataFrame(data)
    df.to_excel(output_excel, index=False)
    print(f"Data saved to {output_excel}")

def anal_global():
    experiment_name = "new_test" 
    workdir = os.path.join(os.getcwd(), 'results', experiment_name, 'MEG')
    output_excel = f'molecule_results_with_highlights_{datetime.now().strftime("%H-%M-%S")}.xlsx'

    json_files = find_json_files(workdir)
    all_data = collect_all_data(json_files)
    print(all_data[:3], len(all_data), all_data[0].keys()) 
    # save_to_excel(all_data, output_excel)

if __name__ == "__main__":
    anal_global()