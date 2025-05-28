import os
import json
import pandas as pd
import glob

def read_shap_data(workdir, experiment_name):
    shap_path = os.path.join(workdir, 'results', experiment_name, 'SHAP', 'predicted_capacity.csv')
    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"SHAP file not found: {shap_path}")
    shap_df = pd.read_csv(shap_path, header=None)
    return shap_df

def read_meg_data(workdir, experiment_name):
    meg_dir = os.path.join(workdir, 'results', experiment_name, 'MEG', 'meg_output')
    meg_data = pd.DataFrame(columns=range(14), index=range(5))
    if not os.path.exists(meg_dir):
        raise FileNotFoundError(f"MEG directory not found: {meg_dir}")
    for folder in os.listdir(meg_dir):
        folder_path = os.path.join(meg_dir, folder)
        if os.path.isdir(folder_path):
            data_json_path = os.path.join(folder_path, 'data.json')
            if os.path.exists(data_json_path):
                with open(data_json_path, 'r') as f:
                    data = json.load(f)
                    pred = data[0]['prediction']['output']
                    fold, mol_idx = folder.split("_", 1)
                    # Parse fold and id from folder name
                    fold = int(fold)
                    mol_idx = int(mol_idx)
                    meg_data.at[fold, mol_idx] = pred
    return meg_data

def make_plot(data1: pd.DataFrame, data2: pd.DataFrame, save_dir: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    num_rows = data1.shape[0]
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 5* num_rows), constrained_layout=True)
    
    # If there's only one row, axes won't be an array
    if num_rows == 1:
        axes = [axes]
        
    for row in range(num_rows):
        d1 = pd.Series(data1.iloc[row, :])
        d2 = pd.Series(data2.iloc[row, :])
        
        # Filter out NaN values
        d1 = d1.dropna().values
        d2 = d2.dropna().values
        
        # Create boxplot on the current axis
        ax = axes[row]
        
        # If data is available, create boxplot
        if len(d1) > 0 and len(d2) > 0:
            plot_data = [d1, d2]
            ax.boxplot(plot_data, tick_labels=['SHAP', 'MEG'], patch_artist=True,
                      boxprops=dict(facecolor="lightblue"),
                      medianprops=dict(color="red"))
            
            # Add some statistics as text
            stats_text = f"SHAP: mean={np.mean(d1):.2f}, median={np.median(d1):.2f}\n" + \
                         f"MEG: mean={np.mean(d2):.2f}, median={np.median(d2):.2f}"
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, "Insufficient data for this fold", 
                  horizontalalignment='center', verticalalignment='center')
        
        ax.set_title(f'Fold {row} Comparison')
        ax.set_ylabel('Predicted Capacity')
    
    # Add a common title
    plt.suptitle('Comparison of SHAP and MEG Predictions by Fold', fontsize=16)
    
    # Save the figure
    fig.savefig(os.path.join(save_dir, 'shap_vs_meg_comparison_by_fold.png'), dpi=300)
    
    # Now create an overall comparison plot
    plt.figure(figsize=(10, 6))
    
    # Safely convert and flatten all data
    all_shap = []
    for val in data1.values.flatten():
        try:
            if pd.notna(val):  # Use pandas notna instead of numpy isnan
                all_shap.append(float(val))
        except (TypeError, ValueError):
            pass  # Skip values that can't be converted
    
    all_meg = []
    for val in data2.values.flatten():
        try:
            if pd.notna(val):
                all_meg.append(float(val))
        except (TypeError, ValueError):
            pass
    
    # Create overall boxplot
    if all_shap and all_meg:  # Check if both lists have data
        overall_data = pd.DataFrame({
            'Method': ['SHAP']*len(all_shap) + ['MEG']*len(all_meg),
            'Capacity': all_shap + all_meg
        })
        
        sns.boxplot(x='Method', y='Capacity', data=overall_data, palette="Set3", hue='Method', legend=False)
        plt.title('Overall Comparison of SHAP and MEG Predictions')
        plt.ylabel('Predicted Capacity')
        plt.savefig(os.path.join(save_dir, 'shap_vs_meg_overall_comparison.png'), dpi=300)
    else:
        print("Warning: Not enough data for overall comparison plot")
    
    plt.close('all')
    
def make_bar_plot(data1:pd.DataFrame, data2:pd.DataFrame, save_dir:str):
    import matplotlib.pyplot as plt
    import numpy as np

    num_rows = data1.shape[0]
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 5 * num_rows), constrained_layout=True)
    
    if num_rows == 1:
        axes = [axes]
        
    for row in range(num_rows):
        d1 = pd.Series(data1.iloc[row, :])
        d2 = pd.Series(data2.iloc[row, :])
        
        d1 = d1.dropna().values
        d2 = d2.dropna().values
        
        ax = axes[row]
        
        if len(d1) > 0 and len(d2) > 0:
            indices = np.arange(len(d1))
            width = 0.35
            
            ax.bar(indices - width/2, d1, width, label='SHAP', color='lightblue')
            ax.bar(indices + width/2, d2, width, label='MEG', color='lightgreen')
            
            ax.set_xticks(indices)
            ax.set_xticklabels([f'Fold {i}' for i in range(len(d1))])
            ax.set_title(f'Fold {row} Comparison')
            ax.set_ylabel('Predicted Capacity')
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Insufficient data for this fold", 
                  horizontalalignment='center', verticalalignment='center')
    
    plt.suptitle('Bar Plot Comparison of SHAP and MEG Predictions by Fold', fontsize=16)
    fig.savefig(os.path.join(save_dir, 'shap_vs_meg_bar_comparison_by_fold.png'), dpi=300)
    
def read_data_scores_SHAP(workdir, experiment_name):
    shap_path = os.path.join(workdir, 'RFReg', experiment_name, 'ckpt', 'molecule_scores.xlsx')
    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"SHAP scores file not found: {shap_path}")
    shap_scores = pd.read_excel(shap_path, index_col=0, engine='openpyxl')
    return shap_scores

def read_data_scores_MEG(workdir, experiment_name):
    meg_scores_dir = os.path.join(workdir, 'RFReg', experiment_name)
    json_files = sorted(glob.glob(os.path.join(meg_scores_dir, "rf_regressor_results_*.json")))
    if not json_files:
        raise FileNotFoundError(f"No MEG scores JSON files found in: {meg_scores_dir}")

    rows = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            rows.append(data)

    meg_scores = pd.DataFrame(rows)
    return meg_scores

if __name__ == "__main__":
    workdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    experiment_name = "new_test"  
    # print(workdir)

    shap_df = read_shap_data(workdir, experiment_name)
    # print("SHAP Data:")
    # print(shap_df)

    meg_data = read_meg_data(workdir, experiment_name)
    # print("\nMEG Data:")
    # print(meg_data)
    
    make_plot(shap_df, meg_data, os.path.join(workdir, 'results', experiment_name))
    make_bar_plot(shap_df, meg_data, os.path.join(workdir, 'results', experiment_name))
    
    shap_scores = read_data_scores_SHAP(workdir, experiment_name)
    print("\nSHAP Scores:")
    print(shap_scores)
    
    meg_scores = read_data_scores_MEG(workdir, experiment_name)
    print("\nMEG Scores:")
    print(meg_scores)