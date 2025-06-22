import os
import json
import pandas as pd
import glob

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use(['fast'])

 
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
            'Method': ['Py3.10']*len(all_shap) + ['Py3.7']*len(all_meg),
            # 'Method': ['SHAP']*len(all_shap) + ['MEG']*len(all_meg),
            'Capacity': all_shap + all_meg
        })
        
        sns.boxplot(x='Method', y='Capacity', data=overall_data, 
                    # palette="Set3", 
                    hue='Method', legend=False)
        plt.title('Overall Comparison of Py3.10 and Py3.7 Predictions')
        plt.ylabel('Predicted Capacity')
        plt.savefig(os.path.join(save_dir, 'shap_vs_meg_overall_comparison.png'), dpi=400)
    else:
        print("Warning: Not enough data for overall comparison plot")
    
    plt.close('all')
    
def make_bar_plot(data1:pd.DataFrame, data2:pd.DataFrame, save_dir:str):
    num_rows = data1.shape[0]
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 5 * num_rows), constrained_layout=True)
    
    if num_rows == 1:
        axes = [axes]
        
    for row in range(num_rows):
        d1 = pd.Series(data1.iloc[row, :])
        d2 = pd.Series(data2.iloc[row, :])
        
        d1 = d1.dropna().values
        d2 = d2.dropna().values
        if row == 4:
            d1 = d1[:-1]
        
        ax = axes[row]
        
        if len(d1) > 0 and len(d2) > 0:
            indices = np.arange(len(d1))
            width = 0.35
            
            ax.bar(indices - width/2, d1, width, label='rfRef_3.10')
            ax.bar(indices + width/2, d2, width, label='rfReg_3.7') 
            
            ax.set_xticks(indices)
            ax.set_xticklabels([f'Mol {i}' for i in range(len(d1))])
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

def plot_scores(shap_scores_df, meg_scores_df, save_path):
    """
    Plots a comparison of RMSE values between SHAP and MEG results.
    
    Parameters:
    shap_scores_df (pd.DataFrame): DataFrame containing SHAP scores with 'rmse' column
    meg_scores_df (pd.DataFrame): DataFrame containing MEG scores with 'test_mse' column
    save_path (str): Directory to save the plot
    """
    # Extract RMSE from SHAP and calculate RMSE from MEG's MSE
    shap_rmse = shap_scores_df['rmse'].values
    meg_rmse = np.sqrt(meg_scores_df['test_mse'].values)
    
    # Calculate the final/average values
    shap_final = shap_rmse[-1] if 'Final' in shap_scores_df.index else shap_rmse.mean()
    meg_final = meg_rmse.mean()  # Calculate average for MEG
    
    # Create a DataFrame for easy plotting
    folds = list(range(len(meg_rmse)))
    data = pd.DataFrame({
        'Fold': folds + ['Final'],
        'P3.10 RMSE': list(shap_rmse[:-1] if 'Final' in shap_scores_df.index else shap_rmse) + [shap_final],
        'P3.7 RMSE': list(meg_rmse) + [meg_final]
    })
    
    # Reshape data for seaborn
    # plot_data = pd.melt(data, id_vars=['Fold'], 
    #                     value_vars=['P3.10 RMSE', 'P3.7 RMSE'],
    #                     var_name='Method', value_name='RMSE')
    
    # Create a second plot with a line chart for comparison
    plt.figure(figsize=(10, 6))
    
    # Line plot excluding the final point to see trends
    line_data = data[data['Fold'] != 'Final'].copy()
    line_data['Fold'] = line_data['Fold'].astype(int)  # Ensure fold is integer for line plot
    
    plt.plot(line_data['Fold'], line_data['P3.10 RMSE'], 'o-', color='red', linewidth=2, label='P3.10 RMSE')
    plt.plot(line_data['Fold'], line_data['P3.7 RMSE'], 's-', color='blue', linewidth=2, label='P3.7 RMSE')
    
    # Add horizontal lines for final values
    shap_final_value = data.loc[data['Fold'] == 'Final', 'P3.10 RMSE'].values[0]
    meg_final_value = data.loc[data['Fold'] == 'Final', 'P3.7 RMSE'].values[0]
    
    plt.axhline(y=shap_final_value, color='red', linestyle='--', alpha=0.7, 
               label=f'P3.10 Final')
            #    label=f'P3.10 Final: {shap_final_value:.2f}')
    plt.axhline(y=meg_final_value, color='blue', linestyle='--', alpha=0.7,
               label=f'P3.7 Final')
            #    label=f'P3.7 Final: {meg_final_value:.2f}')
    
    
    # Add text annotations for the horizontal lines
    plt.text(folds[-1] + 0.1, shap_final_value, f'{shap_final_value:.2f}', 
            color='red', ha='center', va='bottom', fontweight='bold')
    plt.text(folds[-1] + 0.1, meg_final_value, f'{meg_final_value:.2f}', 
            color='blue', ha='center', va='bottom', fontweight='bold')
    
    plt.title('RMSE Trend Comparison by Fold', fontsize=16)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('RMSE Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(folds)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'shap_vs_meg_rmse_trend.png'), dpi=300)
    
    plt.close('all')
    return data

if __name__ == "__main__":
    workdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    experiment_name = "final"  

    shap_df = read_shap_data(workdir, experiment_name)
    # print("SHAP Data:")
    # print(shap_df)

    meg_data = read_meg_data(workdir, experiment_name)
    # print("\nMEG Data:")
    # print(meg_data)
    
    make_plot(shap_df, meg_data, os.path.join(workdir, 'results', experiment_name))
    # make_bar_plot(shap_df, meg_data, os.path.join(workdir, 'results', experiment_name))
    
    shap_scores = read_data_scores_SHAP(workdir, experiment_name)
    print("\nSHAP Scores:")
    print(shap_scores)
    
    meg_scores = read_data_scores_MEG(workdir, experiment_name)
    print("\nMEG Scores:")
    print(meg_scores)
    # plot_scores(shap_scores, meg_scores, os.path.join(workdir, 'results', experiment_name))