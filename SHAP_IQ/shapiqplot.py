import os
import numpy as np
import pandas as pd
import shapiq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def plot_shapiq_local(data, shap_values, plots_dir, plots_to_run=None):
    """
    Generate SHAP-IQ plots for all features and save them in the specified results directory.

    Parameters:
    - data (DataFrame): The dataset containing features and SMILES.
    - shap_values (list): SHAP-IQ values for each fold.
    - results_dir (str): Directory to save the SHAP-IQ plots.
    - plots_to_run (list): List of plots to generate. Options are 'all', 'summary', 'linear', 'force', 'waterfall'.
    """
    if plots_to_run is None:
        return
    
    if 'all' in plots_to_run:
        plots_to_run = ['force', 'waterfall']
        # plots_to_run = ['force', 'waterfall', 'bar', 'upset', 'network']
    
    os.makedirs(plots_dir, exist_ok=True)
    
    pdf_path = os.path.join(plots_dir, f"shapiq_plots_local_{datetime.now().strftime('%H_%M_%S')}.pdf")
    with PdfPages(pdf_path) as pdf:
        for i, fold_shap_values in enumerate(shap_values):
            for j, shap_val in enumerate(fold_shap_values):
                smiles = data.iloc[j]['smiles']
                feature_names = data.drop(columns=['capacity_max', 'smiles']).columns.str.replace('fingerprint', '', regex=False)
                
                if 'force' in plots_to_run:
                    # Plot force plot
                    plt.figure(figsize=(8, 7))
                    shapiq.plot.force_plot(
                        interaction_values=shap_val,
                        feature_names=feature_names,
                        abbreviate=False,
                        show=False,
                    )
                    plt.title(f"SHAP-IQ Force Plot for SMILES: {smiles}", fontsize=12)
                    pdf.savefig(bbox_inches='tight')
                    plt.close('all') 

                if 'waterfall' in plots_to_run:
                    # Plot waterfall plot
                    plt.figure(figsize=(8, 6))
                    shapiq.plot.waterfall_plot(
                        interaction_values=shap_val,
                        feature_names=feature_names,
                        abbreviate=False,
                        show=False,
                    )
                    plt.title(f"SHAP-IQ Waterfall Plot for SMILES: {smiles}", fontsize=12)
                    pdf.savefig(bbox_inches='tight')
                    plt.close('all') 
                
                if 'upset' in plots_to_run:
                    # Plot upset plot
                    plt.figure(figsize=(8, 6))
                    shapiq.plot.upset_plot(
                        interaction_values=shap_val,
                        feature_names=feature_names,
                        show=False,
                    )
                    plt.title(f"SHAP-IQ Upset Plot for SMILES: {smiles}", fontsize=12)
                    pdf.savefig(bbox_inches='tight')
                    plt.close('all') 
                    
                if 'network' in plots_to_run:
                    # Plot network plot
                    plt.figure(figsize=(8, 6))
                    shapiq.plot.network_plot(
                        interaction_values=shap_val,
                        feature_names=feature_names,
                        show=False,
                    )
                    plt.title(f"SHAP-IQ Network Plot for SMILES: {smiles}", fontsize=12)
                    pdf.savefig(bbox_inches='tight')
                    plt.close('all') 
                    
            # if 'bar' in plots_to_run:
            #     # Plot bar plot
            #     plt.figure(figsize=(8, 6))
            #     shapiq.plot.bar_plot(
            #         list_of_interaction_values=fold_shap_values,
            #         feature_names=data.drop(columns=['capacity_max', 'smiles']).columns.str.replace('fingerprint', '', regex=False),
            #         abbreviate=False,
            #         show=False,
            #     )
            #     plt.title(f"SHAP-IQ Network Plot for SMILES: {smiles}", fontsize=12)
            #     pdf.savefig(bbox_inches='tight')
            #     plt.close('all') 
                
                
def plot_shapiq_fold(data, shap_values, plots_dir, plots_to_run=None):
    """
    Generate SHAP-IQ plots for all features and save them in the specified results directory.

    Parameters:
    - data (DataFrame): The dataset containing features and SMILES.
    - shap_values (list): SHAP-IQ values for each fold.
    - results_dir (str): Directory to save the SHAP-IQ plots.
    - plots_to_run (list): List of plots to generate. Options are 'all', 'summary', 'linear', 'force', 'waterfall'.
    """
    if plots_to_run is None:
        return
    
    if 'all' in plots_to_run:
        plots_to_run = ['bar']
        # plots_to_run = ['force', 'waterfall', 'bar', 'upset', 'network']
    
    os.makedirs(plots_dir, exist_ok=True)
    
    pdf_path = os.path.join(plots_dir, f"shapiq_plots_fold_{datetime.now().strftime('%H_%M_%S')}.pdf")
    with PdfPages(pdf_path) as pdf:
        for i, fold_shap_values in enumerate(shap_values):                    
            if 'bar' in plots_to_run:
                # Plot bar plot
                plt.figure(figsize=(8, 6))
                shapiq.plot.bar_plot(
                    list_of_interaction_values=fold_shap_values,
                    feature_names=data.drop(columns=['capacity_max', 'smiles']).columns.str.replace('fingerprint', '', regex=False),
                    abbreviate=False,
                    show=False,
                )
                plt.title(f"SHAP-IQ Bar Plot for Fold_id: {i}", fontsize=12)
                pdf.savefig(bbox_inches='tight')
                plt.close('all') 