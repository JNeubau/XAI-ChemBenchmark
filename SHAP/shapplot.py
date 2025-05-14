import os
import numpy as np
import pandas as pd
from datetime import datetime
import shap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_shap_plots_local(data, shap_values, folds,plots_dir, plots_to_run=None):
    """
    Generate SHAP local plots (force and waterfall) for each SMILES and save them in a PDF.

    Parameters:
    - data (DataFrame): The dataset containing features and SMILES.
    - shap_values (list): SHAP values for each fold.
    - plots_dir (str): Directory to save the SHAP plots.
    - plots_to_run (list): List of plots to generate. Options are 'force', 'waterfall', or 'all'.
    """
    if plots_to_run is None:
        return

    if 'all' in plots_to_run:
        plots_to_run = ['force', 'waterfall']

    os.makedirs(plots_dir, exist_ok=True)

    pdf_path = os.path.join(plots_dir, f"shap_local_plots_{datetime.now().strftime('%H_%M_%S')}.pdf")
    with PdfPages(pdf_path) as pdf:
        for i, (fold_shap_values, fold_data) in enumerate(zip(shap_values,folds)):
            test_f = data.loc[fold_data[1]]
            for j, shap_value in enumerate(fold_shap_values):
                smiles = test_f.iloc[j]['smiles']
                features = data.drop(columns=['capacity_max', 'smiles']).iloc[j]
                feature_names = data.drop(columns=['capacity_max', 'smiles']).columns

                if 'force' in plots_to_run:
                    # Generate SHAP force plot
                    plt.figure(figsize=(8, 6))
                    shap.force_plot(
                        base_value=0,  # Replace with actual base value if available
                        shap_values=shap_value,
                        features=features,
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False
                    )
                    plt.title(f"SHAP Force Plot for SMILES: {smiles}", fontsize=12)
                    pdf.savefig(bbox_inches='tight')
                    plt.close('all')

                if 'waterfall' in plots_to_run:
                    # Generate SHAP waterfall plot
                    plt.figure(figsize=(8, 6))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,
                            base_values=0,  # Replace with actual base value if available
                            data=features,
                            feature_names=feature_names
                        ),
                        show=False
                    )
                    plt.title(f"SHAP Waterfall Plot for SMILES: {smiles}", fontsize=12)
                    pdf.savefig(bbox_inches='tight')
                    plt.close('all')

def generate_shap_plots_folds(data, shap_values, plots_dir, plots_to_run=None):
    """
    Generate SHAP fold-level plots (e.g., summary) and save them as PNG.

    Parameters:
    - data (DataFrame): The dataset containing features and SMILES.
    - shap_values (list): SHAP values for each fold.
    - plots_dir (str): Directory to save the SHAP plots.
    - plots_to_run (list): List of plots to generate. Options are 'summary' or 'all'.
    """
    if plots_to_run is None:
        return

    if 'all' in plots_to_run:
        plots_to_run = ['summary']

    os.makedirs(plots_dir, exist_ok=True)

    if 'summary' in plots_to_run:
        # Generate overall SHAP summary plot
        plt.figure(figsize=(8.27, 11.69))  # A4 size in inches (width x height)
        shap.summary_plot(
            shap_values=np.concatenate(shap_values, axis=0),
            features=data.drop(columns=['capacity_max', 'smiles']),
            feature_names=data.drop(columns=['capacity_max', 'smiles']).columns,
            show=False
        )
        plt.savefig(os.path.join(plots_dir, f"shap_summary_plot_all_{datetime.now().strftime('%H_%M_%S')}.png"), bbox_inches='tight')
        plt.close('all')

        # Generate SHAP summary plot for each fold
        for fold_idx, fold_shap_values in enumerate(shap_values):
            # print(f"Generating SHAP summary plot for fold {fold_idx}...")
            # print(f"Fold SHAP values: {fold_shap_values[fold_idx]}")
            plt.figure(figsize=(8.27, 11.69))  # A4 size in inches (width x height)
            shap.summary_plot(
                shap_values=fold_shap_values,
                features=data.drop(columns=['capacity_max', 'smiles']).iloc[fold_idx::len(shap_values)],
                feature_names=data.drop(columns=['capacity_max', 'smiles']).columns,
                show=False
            )
            plt.savefig(os.path.join(plots_dir, f"shap_summary_plot_fold_{fold_idx}_{datetime.now().strftime('%H_%M_%S')}.png"), bbox_inches='tight')
            plt.close('all')