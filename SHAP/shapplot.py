import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def generate_shap_plots(shap_values, data, results_dir):
    """
    Generate SHAP dot plots for all features, linear plots for every SMILES, 
    and waterfall plots for every SMILES.
    Additionally, create a PDF containing all SHAP plots for every SMILES, formatted to fit an A4 page.

    Parameters:
    - shap_values (list): SHAP values for each fold.
    - data (DataFrame): The dataset containing features and SMILES.
    - results_dir (str): Directory to save the SHAP plots.
    """
    shap_dir = os.path.join(results_dir, "shap_plots")
    os.makedirs(shap_dir, exist_ok=True)

    # Generate SHAP dot plot for all features
    shap.summary_plot(
        shap_values=np.concatenate(shap_values, axis=0),
        features=data.drop(columns=['capacity_max', 'smiles']),
        feature_names=data.drop(columns=['capacity_max', 'smiles']).columns,
        show=False
    )
    plt.gcf().set_size_inches(8.27, 11.69)  # A4 size in inches (width x height)
    plt.savefig(os.path.join(shap_dir, "shap_summary_dot_plot.png"))
    plt.close()

    # Generate SHAP linear and waterfall plots for every SMILES and save them in a PDF
    pdf_path = os.path.join(shap_dir, "shap_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        for i, fold_shap_values in enumerate(shap_values):
            for j, shap_value in enumerate(fold_shap_values):
                smiles = data.iloc[j]['smiles']
                features = data.drop(columns=['capacity_max', 'smiles']).iloc[j]
                feature_names = data.drop(columns=['capacity_max', 'smiles']).columns

                # Linear plot
                plt.figure().set_figheight(10)
                shap.force_plot(
                    base_value=0,  # Replace with actual base value if available
                    shap_values=shap_value,
                    features=features,
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.title(f"SHAP Linear Plot for SMILES: {smiles}", fontsize=10)
                pdf.savefig()
                plt.close()

                # Waterfall plot
                plt.figure().set_figheight(10)
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_value,
                        base_values=0,  # Replace with actual base value if available
                        data=features,
                        feature_names=feature_names
                    ),
                    show=False
                )
                plt.title(f"SHAP Waterfall Plot for SMILES: {smiles}", fontsize=10)
                pdf.savefig()
                plt.close()