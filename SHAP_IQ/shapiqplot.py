import os
import numpy as np
import pandas as pd
import shapiq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# def generate_shap_plots(shap_values, data, results_dir):
#     """
#     Generate SHAP dot plots for all features and linear plots for every SMILES.
#     Additionally, create a PDF containing all SHAP plots for every SMILES, formatted to fit an A4 page.

#     Parameters:
#     - shap_values (list): SHAP values for each fold.
#     - data (DataFrame): The dataset containing features and SMILES.
#     - results_dir (str): Directory to save the SHAP plots.
#     """
#     shap_dir = os.path.join(results_dir, "shap_plots")
#     os.makedirs(shap_dir, exist_ok=True)

#     # Generate SHAP dot plot for all features
#     # shapiq.si_graph_plot(
#     #     shap_values=np.concatenate(shap_values, axis=0),
#     #     features=data.drop(columns=['capacity_max', 'smiles']),
#     #     feature_names=data.drop(columns=['capacity_max', 'smiles']).columns,
#     #     show=False
#     # )
#     # shapiq.summary_plot(
#     #     shap_values=np.concatenate(shap_values, axis=0),
#     #     features=data.drop(columns=['capacity_max', 'smiles']),
#     #     feature_names=data.drop(columns=['capacity_max', 'smiles']).columns,
#     #     show=False
#     # )
#     plt.gcf().set_size_inches(8.27, 11.69)  # A4 size in inches (width x height)
#     plt.savefig(os.path.join(shap_dir, "shap_summary_dot_plot.png"))
#     plt.close()

#     # Generate SHAP linear plots for every SMILES and save them in a PDF
#     pdf_path = os.path.join(shap_dir, "shap_linear_plots.pdf")
#     with PdfPages(pdf_path) as pdf:
#         for i, fold_shap_values in enumerate(shap_values):
#             for j, shap_value in enumerate(fold_shap_values):
#                 smiles = data.iloc[j]['smiles']
#                 plt.figure(figsize=(8, 6))
#                 shapiq.force_plot(
#                     interaction_values=shap_value,
#                     feature_names=data.drop(columns=['capacity_max', 'smiles']).columns,
#                     show=False
#                 )
#                 plt.title(f"SHAP Linear Plot for SMILES: {smiles}", fontsize=12)
#                 pdf.savefig(bbox_inches='tight')
#                 plt.close()
                

def plot_shapiq(data, shap_values, plots_dir):
    """
    Generate SHAP-IQ plots for all features and save them in the specified results directory.

    Parameters:
    - data (DataFrame): The dataset containing features and SMILES.
    - shap_values (list): SHAP-IQ values for each fold.
    - results_dir (str): Directory to save the SHAP-IQ plots.
    """
    print(shap_values)
    os.makedirs(plots_dir, exist_ok=True)

    # Generate SHAP-IQ summary plot for all features
    # pdf_path = os.path.join(plots_dir, "shapiq_si_graph_plots.pdf")
    # with PdfPages(pdf_path) as pdf:
    #     for i, fold_shap_values in enumerate(shap_values):
    #         for j, shap_val in enumerate(fold_shap_values):
    #             shapiq.si_graph_plot(
    #                 interaction_values=shap_val,
    #                 # features=data.drop(columns=['capacity_max', 'smiles']),
    #                 # feature_names=data.drop(columns=['capacity_max', 'smiles']).columns,
    #                 show=False  
    #             )
    #             plt.gcf().set_size_inches(8.27, 11.69)  # A4 size in inches (width x height)
    #             plt.title(f"SHAP-IQ Summary Plot for Fold {i} and val {j}", fontsize=12)
    #             pdf.savefig(bbox_inches='tight')
    #             plt.close()
    
    # shapiq.network_plot(
    # first_order_values=shap_values.get_n_order_values(1),
    # second_order_values=shap_values.get_n_order_values(2)
    # )
    # plt.gcf().set_size_inches(8.27, 11.69)  # A4 size in inches (width x height)
    # plt.savefig(os.path.join(plots_dir, "shapiq_network_plot.png"))

    # Generate SHAP-IQ linear plots for every SMILES and save them in a PDF
    pdf_path = os.path.join(plots_dir, f"shapiq_linear_plots_{datetime.now().strftime('%H_%M_%S')}.pdf")
    with PdfPages(pdf_path) as pdf:
        for i, fold_shap_values in enumerate(shap_values):
            for j, shap_val in enumerate(fold_shap_values):
                smiles = data.iloc[j]['smiles']
                plt.figure(figsize=(8, 6))
                shap_val.plot_force(
                    feature_names = data.drop(columns=['capacity_max', 'smiles']).columns,
                    show = False,
                )
                plt.title(f"SHAP-IQ Linear Plot for SMILES: {smiles}", fontsize=12)
                pdf.savefig(bbox_inches='tight')
                plt.close()