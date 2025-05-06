import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import selfies as sf
import exmol
import random
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

from utils.data_split import custom_data_kfold
from utils.exportlib import save_data_to_excel_with_highlights, save_scores_to_excel_new_sheet
from MMACE.mmace_cross_validation_pipeline import CrossValidationMMACEPipeline
from MMACE.timeoutexception import timeout
from MMACE.savemmacecfexcel import save_mmace_explanations_to_excel,create_mmace_pdf


def mainMMACEFlow():
    np.random.seed(0)
    random.seed(0)
    print("Running MMACE explanation pipeline...")
    parent_dir = os.path.dirname(os.getcwd())
    print("Parent directory:", parent_dir)
    

    maccs_fingerprints = os.path.join(parent_dir, 'data', 'new_maccs_merged.csv')
    # maccs_fingerprints = os.path.join(parent_dir, 'data', 'maccs_merged.csv')
    results_dir = os.path.join(parent_dir, 'results', 'battery', 'MMACE', 'local', datetime.today().strftime("%d-%m-%Y"))

    data = pd.read_csv(maccs_fingerprints)
    print(data.head())
    os.makedirs(results_dir, exist_ok=True)
    # custom_alphabet = get_custom_alphabet(data)
    custom_alphabet = get_basic_alphabet()
    print(f"Custom alphabet: {custom_alphabet}")


    folds = custom_data_kfold(data.drop(columns=['capacity_max']), data[['capacity_max']], 5)

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

    results, scores,cfs,samples,MMACE_Explanations = cv_pipeline.train_pipeline('RFReg')
    print("Results:", results)
    print("Scores:", scores)

    save_mmace_explanations_to_excel(MMACE_Explanations, results_dir=results_dir)
   
    process_folds_local(folds, data, samples, cfs,MMACE_Explanations,results_dir)

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

def process_folds_local(folds,data,samples,cfs,MMACE_Explanations,results_dir):
    """
    Process folds for local MMACE explanations.
    :param folds: List of folds for cross-validation.
    :param data: DataFrame containing the dataset.
    :param samples: List of samples for each fold.
    :param cfs: List of counterfactuals for each fold.
    :param MMACE_Explanations: List of MMACE explanations for each fold.
    """
    print("Processing folds for local MMACE...")

    for i, fold in enumerate(folds):
        test_f = data.loc[fold[1]]
        mmace_cf = MMACE_Explanations[i]["explanations"]
        pdf_path = create_mmace_pdf(MMACE_Explanations, i, results_dir)
        print(f"Created PDF report for fold {i} at: {pdf_path}")
       
if __name__ == '__main__':
    mainMMACEFlow()
