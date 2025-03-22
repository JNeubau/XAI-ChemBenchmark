import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import xlsxwriter
import os

def save_data_to_excel(data, smiles_list, excel_file):
    """
    Save data to an Excel file with molecule images generated from SMILES strings.

    Parameters:
    - data (dict): A dictionary containing the data to be saved. 
                   Keys should be column names, and values should be lists of column data.
                   Example:
                   {
                       "ID": [1, 2, 3],
                       "Name": ["Molecule1", "Molecule2", "Molecule3"],
                       "SMILES": ["CCO", "CCN", "CCCCCCCCCCCCCCCCCCCCCCCCC"]
                   }
    - smiles_list (list): A list of SMILES strings corresponding to the molecules.
                          Example: ["CCO", "CCN", "CCCCCCCCCCCCCCCCCCCCCCCCC"]
    - excel_file (str): The path to the Excel file where the data will be saved.
                        Example: "output.xlsx"
    """
    df = pd.DataFrame(data)

    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data", startrow=0, startcol=6)

        workbook = writer.book
        worksheet = writer.sheets["Data"]

        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                os.makedirs("png", exist_ok=True)
                img_path = f"png/mol_{i}.png"
                Draw.MolToFile(mol, img_path)

                row = i + 1
                col = 0
                worksheet.insert_image(row, col, img_path)

                worksheet.set_row(row, 250)
