import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
# import xlsxwriter
import os
from matplotlib.colors import ColorConverter
from io import BytesIO
from matplotlib.colors import ColorConverter
from io import BytesIO


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
    clean_up_png_files_from_dir("png")
    

def save_data_to_excel_with_highlights_no_sort(data, smiles_list, smarts_list, excel_file):
    """
    Save data to an Excel file with molecule images generated from SMILES strings and highlighted substructures.

    Parameters:
    - data (dict): A dictionary containing the data to be saved.
                   Keys should be column names, and values should be lists of column data.
    - smiles_list (list): A list of SMILES strings corresponding to the molecules.
    - smarts_list (list): A list of SMARTS patterns to highlight in the molecules.
    - excel_file (str): The path to the Excel file where the data will be saved.
    """
    image_dir = os.path.dirname(os.getcwd()) + '\\png'
    df = pd.DataFrame(data)

    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data", startrow=0, startcol=6)
        worksheet = writer.sheets["Data"]

        for i, (smiles, smarts) in enumerate(zip(smiles_list, smarts_list)):

            if not smiles or not smiles[0]:
                continue
            # print(smiles, smarts)
            # print(type(smiles),smiles[0])
            # print(type(smarts))
            mol = Chem.MolFromSmiles(smiles[0])

            if not smiles or not smiles[0]:
                continue
            # print(smiles, smarts)
            # print(type(smiles),smiles[0])
            # print(type(smarts))
            mol = Chem.MolFromSmiles(smiles[0])
            match_smart = Chem.MolFromSmarts(smarts)
            if mol and match_smart:
                os.makedirs(image_dir, exist_ok=True)
                img_path = image_dir + f"\\mol_{i}.png"
                highlight_atoms = [atom for match in mol.GetSubstructMatches(match_smart) for atom in match]
                Draw.MolToFile(mol, img_path, highlightAtoms=highlight_atoms)

                row = i + 1
                col = 0
                worksheet.insert_image(row, col, img_path)

                worksheet.set_row(row, 250)
    clean_up_png_files_from_dir(image_dir)
    
    
def save_data_to_excel_with_highlights(data, excel_file):
    """
    Save data to an Excel file with molecule images generated from SMILES strings and highlighted substructures.
    The data will be sorted by SMILES and a specified feature.

    Parameters:
    - data (dict): A dictionary containing the data to be saved.
                   Keys should be column names, and values should be lists of column data.
    - excel_file (str): The path to the Excel file where the data will be saved.
    """    
    df = pd.DataFrame(data, 
                      columns=["Fold_No", 'Smiles_key', 'Feature_key', 
                               'SMARTS', 'Molecule', 'number_of_molecules_where_fingerprint', 'Number_where_important',
                               'feature_in_smiles','Shap_value', 'shap_sign',
                               'Capacity Max', 'Capacity Pred'])
    # df['Shap_sign'] = df['Shap_value'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    # df['Shap_value'] = df['Shap_value'].abs()
    df = df.sort_values(by=['Molecule', 'Feature_key', 'Fold_No'], ignore_index=True)

    with pd.ExcelWriter(excel_file, engine="xlsxwriter", mode='w') as writer:
        df.to_excel(writer, index=False, sheet_name="Data", startrow=0, startcol=13)
        worksheet = writer.sheets["Data"]

        for i, row in df.iterrows():
            smiles = row['Molecule']
            smarts = row['SMARTS']
            mol = Chem.MolFromSmiles(smiles)
            match_smart = Chem.MolFromSmarts(smarts)
            if mol and match_smart:
                highlight_atoms = [atom for match in mol.GetSubstructMatches(match_smart) for atom in match]
                
                # highlight_bonds = []
                # for match in mol.GetSubstructMatches(match_smart):
                #     for i in range(len(match) - 1):
                #         bond = mol.GetBondBetweenAtoms(match[i], match[i + 1])
                #         if bond:
                #             highlight_bonds.append(bond.GetIdx())

                if row['shap_sign'].split('|')[0] == 'Negative':
                    color = 'lightcoral'
                else:
                    color = 'aquamarine'
                
                img = Draw.MolToImage(
                    mol, 
                    highlightAtoms=highlight_atoms, 
                    # highlight_bonds=highlight_bonds, 
                    highlightColor=ColorConverter().to_rgb(color))
                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                row_num = i + 1
                col = 0
                worksheet.insert_image(row_num, col, '', {'image_data': img_buffer})

                img_smt = Draw.MolToImage(match_smart)
                img_smt_buffer = BytesIO()
                img_smt.save(img_smt_buffer, format='PNG')
                img_smt_buffer.seek(0)

                col_smt = 6
                worksheet.insert_image(row_num, col_smt, '', {'image_data': img_smt_buffer})
                worksheet.set_row(row_num, 250)
    # clean_up_png_files_from_dir(image_dir)
    
def save_interactions_to_excel_with_highlights(data, excel_file):
    """
    Save data to an Excel file with molecule images generated from SMILES strings and highlighted substructures.
    The data will be sorted by SMILES and a specified feature.

    Parameters:
    - data (dict): A dictionary containing the data to be saved.
                   Keys should be column names, and values should be lists of column data.
    - excel_file (str): The path to the Excel file where the data will be saved.
    """    
    df = pd.DataFrame(data, 
                      columns=["Fold_No", 'Smiles_key', 'Feature_key', 
                               'SMARTS', 'Molecule', 'number_of_molecules_where_fingerprint', 'Number_where_important',
                               'feature_in_smiles','Shap_value', 'shap_sign',
                               'Capacity Max', 'Capacity Pred'])
    df = df.sort_values(by=['Molecule', 'Feature_key', 'Fold_No'], ignore_index=True)

    with pd.ExcelWriter(excel_file, engine="xlsxwriter", mode='w') as writer:
        df.to_excel(writer, index=False, sheet_name="Data", startrow=0, startcol=0)
        worksheet = writer.sheets["Data"]

        for i, row in df.iterrows():
            smiles = row['Molecule']
            smarts = row['SMARTS']
            mol = Chem.MolFromSmiles(smiles)
            match_smarts = []
            highlight_atoms_list = []
            for j, smart in enumerate(smarts):
                match_smart = Chem.MolFromSmarts(smart)
                # match_smart = Chem.MolFromSmarts(smart)
                if mol and match_smart:
                    match_smarts.append(match_smart)
                    highlight_atoms = [atom for match in mol.GetSubstructMatches(match_smart) for atom in match]
                    highlight_atoms_list = highlight_atoms_list + highlight_atoms
                    
                    # highlight_bonds = []
                    # for match in mol.GetSubstructMatches(match_smart):
                    #     for i in range(len(match) - 1):
                    #         bond = mol.GetBondBetweenAtoms(match[i], match[i + 1])
                    #         if bond:
                    #             highlight_bonds.append(bond.GetIdx())

            if row['shap_sign'].split('|')[0] == 'Negative':
                color = 'lightcoral'
            else:
                color = 'aquamarine'
            
            # for j, match_smart in enumerate(match_smarts):
            img = Draw.MolToImage(
                mol, 
                highlightAtoms=highlight_atoms_list, 
                # highlight_bonds=highlight_bonds, 
                highlightColor=ColorConverter().to_rgb(color))
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            row_num = i + 1
            col = len(df.columns) + 1
            worksheet.insert_image(row_num, col, '', {'image_data': img_buffer})

            for j, match_smart in enumerate(match_smarts):
                img_smt = Draw.MolToImage(match_smart)
                img_smt_buffer = BytesIO()
                img_smt.save(img_smt_buffer, format='PNG')
                img_smt_buffer.seek(0)

                col_smt = col + (6 * (j +1))
                worksheet.insert_image(row_num, col_smt, '', {'image_data': img_smt_buffer})
            worksheet.set_row(row_num, 250)
    # clean_up_png_files_from_dir(image_dir)
    
def save_data_to_excel_with_highlights_lime(data, excel_file):
    """
    Save data to an Excel file with molecule images generated from SMILES strings and highlighted substructures.
    The data will be sorted by SMILES and a specified feature.

    Parameters:
    - data (dict): A dictionary containing the data to be saved.
                   Keys should be column names, and values should be lists of column data.
    - excel_file (str): The path to the Excel file where the data will be saved.
    """    
    df = pd.DataFrame(data, 
                      columns=["Fold_No", 'Smiles_key', 'Feature_key', 
                               'SMARTS', 'Molecule', 'number_of_molecules_where_fingerprint', 'Number_where_important',
                               'feature_in_smiles','lime_value', 'lime_sign',
                               'Capacity Max', 'Capacity Pred'])
    # df['Shap_sign'] = df['Shap_value'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    # df['Shap_value'] = df['Shap_value'].abs()
    df = df.sort_values(by=['Molecule', 'Feature_key', 'Fold_No'], ignore_index=True)

    with pd.ExcelWriter(excel_file, engine="xlsxwriter", mode='w') as writer:
        df.to_excel(writer, index=False, sheet_name="Data", startrow=0, startcol=13)
        worksheet = writer.sheets["Data"]

        for i, row in df.iterrows():
            smiles = row['Molecule']
            smarts = row['SMARTS']
            mol = Chem.MolFromSmiles(smiles)
            match_smart = Chem.MolFromSmarts(smarts)
            if mol and match_smart:
                highlight_atoms = [atom for match in mol.GetSubstructMatches(match_smart) for atom in match]
                
                # highlight_bonds = []
                # for match in mol.GetSubstructMatches(match_smart):
                #     for i in range(len(match) - 1):
                #         bond = mol.GetBondBetweenAtoms(match[i], match[i + 1])
                #         if bond:
                #             highlight_bonds.append(bond.GetIdx())

                if row['lime_sign'].split('|')[0] == 'Negative':
                    color = 'lightcoral'
                else:
                    color = 'aquamarine'
                
                img = Draw.MolToImage(
                    mol, 
                    highlightAtoms=highlight_atoms, 
                    # highlight_bonds=highlight_bonds, 
                    highlightColor=ColorConverter().to_rgb(color))
                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                row_num = i + 1
                col = 0
                worksheet.insert_image(row_num, col, '', {'image_data': img_buffer})

                img_smt = Draw.MolToImage(match_smart)
                img_smt_buffer = BytesIO()
                img_smt.save(img_smt_buffer, format='PNG')
                img_smt_buffer.seek(0)

                col_smt = 6
                worksheet.insert_image(row_num, col_smt, '', {'image_data': img_smt_buffer})
                worksheet.set_row(row_num, 250)
    # clean_up_png_files_from_dir(image_dir)
    
def save_scores_to_excel_new_sheet(data, excel_file):
    """
    Save scores to a new sheet in an Excel file.

    Parameters:
    - data (dict): A dictionary containing the data to be saved.
                   Keys should be column names, and values should be lists of column data.
    - excel_file (str): The path to the Excel file where the data will be saved.
    - sheet_name (str): The name of the new sheet to save the data.
    """
    df = pd.DataFrame(data)

    with pd.ExcelWriter(excel_file, engine="xlsxwriter", mode='w') as writer:
        df.to_excel(writer, index=True, sheet_name="Scores")


def clean_up_png_files_from_dir(dir_name):
    """
    Clean up the directory containing molecule images.
    Will remove the dir if it is empty.

    Parameters:
    - dir_name (str): The directory containing the molecule images.
    """
    for file in os.listdir(dir_name):
        if file.endswith(".png"):
            os.remove(os.path.join(dir_name, file))
    
    if not os.listdir(dir_name):
        os.rmdir(dir_name)