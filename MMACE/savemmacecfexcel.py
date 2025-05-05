def save_mmace_explanations_to_excel(MMACE_Explanations, results_dir):
    """
    Save MMACE counterfactual explanations to Excel with molecule visualizations.
    
    Parameters:
    - data: original dataset with SMILES and predictions
    - MMACE_Explanations: list of dictionaries containing explanations per fold
    - results_dir: directory to save results
    """
    from datetime import datetime
    from rdkit import Chem
    from rdkit.Chem import Draw
    from io import BytesIO
    import pandas as pd
    import os
    
    excel_data = {
        "Fold": [],
        "Instance": [],
        "Original_SMILES": [],
        "Original_Prediction": [],
        "Counterfactual_SMILES": [],
        "Counterfactual_Prediction": [], 
        "Similarity": [],
        "Label": []
    }
    
    # Process each fold
    for fold_idx, fold_dict in enumerate(MMACE_Explanations):
        fold_explanations = fold_dict["explanations"]
        
        # Process each molecule in the fold
        for mol_idx, cf_list in enumerate(fold_explanations):
            if len(cf_list) == 0:
                continue
                
            # Get original molecule info
            original = next((cf for cf in cf_list if cf.is_origin), None)
            if not original:
                continue
                
            # Process each counterfactual
            for cf in cf_list:
                if not cf.is_origin:  # Skip the original molecule
                    excel_data["Fold"].append(fold_idx)
                    excel_data["Instance"].append(mol_idx)
                    excel_data["Original_SMILES"].append(original.smiles)
                    excel_data["Original_Prediction"].append(float(original.yhat))
                    excel_data["Counterfactual_SMILES"].append(cf.smiles)
                    excel_data["Counterfactual_Prediction"].append(float(cf.yhat))
                    excel_data["Similarity"].append(cf.similarity)
                    excel_data["Label"].append(cf.label)

    # Create DataFrame
    df = pd.DataFrame(excel_data)
    
    # Save to Excel with molecule visualizations
    excel_path = os.path.join(results_dir, f'mmace_explanations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Explanations', index=False, startrow=0, startcol=7)
        worksheet = writer.sheets['Explanations']
        
        # Add molecule visualizations
        for idx, row in df.iterrows():
            # Original molecule
            orig_mol = Chem.MolFromSmiles(row['Original_SMILES'])
            if orig_mol:
                img = Draw.MolToImage(orig_mol)
                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                row_num = idx + 1
                worksheet.insert_image(row_num, 0, '', {'image_data': img_buffer})
                
            # Counterfactual molecule
            cf_mol = Chem.MolFromSmiles(row['Counterfactual_SMILES'])
            if cf_mol:
                img = Draw.MolToImage(cf_mol)
                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                worksheet.insert_image(row_num, 3, '', {'image_data': img_buffer})
                
            worksheet.set_row(row_num, 150)  # Set row height to accommodate images
            
        # Adjust column widths
        worksheet.set_column('A:A', 20)  # Original molecule column
        worksheet.set_column('D:D', 20)  # Counterfactual molecule column
        worksheet.set_column('H:O', 15)  # Data columns
    
    print(f"MMACE explanations saved to: {excel_path}")