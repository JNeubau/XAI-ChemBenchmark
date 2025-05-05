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
                
            worksheet.set_row(row_num, 250)  # Set row height to accommodate images
            
        # Adjust column widths
        worksheet.set_column('A:A', 20)  # Original molecule column
        worksheet.set_column('D:D', 20)  # Counterfactual molecule column
        worksheet.set_column('H:O', 15)  # Data columns
    
    print(f"MMACE explanations saved to: {excel_path}")

import os
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFMCS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from PIL import Image
import io

def find_molecular_changes(mol1, mol2):
    """
    Find atoms and bonds that differ between two molecules.
    """
    try:
        mcs = rdFMCS.FindMCS([mol1, mol2], 
                            completeRingsOnly=True,
                            matchValences=True,
                            ringMatchesRingOnly=True,
                            timeout=1)
        
        if mcs.numAtoms < 3:
            return set(range(mol2.GetNumAtoms())), set(range(mol2.GetNumBonds()))
        
        patt = Chem.MolFromSmarts(mcs.smartsString)
        match1 = mol1.GetSubstructMatch(patt)
        match2 = mol2.GetSubstructMatch(patt)
        atom_map = {match1[i]: match2[i] for i in range(len(match1))}
        
        changed_atoms = set(range(mol2.GetNumAtoms())) - set(atom_map.values())
        changed_bonds = set()
        
        for bond_idx in range(mol2.GetNumBonds()):
            bond = mol2.GetBondWithIdx(bond_idx)
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            
            if begin_atom in changed_atoms or end_atom in changed_atoms:
                changed_bonds.add(bond_idx)
            elif (begin_atom in atom_map.values() and end_atom in atom_map.values()):
                begin_orig = next(k for k, v in atom_map.items() if v == begin_atom)
                end_orig = next(k for k, v in atom_map.items() if v == end_atom)
                
                bond_orig = mol1.GetBondBetweenAtoms(begin_orig, end_orig)
                if bond_orig is None or bond_orig.GetBondType() != bond.GetBondType():
                    changed_bonds.add(bond_idx)
        
        return changed_atoms, changed_bonds
    
    except Exception as e:
        print(f"Error finding molecular changes: {e}")
        return set(range(mol2.GetNumAtoms())), set(range(mol2.GetNumBonds()))

def draw_molecule_with_highlights(mol, changed_atoms=None, changed_bonds=None, mol_size=(800, 800), diff_color=None):
    """
    Draw a molecule with highlighted changes.
    """
    if changed_atoms is None:
        changed_atoms = set()
    if changed_bonds is None:
        changed_bonds = set()
    
    drawer = rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1])
    drawer.SetFontSize(1.5)
    drawer.SetLineWidth(2)
    
    highlight_atom_dict = {}
    highlight_bond_dict = {}
    
    color = (0.6, 0.8, 1.0, 0.6) if diff_color == 'positive' else (1.0, 0.6, 0.6, 0.6)
    
    for atom_idx in changed_atoms:
        highlight_atom_dict[atom_idx] = color
    for bond_idx in changed_bonds:
        highlight_bond_dict[bond_idx] = color
    
    drawer.DrawMolecule(
        mol, 
        highlightAtoms=list(changed_atoms),
        highlightBonds=list(changed_bonds),
        highlightAtomColors=highlight_atom_dict,
        highlightBondColors=highlight_bond_dict
    )
    drawer.FinishDrawing()
    
    png_data = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png_data))

def create_mmace_pdf(MMACE_results, fold_idx, output_dir):
    """
    Create PDF visualizations for MMACE counterfactuals.
    """
    os.makedirs(output_dir, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'mmace_fold_{fold_idx}_{datetime_str}.pdf')
    
    with PdfPages(output_path) as pdf:
        for mol_idx, cf_list in enumerate(MMACE_results[fold_idx]["explanations"]):
            if not cf_list:
                continue
                
            original = next((cf for cf in cf_list if cf.is_origin), None)
            if not original:
                continue
                
            original_mol = Chem.MolFromSmiles(original.smiles)
            
            # Create page for original molecule
            fig = plt.figure(figsize=(12, 16))
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
            
            # Original molecule visualization
            ax1 = plt.subplot(gs[0])
            img = Draw.MolToImage(original_mol, size=(1200, 1200))
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title(f"Original Molecule (Instance {mol_idx})", fontsize=16, pad=20)
            
            # Original molecule info
            ax2 = plt.subplot(gs[1])
            ax2.axis('off')
            pred_text = [
                f"SMILES: {original.smiles}",
                f"Prediction: {float(original.yhat):.4f}"
            ]
            ax2.text(0.5, 0.7, '\n'.join(pred_text), ha='center', va='center', fontsize=14)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
            # Create pages for counterfactuals
            for cf_idx, cf in enumerate(cf_list):
                if cf.is_origin:
                    continue
                    
                cf_mol = Chem.MolFromSmiles(cf.smiles)
                changed_atoms, changed_bonds = find_molecular_changes(original_mol, cf_mol)
                
                fig = plt.figure(figsize=(12, 16))
                gs = gridspec.GridSpec(3, 2, height_ratios=[5, 5, 2])
                
                # Original molecule
                ax1 = plt.subplot(gs[0, :])
                ax1.imshow(Draw.MolToImage(original_mol, size=(1200, 800)))
                ax1.axis('off')
                ax1.set_title("Original Molecule", fontsize=16)
                
                # Counterfactual with highlights
                ax2 = plt.subplot(gs[1, :])
                is_positive = float(cf.yhat) > float(original.yhat)
                img_cf = draw_molecule_with_highlights(
                    cf_mol, 
                    changed_atoms, 
                    changed_bonds, 
                    mol_size=(1200, 800),
                    diff_color='positive' if is_positive else 'negative'
                )
                ax2.imshow(img_cf)
                ax2.axis('off')
                ax2.set_title(f"Counterfactual #{cf_idx+1}", fontsize=16)
                
                # Information section
                ax3 = plt.subplot(gs[2, :])
                ax3.axis('off')
                info_text = [
                    f"SMILES: {cf.smiles}",
                    f"Original Prediction: {float(original.yhat):.4f}",
                    f"Counterfactual Prediction: {float(cf.yhat):.4f}",
                    f"Difference: {float(cf.yhat - original.yhat):+.4f}",
                    f"Similarity: {cf.similarity:.4f}",
                    f"Label: {cf.label}"
                ]
                ax3.text(0.5, 0.7, '\n'.join(info_text), ha='center', va='center', fontsize=14)
                
                plt.suptitle(f"Counterfactual Analysis - Instance {mol_idx}", fontsize=18)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                pdf.savefig(fig)
                plt.close()
    
    print(f"PDF created at: {output_path}")
    return output_path