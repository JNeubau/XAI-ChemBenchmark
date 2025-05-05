from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFMCS
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

def load_meg_data(json_path):
    """
    Load MEG data from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file with MEG results
        
    Returns:
        dict: Dictionary with organized data containing:
            - original: Original molecule data
            - counterfactuals: List of counterfactual molecules
            - feature_diffs: Array of feature differences between original and CFs
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Organize data into a more usable structure
    result = {
        'original': None,
        'counterfactuals': [],
        'feature_diffs': []
    }
    
    # Extract original molecule
    for entry in data:
        if entry['marker'] == 'og':
            result['original'] = entry
            break
    
    # Extract counterfactuals and compute feature differences if available
    original_features = None
    if result['original'] and 'features' in result['original']:
        original_features = np.array(result['original']['features'])
    
    for entry in data:
        if entry['marker'] == 'cf':
            result['counterfactuals'].append(entry)
            
            # If features are available for both original and CF, compute differences
            if original_features is not None and 'features' in entry:
                cf_features = np.array(entry['features'])
                feature_diff = cf_features - original_features
                result['feature_diffs'].append(feature_diff)
    
    # Sort counterfactuals by reward (descending)
    if result['counterfactuals'] and 'reward' in result['counterfactuals'][0]:
        sorted_indices = sorted(range(len(result['counterfactuals'])), 
                              key=lambda i: result['counterfactuals'][i].get('reward', 0), 
                              reverse=True)
        
        result['counterfactuals'] = [result['counterfactuals'][i] for i in sorted_indices]
        if result['feature_diffs']:
            result['feature_diffs'] = [result['feature_diffs'][i] for i in sorted_indices]
    
    return result

def find_molecular_changes(mol1, mol2):
    """
    Find atoms and bonds that differ between two molecules.
    
    Args:
        mol1 (rdkit.Chem.Mol): First molecule (original)
        mol2 (rdkit.Chem.Mol): Second molecule (counterfactual)
        
    Returns:
        tuple: (changed_atoms, changed_bonds) - sets of indices of changed atoms and bonds
    """
    # Try MCS (Maximum Common Substructure) approach
    try:
        mcs = rdFMCS.FindMCS([mol1, mol2], 
                                  completeRingsOnly=True,
                                  matchValences=True,
                                  ringMatchesRingOnly=True,
                                  timeout=1)  # Limit search time
        
        # If no MCS found or it's too small, consider everything changed
        if mcs.numAtoms < 3:
            return set(range(mol2.GetNumAtoms())), set(range(mol2.GetNumBonds()))
        
        # Get the common substructure pattern
        patt = Chem.MolFromSmarts(mcs.smartsString)
        
        # Get atom mappings for both molecules
        match1 = mol1.GetSubstructMatch(patt)
        match2 = mol2.GetSubstructMatch(patt)
        
        # Create mapping from original to counterfactual
        atom_map = {match1[i]: match2[i] for i in range(len(match1))}
        
        # Find changed atoms (those not in the mapping)
        changed_atoms = set(range(mol2.GetNumAtoms())) - set(atom_map.values())
        
        # Find changed bonds
        changed_bonds = set()
        for bond_idx in range(mol2.GetNumBonds()):
            bond = mol2.GetBondWithIdx(bond_idx)
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            
            # If either atom is changed, the bond is changed
            if begin_atom in changed_atoms or end_atom in changed_atoms:
                changed_bonds.add(bond_idx)
            # Both atoms are unchanged, check if bond type is changed
            elif (begin_atom in atom_map.values() and end_atom in atom_map.values()):
                # Find corresponding atoms in original
                begin_orig = None
                end_orig = None
                for k, v in atom_map.items():
                    if v == begin_atom:
                        begin_orig = k
                    if v == end_atom:
                        end_orig = k
                
                if begin_orig is not None and end_orig is not None:
                    # Check if bond exists in original
                    bond_orig = mol1.GetBondBetweenAtoms(begin_orig, end_orig)
                    if bond_orig is None or bond_orig.GetBondType() != bond.GetBondType():
                        changed_bonds.add(bond_idx)
        
        return changed_atoms, changed_bonds
    
    except Exception as e:
        print(f"Error finding molecular changes: {e}")
        # Return all atoms and bonds as changed if there's an error
        return set(range(mol2.GetNumAtoms())), set(range(mol2.GetNumBonds()))

def draw_molecule_with_highlights(mol, changed_atoms=None, changed_bonds=None, mol_size=(800, 800), diff_color=None):
    """
    Draw a molecule with changed atoms and bonds highlighted with higher resolution.
    
    Args:
        mol (rdkit.Chem.Mol): Molecule to draw
        changed_atoms (set, optional): Set of atom indices to highlight
        changed_bonds (set, optional): Set of bond indices to highlight
        mol_size (tuple, optional): Size of the image (width, height)
        diff_color (str, optional): 'positive' for blue highlights, 'negative' or None for red highlights
        
    Returns:
        PIL.Image: High-resolution image of the molecule with highlights
    """
    if changed_atoms is None:
        changed_atoms = set()
    if changed_bonds is None:
        changed_bonds = set()
    
    # Initialize the drawing object with higher resolution
    drawer = rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1])
    
    # Increase font and bond size for better visibility
    drawer.SetFontSize(1.5)  # Larger font for atom labels
    drawer.SetLineWidth(2)    # Integer value as required by the API
    
    # Set up highlights with more opacity for better visibility
    highlight_atom_dict = {}
    highlight_bond_dict = {}

    # Use blue for positive changes, red for negative changes or default
    if diff_color == 'positive':
        # Light blue highlights
        for atom_idx in changed_atoms:
            highlight_atom_dict[atom_idx] = (0.6, 0.8, 1.0, 0.6)  # Light blue for atoms
        for bond_idx in changed_bonds:
            highlight_bond_dict[bond_idx] = (0.6, 0.8, 1.0, 0.6)  # Light blue for bonds
    else:
        # Red highlights
        for atom_idx in changed_atoms:
            highlight_atom_dict[atom_idx] = (1.0, 0.6, 0.6, 0.6)  # Brighter red for atoms
        for bond_idx in changed_bonds:
            highlight_bond_dict[bond_idx] = (1.0, 0.6, 0.6, 0.6)  # Salmon pink for bonds
    
    # Draw the molecule with highlights
    drawer.DrawMolecule(
        mol, 
        highlightAtoms=list(changed_atoms),
        highlightBonds=list(changed_bonds),
        highlightAtomColors=highlight_atom_dict,
        highlightBondColors=highlight_bond_dict
    )
    drawer.FinishDrawing()
    
    # Get the image as PNG data
    png_data = drawer.GetDrawingText()
    
    # Convert to PIL Image
    from PIL import Image
    import io
    return Image.open(io.BytesIO(png_data))

def format_smiles(smiles, max_len=80):
    if len(smiles) <= max_len:
        return smiles
    chunks = []
    for i in range(0, len(smiles), max_len):
        chunks.append(smiles[i:i+max_len])
    return '\n'.join(chunks)
            
def create_cf_pdf(data, output_path, max_cf=10):
    """
    Create a PDF with counterfactual molecules, one per page, with changes highlighted.
    
    Args:
        data (dict): Data structure from load_meg_data()
        output_path (str): Path to save the PDF file
        max_cf (int): Maximum number of counterfactuals to include
    """
    if not data['original'] or not data['counterfactuals']:
        print("No data available for PDF creation")
        return
    
    original = data['original']
    counterfactuals = data['counterfactuals'] if max_cf is None else data['counterfactuals'][:max_cf]
    
    # Save SMILES to a text file in the same directory as the PDF
    smiles_path = os.path.splitext(output_path)[0] + '_smiles.txt'
    with open(smiles_path, 'w') as f:
        f.write(f"Original molecule: {original['smiles']}\n")
        f.write(f"Original prediction: {original['prediction']['output']:.4f}\n\n")
        
        f.write("Counterfactuals:\n")
        for i, cf in enumerate(counterfactuals):
            f.write(f"CF #{i+1}: {cf['smiles']}\n")
            f.write(f"  Prediction: {cf['prediction']['output']:.4f}\n")
            f.write(f"  Difference: {cf['prediction']['difference']:+.4f}\n")
            f.write(f"  Reward: {cf.get('reward', 'N/A')}\n\n")
    
    # Get original molecule
    original_mol = Chem.MolFromSmiles(original['smiles'])
    
    # Create a PDF file with high-res images
    with PdfPages(output_path) as pdf:
        # First page: Original molecule
        fig = plt.figure(figsize=(12, 16), dpi=300)  # Higher DPI for better quality
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        
        # Molecule visualization - much larger image
        ax1 = plt.subplot(gs[0])
        img = Draw.MolToImage(original_mol, size=(1200, 1200))  # Significantly increased size
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f"Original Molecule", fontsize=16, pad=20)
        
        # Prediction info
        ax2 = plt.subplot(gs[1])
        ax2.axis('off')
        pred_text = [
            f"SMILES: {format_smiles(original['smiles'])}",
            f"Prediction: {original['prediction']['output']:.4f}"
        ]
        ax2.text(0.5, 0.7, '\n'.join(pred_text), ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=300)  # Save with high DPI
        plt.close()
        
        # One page per counterfactual
        for i, cf in enumerate(counterfactuals):
            cf_mol = Chem.MolFromSmiles(cf['smiles'])
            
            # Find changes between original and counterfactual
            changed_atoms, changed_bonds = find_molecular_changes(original_mol, cf_mol)
            has_changes = bool(changed_atoms or changed_bonds)
            
            # Create figure with high resolution
            fig = plt.figure(figsize=(12, 16), dpi=300)
            
            # Use a simpler layout to maximize molecule size
            gs = gridspec.GridSpec(3, 2, height_ratios=[5, 5, 2], width_ratios=[1, 1])
            
            # Original molecule - larger size
            ax1 = plt.subplot(gs[0, :])
            ax1.set_title("Original Molecule", fontsize=16, pad=20)
            img_orig = Draw.MolToImage(original_mol, size=(1200, 800))
            ax1.imshow(img_orig)
            ax1.axis('off')
            
            # Counterfactual molecule with highlights - larger size
            ax2 = plt.subplot(gs[1, :])
            if has_changes:
                # Determine if this is a positive change (higher prediction)
                is_positive = cf['prediction']['output'] > original['prediction']['output']
                diff_color = 'positive' if is_positive else 'negative'
                
                img_cf = draw_molecule_with_highlights(
                    cf_mol, 
                    changed_atoms, 
                    changed_bonds, 
                    mol_size=(1200, 800),
                    diff_color=diff_color
                )
                
                change_type = "positive" if is_positive else "negative"
                title = f"Counterfactual #{i+1} (with highlighted {change_type} changes)"
            else:
                img_cf = Draw.MolToImage(cf_mol, size=(1200, 800))
                title = f"Counterfactual #{i+1} (No structural changes)"
            
            ax2.imshow(img_cf)
            ax2.axis('off')
            ax2.set_title(title, fontsize=16, pad=20)
            
            # Prediction and feature info
            ax3 = plt.subplot(gs[2, :])
            ax3.axis('off')
            
            pred_diff = cf['prediction']['difference']
            reward = cf['reward'] if 'reward' in cf else 'N/A'
            
            # Format SMILES to be more readable
            # def format_smiles(smiles, max_len=80):
            #     if len(smiles) <= max_len:
            #         return smiles
            #     chunks = []
            #     for i in range(0, len(smiles), max_len):
            #         chunks.append(smiles[i:i+max_len])
            #     return '\n'.join(chunks)
            
            info_text = [
                f"SMILES: {format_smiles(cf['smiles'])}",
                f"Original Prediction: {original['prediction']['output']:.4f}",
                f"Counterfactual Prediction: {cf['prediction']['output']:.4f}",
                f"Difference: {pred_diff:+.4f}",
                f"Reward: {reward}"
            ]
            
            ax3.text(0.5, 0.7, '\n'.join(info_text), ha='center', va='center', fontsize=14)
            
            plt.suptitle(f"Counterfactual Analysis for Battery Capacity", fontsize=18, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
            
            pdf.savefig(fig, dpi=300)  # Save with high DPI
            plt.close()
        
        # Add feature importance summary if available
        if data['feature_diffs']:
            # Calculate average changes across all counterfactuals
            avg_changes = np.mean([diff for diff in data['feature_diffs']], axis=0)
            
            # Get top features by absolute magnitude
            top_n = min(20, len(avg_changes))
            sorted_indices = np.argsort(np.abs(avg_changes))[::-1]
            top_indices = sorted_indices[:top_n]
            
            # Create figure with high resolution
            fig = plt.figure(figsize=(12, 16), dpi=300)
            
            # Bar chart of top features
            plt.bar(
                [f"Feature {i+1}" for i in top_indices], 
                [avg_changes[i] for i in top_indices],
                color=['green' if avg_changes[i] > 0 else 'red' for i in top_indices],
                width=0.7  # Wider bars for visibility
            )
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title('Top Feature Changes Across All Counterfactuals', fontsize=18)
            plt.xlabel('MACCS Fingerprint Features', fontsize=16)
            plt.ylabel('Average Change', fontsize=16)
            plt.xticks(rotation=90, fontsize=14)
            plt.yticks(fontsize=14)
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)  # Save with high DPI
            plt.close()
    
    print(f"High-resolution PDF created at: {output_path}")
    print(f"SMILES saved to: {smiles_path}")
    
    # Also save SVG versions of the molecules if you want vector graphics
    svg_dir = os.path.splitext(output_path)[0] + '_svg'
    os.makedirs(svg_dir, exist_ok=True)
    
    # Save original as SVG
    orig_svg = os.path.join(svg_dir, 'original.svg')
    drawer = rdMolDraw2D.MolDraw2DSVG(800, 800)
    drawer.DrawMolecule(original_mol)
    drawer.FinishDrawing()
    with open(orig_svg, 'w') as f:
        f.write(drawer.GetDrawingText())
    
    # Save each counterfactual as SVG
    for i, cf in enumerate(counterfactuals):
        cf_mol = Chem.MolFromSmiles(cf['smiles'])
        cf_svg = os.path.join(svg_dir, f'counterfactual_{i+1}.svg')
        
        # Get changes
        changed_atoms, changed_bonds = find_molecular_changes(original_mol, cf_mol)
        
        # Draw with highlights if there are changes
        drawer = rdMolDraw2D.MolDraw2DSVG(800, 800)
        if changed_atoms or changed_bonds:
            highlight_atom_dict = {idx: (0.9, 0.2, 0.2, 0.6) for idx in changed_atoms}
            highlight_bond_dict = {idx: (0.9, 0.2, 0.2, 0.6) for idx in changed_bonds}
            drawer.DrawMolecule(
                cf_mol, 
                highlightAtoms=list(changed_atoms),
                highlightBonds=list(changed_bonds),
                highlightAtomColors=highlight_atom_dict,
                highlightBondColors=highlight_bond_dict
            )
        else:
            drawer.DrawMolecule(cf_mol)
        
        drawer.FinishDrawing()
        with open(cf_svg, 'w') as f:
            f.write(drawer.GetDrawingText())
    
    print(f"SVG images saved to: {svg_dir}")
    return output_path, smiles_path, svg_dir

# Example usage
if __name__ == "__main__":
    # 1, 3, 5
    work_dir = os.getcwd()
    sample_dir = os.path.join(work_dir, 'runs/battery/test/meg_output/1')
    json_path = os.path.join(sample_dir, 'data.json')
    plots_dir = os.path.join(sample_dir, 'cf_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    data = load_meg_data(json_path)
    
    # Create PDF with counterfactuals
    pdf_path = os.path.join(plots_dir, 'counterfactuals.pdf')
    create_cf_pdf(data, pdf_path)
    
    print(f"PDF created at: {pdf_path}")
    print(f"Original molecule target: {data['original']['prediction']['output']}")
    print(f"Found {len(data['counterfactuals'])} counterfactuals")
    
    # Show counterfactuals
    for i, cf in enumerate(data['counterfactuals']):
        print(f"\nCounterfactual #{i+1}:")
        print(f"  Reward: {cf.get('reward', 'N/A')}")
        print(f"  Prediction: {cf['prediction']['output']:.2f}")
        print(f"  Difference: {cf['prediction']['difference']:+.2f}")