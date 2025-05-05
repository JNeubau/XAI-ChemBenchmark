from rdkit import Chem
from rdkit.Chem import Draw
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

def plot_meg_cf(data=None, num_cf=3):
    """
    Plot original molecule and its counterfactuals side by side.
    
    Args:
        data (dict, optional): Data structure from load_meg_data()
        num_cf (int): Number of counterfactuals to display (default: 3)
    """
    
    if not data['original'] or not data['counterfactuals']:
        print("No data available for visualization")
        return
    
    # Get original and top counterfactuals (limit to num_cf)
    original = data['original']
    counterfactuals = data['counterfactuals'][:num_cf]
    
    # Create RDKit molecules
    original_mol = Chem.MolFromSmiles(original['smiles'])
    cf_mols = [Chem.MolFromSmiles(cf['smiles']) for cf in counterfactuals]
    
    # Prepare molecules and labels
    all_mols = [original_mol] + cf_mols
    
    # Create labels with prediction info
    labels = [f"Original\nCapacity: {original['prediction']['output']:.2f}"]
    
    for cf in counterfactuals:
        difference = cf['prediction']['difference']
        labels.append(f"Counterfactual\nCapacity: {cf['prediction']['output']:.2f}\nΔ: {difference:+.2f}")
    
    # Draw molecules
    img = Draw.MolsToGridImage(all_mols, molsPerRow=len(all_mols), subImgSize=(300, 300),
                              legends=labels, useSVG=False)
    
    # Create a figure to display the SVG
    plt.figure(figsize=(5*len(all_mols), 5))
    plt.axis('off')
    plt.imshow(img, interpolation='bilinear')
    plt.title("MEG Counterfactuals for Battery Capacity")
    plt.tight_layout()
    
    return plt

def analyze_feature_changes(data=None, top_n=10):
    """
    Analyze and visualize the key feature changes across counterfactuals.
    
    Args:
        json_path (str, optional): Path to the JSON file with MEG results
        data (dict, optional): Data structure from load_meg_data()
        top_n (int): Number of top features to show
    """    
    if not data['feature_diffs'] or not data['counterfactuals']:
        print("No feature differences available for analysis")
        # Create an empty plot with a message instead of returning None
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No feature differences available", 
                 ha='center', va='center', fontsize=14)
        plt.title('No feature Changes in Counterfactuals')
        plt.axis('off')
        
        # Create an empty dataframe
        df = pd.DataFrame(columns=['Feature', 'Average Change', 'Direction'])
        
        return plt, df
    
    # Aggregate feature changes across all counterfactuals
    avg_changes = np.mean([diff for diff in data['feature_diffs']], axis=0)
    
    # Get indices of top positive and negative changes
    sorted_indices = np.argsort(np.abs(avg_changes))[::-1]
    top_indices = sorted_indices[:top_n]
    
    # Create a dataframe for visualization
    feature_names = [f"Feature {i+1}" for i in range(len(avg_changes))]
    df = pd.DataFrame({
        'Feature': [feature_names[i] for i in top_indices],
        'Average Change': [avg_changes[i] for i in top_indices],
        'Direction': ['Increase' if avg_changes[i] > 0 else 'Decrease' for i in top_indices]
    })
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Feature'], df['Average Change'], color=[
        'green' if change > 0 else 'red' for change in df['Average Change']
    ])
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Top Feature Changes in Counterfactuals')
    plt.xlabel('MACCS Fingerprint Features')
    plt.ylabel('Average Change')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt, df

# Example usage
if __name__ == "__main__":
    # 1, 3, 5
    work_dir = os.getcwd()
    sample_dir = os.path.join(work_dir, 'runs/battery/test/meg_output/3')
    json_path = os.path.join(sample_dir, 'data.json')
    plots_dir = os.path.join(sample_dir, 'cf_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    data = load_meg_data(json_path)
    
    # Plot molecules
    plot_meg_cf(data=data, num_cf=3)
    plt.savefig(os.path.join(plots_dir, 'meg_counterfactuals.png'))
    
    # Analyze feature changes
    plot, feature_df = analyze_feature_changes(data=data)
    plt.savefig(os.path.join(plots_dir, 'feature_changes.png'))
    
    print(f"Original molecule target: {data['original']['prediction']['output']}")
    print(f"Found {len(data['counterfactuals'])} counterfactuals")
    
    # Show top 3 counterfactuals
    for i, cf in enumerate(data['counterfactuals'][:3]):
        print(f"\nCounterfactual #{i+1}:")
        print(f"  Reward: {cf['reward']:.4f}")
        print(f"  Prediction: {cf['prediction']['output']:.2f}")
        print(f"  Difference: {cf['prediction']['difference']:+.2f}")