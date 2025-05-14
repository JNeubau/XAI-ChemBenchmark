import os
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports instead of relative ones
from train_meg_v2 import main as train_meg
from train_RF import main as train_RF
from megplots import main as megplots
import json
    
    
def mainXaiFlow(train_RF_again: bool = True, dataset_name='battery', experiment_name='test', num_folds=5, data_file=''):
    if train_RF_again:
        train_RF(data_file=data_file,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            n_estimators=100,
            max_depth=None,
            batch_size=32,
            folds=num_folds,
            seed=42)
    
    split_dir = os.path.join('RFReg', experiment_name, 'folds')
    split_file = os.path.join(split_dir, 'split_info.json')

    with open(split_file, 'r') as f:
        split_data_list = json.load(f)
        
    split_data = {}
    for entry in split_data_list:
        fold = entry['fold']
        if fold not in split_data:  # Only add each fold once
            split_data[fold] = entry['data']
            
    print(f"Loaded split information for {len(split_data)} folds")  
    
    methods = ['MEG']
    for method in methods:  
        if method == 'MEG':
            for fold in range(num_folds):    
                num_samples = split_data[fold]['test_size']
                
                sample = list(range(0, num_samples))
                print(f"Starting MEG explainations for fold {fold}...")
                for sam in sample:
                    try:
                        train_meg(dataset=dataset_name,
                            experiment_name=experiment_name,
                            sample=sam,
                            epochs=10, # 5000
                            max_steps_per_episode=1, #6
                            num_counterfactuals=12,
                            fp_length=1024,  
                            fp_radius=2,
                            lr=1e-4,
                            polyak=0.995,
                            gamma=0.95,
                            discount=0.9,
                            replay_buffer_size=10000,
                            batch_size=32, 
                            update_interval=1,
                            allow_no_modification=False,
                            allow_removal=True,
                            allow_node_addition=True,
                            allow_edge_addition=True,
                            allow_bonds_between_rings=True,
                            seed=0,
                            fold=fold)
                        megplots(dataset_name=dataset_name, experiment_name=experiment_name, sample=sam, fold=fold)
                    except Exception as e:
                        print(f"Error processing sample {sam} in fold {fold}: {e}")
                        continue
        elif method in ['SHAP', 'SHAP_IQ']:
            os.system(f"{os.path.join(os.getcwd(), '.venv', 'Scripts', 'python')} {os.path.join(os.getcwd(), 'XAIFlow', 'main')}.py --dataset_name {dataset_name} --experiment_name {experiment_name} --fold {fold}")
            os.system("wait")
            
        elif method in ['LIME', 'MMACE']:
            pass


if __name__ == '__main__':
    mainXaiFlow(True, 'battery', 'rf_test', 5, os.path.join(os.getcwd(), 'data', 'new_maccs_merged.csv'))