import os
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports instead of relative ones
from train_meg_v2 import main as train_meg
from train_RF import main as train_RF
from megplots import main as megplots
import json
import argparse

def mainMegFlow(train_RF_again = True, dataset_name='battery', experiment_name='test', num_folds=5, data_file='', seed=42):
    if train_RF_again:
        train_RF(data_file=data_file,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            n_estimators=100,
            max_depth=None,
            batch_size=32,
            folds=num_folds,
            seed=seed)

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
                    seed=seed,
                    fold=fold)
                megplots(dataset_name=dataset_name, experiment_name=experiment_name, sample=sam, fold=fold)
            except Exception as e:
                print(f"Error processing sample {sam} in fold {fold}: {e}")
                continue  
    
def mainXaiFlow(train_RF_again: bool = True, dataset_name='battery', experiment_name='test', num_folds=5, data_file='', seed=42):
    # if train_RF_again:
    #     train_RF(data_file=data_file,
    #         dataset_name=dataset_name,
    #         experiment_name=experiment_name,
    #         n_estimators=100,
    #         max_depth=None,
    #         batch_size=32,
    #         folds=num_folds,
    #         seed=42)
    
    # split_dir = os.path.join('RFReg', experiment_name, 'folds')
    # split_file = os.path.join(split_dir, 'split_info.json')

    # with open(split_file, 'r') as f:
    #     split_data_list = json.load(f)
        
    # split_data = {}
    # for entry in split_data_list:
    #     fold = entry['fold']
    #     if fold not in split_data:  # Only add each fold once
    #         split_data[fold] = entry['data']
            
    # print(f"Loaded split information for {len(split_data)} folds")  
    
    model_trained = False
    methods = ['SHAP', 'SHAP_IQ', 'MEG']    # MEG must run after training model
    for method in methods:  
        if method == 'MEG':
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
                            seed=seed,
                            fold=fold)
                        megplots(dataset_name=dataset_name, experiment_name=experiment_name, sample=sam, fold=fold)
                    except Exception as e:
                        print(f"Error processing sample {sam} in fold {fold}: {e}")
                        continue
        elif method in ['SHAP', 'SHAP_IQ']:                        
            if not model_trained:   # global
                os.system(f"{os.path.join(os.getcwd(), '.venv', 'Scripts', 'python')} {os.path.join(os.getcwd(), 'XAIFlow', 'main')}.py --dataset_name {dataset_name} --experiment_name {experiment_name} --fold {num_folds} --model {method} --max_order 1 --seed {seed} --train_rfreg")
            else:
                os.system(f"{os.path.join(os.getcwd(), '.venv', 'Scripts', 'python')} {os.path.join(os.getcwd(), 'XAIFlow', 'main')}.py --dataset_name {dataset_name} --experiment_name {experiment_name} --fold {num_folds} --model {method} --max_order 1 --seed {seed}")
            print(method, ' done')
            model_trained = True
            
        elif method in ['LIME', 'MMACE']:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run XAI Flow with specified parameters')
    parser.add_argument('--dataset_name', type=str, default='battery', help='Name of the dataset (default: battery)')
    parser.add_argument('--experiment_name', type=str, default='test', help='Name of the experiment (default: test)')
    parser.add_argument('--fold', type=int, default=5, help='Number of folds to process (default: 5)')
    parser.add_argument('--local', action='store_true', default=False, help='Use local explanations (default: False)')
    parser.add_argument('--train_rfreg', action='store_true', default=False, help='Train new model (default: False)')
    parser.add_argument('--seed', type=int, default=42, help='Set seed value (default: 42)')
    parser.add_argument('--data_file', type=str, default=os.path.join(os.getcwd(), 'data', 'new_maccs_merged.csv'), help='Path to the data file (default: new_maccs_merged.csv)')
    
    args = parser.parse_args()
    
    mainMegFlow(train_RF_again=True, 
                dataset_name=args.dataset_name, 
                experiment_name=args.experiment_name, 
                num_folds=args.fold, 
                data_file=args.data_file, 
                seed=args.seed)
    # mainXaiFlow(train_RF_again=True, 
    #             dataset_name='battery', 
    #             experiment_name='new_test', 
    #             num_folds=5, 
    #             data_file=os.path.join(os.getcwd(), 'data', 'new_maccs_merged.csv'), 
    #             seed=42)