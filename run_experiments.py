import os

def mainXaiFlow(dataset_name='battery', experiment_name='test', num_folds=5, data_file='', seed=42):
    model_trained = False
    methods = ['SHAP', 'SHAP_IQ', 'MEG']    # MEG must run after training model
    # methods = ['MEG']    # MEG must run after training model
    
    for method in methods:  
        if method == 'MEG':
            home_dir = os.path.expanduser("~")
            os.system(f"{os.path.join(home_dir, '.conda', 'envs', 'meg', 'python')} {os.path.join(os.getcwd(), 'MEG', 'meg_master', 'main_meg')}.py --dataset_name {dataset_name} --experiment_name {experiment_name} --fold {num_folds} --seed {seed} --data_file {data_file} --train_rfreg")
        elif method in ['SHAP', 'SHAP_IQ']:                        
            if not model_trained:   # global
                os.system(f"{os.path.join(os.getcwd(), '.venv', 'Scripts', 'python')} {os.path.join(os.getcwd(), 'XAIFlow', 'main')}.py --dataset_name {dataset_name} --experiment_name {experiment_name} --fold {num_folds} --model {method} --max_order 1 --seed {seed} --train_rfreg")
            else:
                os.system(f"{os.path.join(os.getcwd(), '.venv', 'Scripts', 'python')} {os.path.join(os.getcwd(), 'XAIFlow', 'main')}.py --dataset_name {dataset_name} --experiment_name {experiment_name} --fold {num_folds} --model {method} --max_order 1 --seed {seed}")
            print(method, ' done')
            model_trained = True
            
        elif method in ['LIME', 'MMACE']:
            print(method, ' done')
            model_trained = True

if __name__ == '__main__':
    mainXaiFlow(dataset_name='battery', 
                experiment_name='new_test', 
                num_folds=5, 
                data_file=os.path.join(os.getcwd(), 'data', 'new_maccs_merged.csv'), 
                seed=42)