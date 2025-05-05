import os
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports instead of relative ones
from train_meg_v2 import main as train_meg
from train_RF import main as train_RF
from megplots import main as megplots


def mainXaiFlow(train_RF_again: bool = True, dataset_name='battery', experiment_name='test'):
    if train_RF_again:
        train_RF(dataset_name=dataset_name,
            experiment_name=experiment_name,
            n_estimators=100,
            max_depth=None,
            batch_size=32,
            seed=0)
    
    sample = list(range(4, 9))
    print("Starting MEG explainations...")
    for i in sample:
        train_meg(dataset=dataset_name,
            experiment_name=experiment_name,
            sample=i,
            epochs=500, # 5000
            max_steps_per_episode=10,
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
            seed=0)
        megplots(dataset_name=dataset_name, experiment_name=experiment_name, sample=i)


if __name__ == '__main__':
    mainXaiFlow(False, 'battery', 'test')