import torch
import random
import os
import os.path as osp
import pandas as pd

from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader, InMemoryDataset, Dataset
from torch.nn import functional as F
from utils.molecules import check_molecule_validity, pyg_to_mol_tox21, pyg_to_mol_esol, mol_from_smiles, mol_to_smiles, mol_to_esol_pyg
from torch_geometric.datasets import TUDataset, MoleculeNet
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.datasets.molecule_net import x_map, e_map
import json

def pre_transform(sample, n_pad):
    sample.x = F.pad(sample.x, (0,n_pad), "constant")
    # mol = mol_from_smiles(mol_to_smiles(pyg_to_mol_tox21(sample)))
    # sample = mol_to_esol_pyg(mol)
    # sample.smiles = mol_to_smiles(sample)
    return sample

def get_split(dataset_name, split, experiment, fold=0):

    if dataset_name.lower() == 'tox21':
        ds = TUDataset('data/tox21',
                       name='Tox21_AhR_testing',
                       pre_transform=lambda sample: pre_transform(sample, 2))

    elif dataset_name.lower() == 'esol':

        ds = MoleculeNet(
            'data/esol',
            name='ESOL'
        )
    
    elif dataset_name.lower() == 'battery':
        # For battery dataset, create a custom dataset
        class BatteryInMemory(InMemoryDataset):
            def __init__(self):
                super().__init__()
                self._num_classes = 1
            
            @property
            def num_classes(self):
                return self._num_classes
                
            @property
            def num_features(self):
                # This will be set after loading the data
                return self.data.x.size(1)
                
            def __len__(self):
                if hasattr(self, 'slices'):
                    return len(self.slices['x']) - 1
                return 0
        
        ds = BatteryInMemory()


    # split_file = f"runs_meg/{dataset_name.lower()}/{experiment}/splits/{split}.pth"
    # Construct the file path based on dataset and fold
    if dataset_name.lower() == 'battery':
        # Use the fold for battery dataset
        split_file = f"runs_meg/{dataset_name.lower()}/{experiment}/splits/{split}_{fold}.pth"
    else:
        # Other datasets don't use folds
        split_file = f"runs_meg/{dataset_name.lower()}/{experiment}/splits/{split}.pth"
    
    
    if dataset_name.lower() == 'battery':
        # For battery, we saved the collated data directly
        loaded_data = torch.load(split_file)
        ds.data, ds.slices = loaded_data[0], loaded_data[1]
    else:
        # For other datasets, we saved a tuple (data, slices)
        ds.data, ds.slices = torch.load(split_file)

    return ds


def preprocess(dataset_name, experiment_name, batch_size, seed=0):
    return _PREPROCESS[dataset_name.lower()](experiment_name, batch_size, seed)

def _preprocess_battery(experiment_name, batch_size, seed=0):    
    # Create directory structure if needed
    os.makedirs(f'runs_meg/battery/{experiment_name}/splits', exist_ok=True)
    
    # Load data from CSV
    csv_path = osp.join(os.getcwd(), 'data', 'new_maccs_merged.csv')
    df = pd.read_csv(csv_path)
    
    # Extract features and target variable
    fingerprint_cols = [col for col in df.columns if col.startswith('maccsfinger')]
    # feature_cols = ['Unnamed: 0'] + fingerprint_cols  # Include ID column
    # ids = df['Unnamed: 0'].values  # Extract IDs from first column
    smiles_col = df['smiles'].values   # Numerical target (smiles)
    
    features = df[fingerprint_cols].values    # All fingerprint features
    targets = df['capacity_max'].values   # Numerical target (capacity)
    
    num_features = features.shape[1]
    # print(f"Number of fingerprint features: {num_features}")
    
    # Convert to PyG data format
    data_list = []
    for i in range(len(df)):
        # Create a Data object for each row
        x = torch.FloatTensor(features[i].reshape(1, -1))  # Reshape to [1, num_features]
        y = torch.FloatTensor([targets[i]])  # Target value
        
        # Create a Data object
        data = Data(x=x, y=y)
        data.smiles = str(smiles_col[i])  # Store the SMILES string as an attribute
        data_list.append(data)
    
    # Shuffle the data
    random.seed(seed)
    random.shuffle(data_list)
    
    # Create 5 splits for cross-validation
    # total_sample
    
    # Initialize the K-fold splitter
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Get your dataset indices
    indices = list(range(len(data_list)))

    # Generate the folds
    folds = []
    for _, test_idx in kf.split(indices):
        folds.append([data_list[i] for i in test_idx])
    
     # For each fold, create train/val/test splits
    for fold_idx in range(5):
        # Use current fold as test set
        test_data = folds[fold_idx]
        
        # Combine remaining folds for train/val
        remaining_data = []
        for i in range(5):
            if i != fold_idx:
                remaining_data.extend(folds[i])
        
        # Split remaining data into train/val (90/10)
        val_size = len(remaining_data) // 10
        val_data = remaining_data[:val_size]
        train_data = remaining_data[val_size:]
        
        print(f"Fold {fold_idx}: Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Save the number of entries in each split to a JSON file
        split_info = {
            "fold": fold_idx,
            "data": {
                "train_size": len(train_data),
                "val_size": len(val_data),
                "test_size": len(test_data)
            }
        }

        json_path = f'runs_meg/battery/{experiment_name}/splits/split_info.json'
        if not os.path.exists(json_path):
            split_data = []
        else:
            with open(json_path, 'r') as f:
                split_data = json.load(f)

        split_data.append(split_info)

        with open(json_path, 'w') as f:
            json.dump(split_data, f, indent=4)
        
        # Create PyG datasets
        class SimpleBatteryDataset:
            def __init__(self, data_list):
                self.data_list = data_list
                self._num_features = num_features
                self._num_classes = 1
                
            def __len__(self):
                return len(self.data_list)
                
            def get(self, idx):
                return self.data_list[idx]
                
            @property
            def num_features(self):
                return self._num_features
                
            @property
            def num_classes(self):
                return self._num_classes
        
        # Create the simple dataset instances
        train = SimpleBatteryDataset(train_data)
        val = SimpleBatteryDataset(val_data)
        test = SimpleBatteryDataset(test_data)
        
        # Use the collate functionality directly from InMemoryDataset
        train_collated = InMemoryDataset.collate(train_data)
        val_collated = InMemoryDataset.collate(val_data)
        test_collated = InMemoryDataset.collate(test_data)
        
        # Save the splits with fold index
        torch.save(train_collated, f'runs_meg/battery/{experiment_name}/splits/train_{fold_idx}.pth')
        torch.save(val_collated, f'runs_meg/battery/{experiment_name}/splits/val_{fold_idx}.pth')
        torch.save(test_collated, f'runs_meg/battery/{experiment_name}/splits/test_{fold_idx}.pth')
    
    
    # # Split into train/val/test
    # n = len(data_list) // 10
    # train_data = data_list[n:]
    # val_data = data_list[:n]
    # test_data = train_data[:n]
    # train_data = train_data[n:]
    
    # # Create PyG datasets
    # class SimpleBatteryDataset:
    #     def __init__(self, data_list):
    #         self.data_list = data_list
    #         self._num_features = num_features
    #         self._num_classes = 1
            
    #     def __len__(self):
    #         return len(self.data_list)
            
    #     def get(self, idx):
    #         return self.data_list[idx]
            
    #     @property
    #     def num_features(self):
    #         return self._num_features
            
    #     @property
    #     def num_classes(self):
    #         return self._num_classes
    
    # # Create the simple dataset instances
    # train = SimpleBatteryDataset(train_data)
    # val = SimpleBatteryDataset(val_data)
    # test = SimpleBatteryDataset(test_data)
    
    # # Use the collate functionality directly from InMemoryDataset
    # train_collated = InMemoryDataset.collate(train_data)
    # val_collated = InMemoryDataset.collate(val_data)
    # test_collated = InMemoryDataset.collate(test_data)
    
    # # Save the splits
    # torch.save(train_collated, f'runs_meg/battery/{experiment_name}/splits/train.pth')
    # torch.save(val_collated, f'runs_meg/battery/{experiment_name}/splits/val.pth')
    # torch.save(test_collated, f'runs_meg/battery/{experiment_name}/splits/test.pth')
    
    # Class for PyG DataLoader compatibility
    class BatteryInMemory(InMemoryDataset):
        def __init__(self, data, slices):
            super(BatteryInMemory, self).__init__()
            self.data = data
            self.slices = slices
            self._num_features = num_features
            self._num_classes = 1
            
        @property
        def num_features(self):
            return self._num_features
            
        @property
        def num_classes(self):
            return self._num_classes
            
        def __len__(self):
            return len(self.slices['x']) - 1
    
    # Load the first fold for default return values
    train_collated = torch.load(f'runs_meg/battery/{experiment_name}/splits/train_0.pth')
    val_collated = torch.load(f'runs_meg/battery/{experiment_name}/splits/val_0.pth')
    test_collated = torch.load(f'runs_meg/battery/{experiment_name}/splits/test_0.pth')
    
    # Create the final dataset objects for the DataLoader
    train_dataset = BatteryInMemory(train_collated[0], train_collated[1])
    val_dataset = BatteryInMemory(val_collated[0], val_collated[1])
    test_dataset = BatteryInMemory(test_collated[0], test_collated[1])
    
    # Also save the full dataset
    full_collated = InMemoryDataset.collate(data_list)
    torch.save(full_collated, f'runs_meg/battery/{experiment_name}/splits/full.pth')
    
    return (
        DataLoader(train_dataset, batch_size=batch_size),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
        train,
        val,
        test,
        num_features,
        1,  # num_classes
    )
    

def _preprocess_tox21(experiment_name, batch_size, seed=0):

    dataset_tr = TUDataset('data/tox21',
                           name='Tox21_AhR_training',
                           pre_transform=lambda sample: pre_transform(sample, 3))

    dataset_vl = TUDataset('data/tox21',
                           name='Tox21_AhR_evaluation',
                           pre_transform=lambda sample: pre_transform(sample, 0))

    dataset_ts = TUDataset('data/tox21',
                           name='Tox21_AhR_testing',
                           pre_transform=lambda sample: pre_transform(sample, 2))

    data_list = (
        [dataset_tr.get(idx) for idx in range(len(dataset_tr))] +
        [dataset_vl.get(idx) for idx in range(len(dataset_vl))] +
        [dataset_ts.get(idx) for idx in range(len(dataset_ts))]
    )

    data_list = list(filter(lambda mol: check_molecule_validity(mol, pyg_to_mol_tox21), data_list))

    POSITIVES = list(filter(lambda x: x.y == 1, data_list))
    NEGATIVES = list(filter(lambda x: x.y == 0, data_list))
    N_POSITIVES = len(POSITIVES)
    N_NEGATIVES = N_POSITIVES
    NEGATIVES = NEGATIVES[:N_NEGATIVES]

    dataset_full = dataset_tr
    data_list = POSITIVES + NEGATIVES
    random.shuffle(data_list)

    n = len(data_list) // 10
    train_data = data_list[n:]
    val_data = data_list[:n]
    test_data = train_data[:n]
    train_data = train_data[n:]

    train = dataset_tr
    val = dataset_vl
    test = dataset_ts

    train.data, train.slices = train.collate(train_data)
    val.data, val.slices = train.collate(val_data)
    test.data, test.slices = train.collate(test_data)

    torch.save((train.data, train.slices), f'runs_meg/tox21/{experiment_name}/splits/train.pth')
    torch.save((val.data, val.slices), f'runs_meg/tox21/{experiment_name}/splits/val.pth')
    torch.save((test.data, test.slices), f'runs_meg/tox21/{experiment_name}/splits/test.pth')

    return (
        DataLoader(train, batch_size=batch_size),
        DataLoader(val,   batch_size=batch_size),
        DataLoader(test,  batch_size=batch_size),
        train,
        val,
        test,
        max(train.num_features, val.num_features, test.num_features),
        train.num_classes,
    )


def _preprocess_esol(experiment_name, batch_size, seed=0):

    dataset = MoleculeNet(
        'data/esol',
        name='ESOL'
    )

    data_list = (
        [dataset.get(idx) for idx in range(len(dataset))]
    )

    random.shuffle(data_list)

    n = len(data_list) // 10

    train_data = data_list[n:]
    val_data = data_list[:n]
    test_data = train_data[:n]
    train_data = train_data[n:]

    train = dataset
    val = dataset.copy()
    test = dataset.copy()

    train.data, train.slices = train.collate(train_data)
    val.data, val.slices = train.collate(val_data)
    test.data, test.slices = train.collate(test_data)

    os.makedirs(f'runs_meg/esol/{experiment_name}/splits', exist_ok=True)
    torch.save((train.data, train.slices), f'runs_meg/esol/{experiment_name}/splits/train.pth')
    torch.save((val.data, val.slices), f'runs_meg/esol/{experiment_name}/splits/val.pth')
    torch.save((test.data, test.slices), f'runs_meg/esol/{experiment_name}/splits/test.pth')


    return (
        DataLoader(train, batch_size=batch_size),
        DataLoader(val,   batch_size=batch_size),
        DataLoader(test,  batch_size=batch_size),
        train,
        val,
        test,
        max(train.num_features, val.num_features, test.num_features),
        train.num_classes,
    )


_PREPROCESS = {
    'tox21': _preprocess_tox21,
    'esol': _preprocess_esol,
    'battery': _preprocess_battery,
}
