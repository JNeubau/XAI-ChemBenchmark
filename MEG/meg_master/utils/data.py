import torch
import random
import os
import os.path as osp
import pandas as pd

from torch_geometric.data import DataLoader, InMemoryDataset, Dataset
from torch.nn import functional as F
from utils.molecules import check_molecule_validity, pyg_to_mol_tox21, pyg_to_mol_esol, mol_from_smiles, mol_to_smiles, mol_to_esol_pyg
from torch_geometric.datasets import TUDataset, MoleculeNet
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.datasets.molecule_net import x_map, e_map

def pre_transform(sample, n_pad):
    sample.x = F.pad(sample.x, (0,n_pad), "constant")
    # mol = mol_from_smiles(mol_to_smiles(pyg_to_mol_tox21(sample)))
    # sample = mol_to_esol_pyg(mol)
    # sample.smiles = mol_to_smiles(sample)
    return sample

def get_split(dataset_name, split, experiment):

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


    # ds.data, ds.slices = torch.load(f"runs_meg/{dataset_name.lower()}/{experiment}/splits/{split}.pth")
    # Load the split data
    split_file = f"runs_meg/{dataset_name.lower()}/{experiment}/splits/{split}.pth"
    
    if dataset_name.lower() == 'battery':
        # For battery, we saved the collated data directly
        loaded_data = torch.load(split_file)
        ds.data, ds.slices = loaded_data[0], loaded_data[1]
    else:
        # For other datasets, we saved a tuple (data, slices)
        ds.data, ds.slices = torch.load(split_file)

    return ds


def preprocess(dataset_name, experiment_name, batch_size):
    return _PREPROCESS[dataset_name.lower()](experiment_name, batch_size)

def _preprocess_battery(experiment_name, batch_size):    
    # Create directory structure if needed
    os.makedirs(f'runs_meg/battery/{experiment_name}/splits', exist_ok=True)
    
    # Load data from CSV
    csv_path = osp.join(os.getcwd(), 'data', 'new_maccs_merged.csv')
    df = pd.read_csv(csv_path)
    
    # Extract features and target variable
    # Assuming the last column is the target and all others are features
    # Adjust this according to your actual CSV structure
    fingerprint_cols = [col for col in df.columns if col.startswith('maccsfinger')]
    # feature_cols = ['Unnamed: 0'] + fingerprint_cols  # Include ID column
    # ids = df['Unnamed: 0'].values  # Extract IDs from first column
    smiles_col = df['smiles'].values   # Numerical target (smiles)
    
    features = df[fingerprint_cols].values    # All fingerprint features
    targets = df['capacity_max'].values   # Numerical target (capacity)
    
    num_features = features.shape[1]
    print(f"Number of fingerprint features: {num_features}")
    
    # Convert to PyG data format
    data_list = []
    for i in range(len(df)):
        # Create a Data object for each row
        # For molecules, typically we'd have node features, edge indices, etc.
        # But for MACCS fingerprints (which are flat features), we'll create a graph with a single node
        x = torch.FloatTensor(features[i].reshape(1, -1))  # Reshape to [1, num_features]
        y = torch.FloatTensor([targets[i]])  # Target value
        
        # Create a Data object
        data = Data(x=x, y=y)
        # data.battery_id = str(ids[i])  # Store the ID as an attribute
        data.smiles = str(smiles_col[i])  # Store the SMILES string as an attribute
        data_list.append(data)
    
    # Shuffle the data
    random.shuffle(data_list)
    
    # Split into train/val/test
    n = len(data_list) // 10
    train_data = data_list[n:]
    val_data = data_list[:n]
    test_data = train_data[:n]
    train_data = train_data[n:]
    
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
    
    # Save the splits
    torch.save(train_collated, f'runs_meg/battery/{experiment_name}/splits/train.pth')
    torch.save(val_collated, f'runs_meg/battery/{experiment_name}/splits/val.pth')
    torch.save(test_collated, f'runs_meg/battery/{experiment_name}/splits/test.pth')
    
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
    
    # Create the final dataset objects for the DataLoader
    train_dataset = BatteryInMemory(train_collated[0], train_collated[1])
    val_dataset = BatteryInMemory(val_collated[0], val_collated[1])
    test_dataset = BatteryInMemory(test_collated[0], test_collated[1])
    
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
    

def _preprocess_tox21(experiment_name, batch_size):

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


def _preprocess_esol(experiment_name, batch_size):

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
