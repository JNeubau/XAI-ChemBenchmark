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

    base_path = osp.join(os.getcwd(), 'RFReg', experiment, 'folds')

    # split_file = f"runs_meg/{dataset_name.lower()}/{experiment}/splits/{split}.pth"
    # Construct the file path based on dataset and fold
    if dataset_name.lower() == 'battery':
        # Use the fold for battery dataset
        split_file = f"{base_path}/{split}_{fold}.pth"
        # split_file = f"runs_meg/{dataset_name.lower()}/{experiment}/splits/{split}_{fold}.pth"
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


def preprocess(data_file, dataset_name, experiment_name, batch_size, folds, seed=0):
    return _PREPROCESS[dataset_name.lower()](data_file, experiment_name, batch_size, folds, seed)

def _preprocess_battery(data_file, experiment_name, batch_size, folds, seed=0):   
    folds_dir = osp.join(os.getcwd(), 'RFReg', experiment_name, 'folds')
    os.makedirs(folds_dir, exist_ok=True)
    
    df = pd.read_csv(data_file)
    
    # Extract features and target variable
    fingerprint_cols = [col for col in df.columns if col.startswith('maccsfinger')]
    # feature_cols = ['Unnamed: 0'] + fingerprint_cols  # Include ID column
    # ids = df['Unnamed: 0'].values  # Extract IDs from first column
    smiles_col = df['smiles'].values   # Numerical target (smiles)
    
    features = df[fingerprint_cols].values    # All fingerprint features
    targets = df['capacity_max'].values   # Numerical target (capacity)
    
    num_features = features.shape[1]
    
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
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Get your dataset indices
    indices = list(range(len(data_list)))

    # Generate the folds
    folds_list = []
    for _, test_idx in kf.split(indices):
        folds_list.append([data_list[i] for i in test_idx])
    
     # For each fold, create train/val/test splits
    for fold_idx in range(folds):
        # Use current fold as test set
        test_data = folds_list[fold_idx]
        
        # Combine remaining folds for train/val
        remaining_data = []
        for i in range(folds):
            if i != fold_idx:
                remaining_data.extend(folds_list[i])
        
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

        json_path = folds_dir + '/split_info.json'
        if not os.path.exists(json_path):
            split_data = []
        else:
            with open(json_path, 'r') as f:
                split_data = json.load(f)

        split_data.append(split_info)

        if os.path.exists(json_path):
            os.remove(json_path)
            
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
        torch.save(train_collated, f'{folds_dir}/train_{fold_idx}.pth')
        torch.save(val_collated, f'{folds_dir}/val_{fold_idx}.pth')
        torch.save(test_collated, f'{folds_dir}/test_{fold_idx}.pth')
    
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
    # train_collated = torch.load(f'{folds_dir}/train_0.pth')
    # val_collated = torch.load(f'{folds_dir}/val_0.pth')
    # test_collated = torch.load(f'{folds_dir}/test_0.pth')
    
    # # Create the final dataset objects for the DataLoader
    # train_dataset = BatteryInMemory(train_collated[0], train_collated[1])
    # val_dataset = BatteryInMemory(val_collated[0], val_collated[1])
    # test_dataset = BatteryInMemory(test_collated[0], test_collated[1])
    
    create_smiles_mapping('battery', experiment_name, folds)
     
    return get_battery_loaders(experiment_name, batch_size, fold=0)
    
def _preprocess_battery2(data_file, experiment_name, batch_size, folds, seed=0):   
    folds_dir = osp.join(os.getcwd(), 'RFReg', experiment_name, 'folds')
    os.makedirs(folds_dir, exist_ok=True)
    
    # Load the dataframe from CSV
    df = pd.read_csv(data_file)
    
    # Extract features and target variable
    fingerprint_cols = [col for col in df.columns if col.startswith('maccsfinger')]
    smiles_col = df['smiles'].values   # SMILES strings
    features = df[fingerprint_cols].values    # All fingerprint features
    targets = df['capacity_max'].values   # Numerical target (capacity)
    
    num_features = features.shape[1]
    
    # Convert to PyG data format
    data_dict = {}
    for i in range(len(df)):
        # Create a Data object for each row
        x = torch.FloatTensor(features[i].reshape(1, -1))  # Reshape to [1, num_features]
        y = torch.FloatTensor([targets[i]])  # Target value
        
        # Create a Data object
        data = Data(x=x, y=y)
        data.smiles = str(smiles_col[i])
        data.original_idx = i  # Store the original index
        data_dict[i] = data
    
    # Clear split_info.json if it exists
    json_path = folds_dir + '/split_info.json'
    if os.path.exists(json_path):
        os.remove(json_path)
    
    split_data = []
    
    # For each fold, load indices from txt files and create datasets
    for fold_idx in range(folds):
        train_indices_path = os.path.join(folds_dir, f'train_{fold_idx}.txt')
        test_indices_path = os.path.join(folds_dir, f'test_{fold_idx}.txt')
        
        # Check if files exist
        if not os.path.exists(train_indices_path) or not os.path.exists(test_indices_path):
            print(f"Warning: Missing train/test files for fold {fold_idx}")
            continue
        
        # Load indices
        try:
            train_indices = pd.read_csv(train_indices_path, header=None).to_numpy().flatten()
            test_indices = pd.read_csv(test_indices_path, header=None).to_numpy().flatten()
            
            # Convert to integers
            train_indices = [int(idx) for idx in train_indices]
            test_indices = [int(idx) for idx in test_indices]
            
            # Create SMILES mapping for this fold
            smiles_mapping_fold = {}
            for i, idx in enumerate(train_indices):
                if idx in data_dict:
                    key = f"{fold_idx}_{i}"
                    smiles_mapping_fold[key] = data_dict[idx].smiles
            
            for i, idx in enumerate(test_indices):
                if idx in data_dict:
                    key = f"{fold_idx}_{i + len(train_indices)}"
                    smiles_mapping_fold[key] = data_dict[idx].smiles
                    
            # Create data lists for this fold using the loaded indices
            train_data = []
            for idx in train_indices:
                if idx in data_dict:
                    train_data.append(data_dict[idx])
                else:
                    print(f"Warning: Index {idx} not found in data")
            
            test_data = []
            for idx in test_indices:
                if idx in data_dict:
                    test_data.append(data_dict[idx])
                else:
                    print(f"Warning: Index {idx} not found in data")
            
            print(f"Fold {fold_idx}: Train: {len(train_data)}, Test: {len(test_data)}")
            if len(train_data) + len(test_data) < len(train_indices) + len(test_indices):
                print(f"Warning: Some indices were not found in the data")
                
            
            # Save the number of entries in each split to a JSON file
            split_info = {
                "fold": fold_idx,
                "data": {
                    "train_size": len(train_data),
                    "val_size": 0,  # No validation set
                    "test_size": len(test_data)
                }
            }
            split_data.append(split_info)
            
            # Use the collate functionality directly from InMemoryDataset
            train_collated = InMemoryDataset.collate(train_data)
            test_collated = InMemoryDataset.collate(test_data)
            
            # Create empty validation set
            empty_data = Data(x=torch.zeros((1, num_features)), y=torch.zeros(1))
            val_collated = InMemoryDataset.collate([empty_data])
            
            # Save the splits with fold index
            torch.save(train_collated, f'{folds_dir}/train_{fold_idx}.pth')
            torch.save(test_collated, f'{folds_dir}/test_{fold_idx}.pth')
            torch.save(val_collated, f'{folds_dir}/val_{fold_idx}.pth')  # Empty validation set
            
        except Exception as e:
            print(f"Error processing fold {fold_idx}: {e}")
            continue
    
    # Write the split info to json
    with open(json_path, 'w') as f:
        json.dump(split_data, f, indent=4)
        
    # Create SMILES mapping to record which molecules are in which folds
    create_smiles_mapping('battery', experiment_name, folds)
    # Return loaders for the first fold by default
    return get_battery_loaders(experiment_name, batch_size, fold=0)

def get_battery_loaders(experiment_name, batch_size, fold=0):
    """
    Load the battery dataset for a specific fold and return DataLoaders.
    
    Args:
        experiment_name: Name of the experiment
        batch_size: Batch size for the DataLoader
        fold: Fold index to load (default: 0)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, num_features, num_classes)
    """
    print(f"Loading battery dataset for fold {fold}...")
    
    # Define the path to the fold data
    folds_dir = osp.join(os.getcwd(), 'RFReg', experiment_name, 'folds')
    
    # Try to load split info to get feature count
    try:
        with open(f'{folds_dir}/split_info.json', 'r') as f:
            split_info = json.load(f)
    except:
        print("Warning: Could not load split info - using default feature count")
        num_features = 166  # Default for MACCS fingerprints
    
    # Class for PyG DataLoader compatibility
    class BatteryInMemory(InMemoryDataset):
        def __init__(self, data, slices, num_features=166):
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
            
        def get(self, idx):
            data = self.data.__class__()
            if hasattr(self.data, '__num_nodes__'):
                data.num_nodes = self.data.__num_nodes__[idx]

            for key in self.slices.keys():
                item, slices = self.data[key], self.slices[key]
                start, end = slices[idx].item(), slices[idx + 1].item()
                if torch.is_tensor(item):
                    s = list(item.size())
                    s[self.data.__cat_dim__(key, item)] = end - start
                    data[key] = item.narrow(self.data.__cat_dim__(key, item), start, end - start)
                elif start + 1 == end:
                    data[key] = item[start]
                else:
                    data[key] = item[start:end]
            return data
    
    try:
        # Load the data for the specified fold
        train_collated = torch.load(f'{folds_dir}/train_{fold}.pth')
        val_collated = torch.load(f'{folds_dir}/val_{fold}.pth')
        test_collated = torch.load(f'{folds_dir}/test_{fold}.pth')
        
        # Determine number of features from the data
        num_features = train_collated[0].x.size(-1)
        
        # Create the dataset objects
        train_dataset = BatteryInMemory(train_collated[0], train_collated[1], num_features)
        val_dataset = BatteryInMemory(val_collated[0], val_collated[1], num_features)
        test_dataset = BatteryInMemory(test_collated[0], test_collated[1], num_features)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        print(f"Successfully loaded data for fold {fold}")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create simple dataset versions
        class SimpleBatteryDataset:
            def __init__(self, dataset):
                self._dataset = dataset
                self._num_features = dataset.num_features
                self._num_classes = dataset.num_classes
                
            def __len__(self):
                return len(self._dataset)
                
            def get(self, idx):
                return self._dataset.get(idx)
                           
            @property
            def num_features(self):
                return self._num_features
                
            @property
            def num_classes(self):
                return self._num_classes
        
        train_simple = SimpleBatteryDataset(train_dataset)
        val_simple = SimpleBatteryDataset(val_dataset)
        test_simple = SimpleBatteryDataset(test_dataset)
        
        return (
            train_loader,
            val_loader,
            test_loader,
            train_simple,
            val_simple,
            test_simple,
            num_features,
            1  # num_classes
        )
             
    except Exception as e:
        print(f"Error loading data for fold {fold}: {e}")
        print(f"Trying to load fold 0 as fallback...")
        
        if fold != 0:
            return get_battery_loaders(experiment_name, batch_size, fold=0)
        else:
            raise ValueError(f"Could not load any fold data for experiment {experiment_name}")
        
    
def create_smiles_mapping(dataset_name, experiment_name, num_folds=5):
    """
    Create a mapping file that maps fold_id: SMILES for all molecules across all folds.
    
    Args:
        dataset_name: Name of the dataset (e.g. 'battery')
        experiment_name: Name of the experiment
        num_folds: Number of folds to process
    """
    print(f"Creating SMILES mapping for {dataset_name}/{experiment_name}...")
    
    # Create mapping directory
    mapping_dir = os.path.join(os.getcwd(), 'RFReg', experiment_name)
    os.makedirs(mapping_dir, exist_ok=True)
    
    # Initialize mapping dictionary
    smiles_mapping = {}
    
    # Process each fold
    for fold in range(num_folds):
        print(f"Processing fold {fold}...")

        # Get the dataset for this split and fold
        dataset = get_split(dataset_name, 'test', experiment_name, fold)
        
        # Extract SMILES strings and create mappings
        for i in range(len(dataset)):
            # Get the molecule
            data = dataset.get(i)
            
            # Get SMILES if available
            if hasattr(data, 'smiles'):
                smiles = data.smiles
                # Create the key as fold_id
                key = f"{fold}_{i}"
                smiles_mapping[key] = smiles
            
    
    # Save the complete mapping to a file
    mapping_file = os.path.join(mapping_dir, "smiles_mapping.txt")
    with open(mapping_file, "w") as f:
        for key, smiles in smiles_mapping.items():
            f.write(f"{key}: {smiles}\n")
    
    print(f"SMILES mapping saved to {mapping_file}")    
    return smiles_mapping

def _preprocess_tox21(data_file, experiment_name, batch_size, folds, seed=0):

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


def _preprocess_esol(data_file, experiment_name, batch_size, folds, seed=0):

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
    'battery': _preprocess_battery2,
}
