import torch
import numpy as np
import json
import os
import networkx as nx
import typer
import random

from models.explainer import CF_Tox21, NCF_Tox21, Agent, CF_Esol, NCF_Esol, CF_Battery
from torch.utils.tensorboard import SummaryWriter
from utils import SortedQueue, morgan_bit_fingerprint, get_split, get_dgn, mol_to_smiles, x_map_tox21, pyg_to_mol_tox21, mol_from_smiles, mol_to_tox21_pyg, pyg_to_mol_battery
from torch.nn import functional as F
from torch_geometric.utils import to_networkx

def battery_1(general_params,
          base_path,
          writer,
          num_counterfactuals,
          original_molecule,
          model_to_explain,
          **args):
    print('Running MEG on Battery dataset...')
    
    # Different prediction approach for Random Forest
    # RF expects flattened features, not graph structure
    features = original_molecule.x.reshape(1, -1)
    print(f"Feature shape for prediction: {features.shape}")            
    
    out = model_to_explain.predict(features.numpy())
    original_encoding = features  # Use the features as encoding since RF doesn't have internal representations
    
    # For regression, we don't have classes, just the predicted value
    pred_value = float(out[0])
    actual_value = float(original_molecule.y.item())
    
    print(f'Original prediction: {pred_value}, Actual: {actual_value}')
    
    # For battery dataset we don't have SMILES, use a placeholder ID
    molecule_id = f"battery_sample_{args['sample']}"
    smiles = getattr(original_molecule, 'smiles', None)
    
    print(f'Battery sample ID: {molecule_id}')
    print(f'Battery SMILES: {smiles}')
    
    # For battery dataset we're using MACCS fingerprints, not atoms
    feature_names = [f'feature_{i}' for i in range(features.shape[1])]
    print(f"Number of features for explanation: {len(feature_names)}")

    params = {
        # General-purpose params
        **general_params,
        'init_mol': molecule_id,  # Use the ID instead of SMILES
        'feature_names': feature_names,  # Use feature names instead of atom types
        # Task-specific params
        'original_molecule': original_molecule,
        'model_to_explain': model_to_explain,
        'weight_sim': 0.2,
        'similarity_measure': 'tanimoto'  # Use euclidean distance for numerical features
    }
    
    cf_queue = SortedQueue(num_counterfactuals, sort_predicate=lambda mol: mol['reward'])
    cf_env = CF_Battery(**params)
    cf_env.initialize()

    def action_encoder(action):
        # For battery, the action is already the feature vector
        return action

    meg_train(writer,
              action_encoder,
              len(feature_names),  # Use the number of features
              cf_env,
              cf_queue,
              marker="cf",
              tb_name="battery",
              id_function=lambda action: molecule_id,  # Use the ID since we don't have SMILES
              args=args)

    overall_queue = []
    overall_queue.append({
        'pyg': original_molecule,
        'marker': 'og',
        'smiles': smiles,  # Use the ID since we don't have SMILES
        'encoding': original_encoding.numpy(),
        'prediction': {
            'type': 'regression',
            'output': float(pred_value),
            'for_explanation': float(actual_value)
        }
    })
    overall_queue.extend(cf_queue.data_)
    
    save_results(base_path, overall_queue, args)

def battery(general_params,
          base_path,
          writer,
          num_counterfactuals,
          original_molecule,
          model_to_explain,
          **args):
    print('Running MEG on battery dataset...')
    
    # First check if we need to create a graph structure
    if not hasattr(original_molecule, 'edge_index') or original_molecule.edge_index is None:
        # Get SMILES if available
        smiles = getattr(original_molecule, 'smiles', None)
        print('original SMILES: ', smiles)
        
        if smiles:
            # Create a graph from SMILES
            from utils.molecules import mol_from_smiles, mol_to_battery_pyg
            rdkit_mol = mol_from_smiles(smiles)
            if rdkit_mol:
                # Extract graph structure from RDKit mol and add it to original_molecule
                pyg_mol = mol_to_battery_pyg(rdkit_mol)
                original_molecule.edge_index = pyg_mol.edge_index
                original_molecule.edge_attr = pyg_mol.edge_attr
            else:
                # Fallback to dummy structure
                num_nodes = original_molecule.x.size(0) if len(original_molecule.x.shape) > 1 else 1
                original_molecule.edge_index = torch.zeros((2, 0), dtype=torch.long)
                original_molecule.edge_attr = torch.zeros((0, 4), dtype=torch.float)
    
    
    # out, (_, original_encoding) = model_to_explain(original_molecule.x, original_molecule.edge_index)
    features = original_molecule.x.reshape(1, -1)
    print(f"Feature shape for prediction: {features.shape}")
    
    # Make prediction with RF model
    out = model_to_explain.predict(features.numpy())
    original_encoding = features  # Use the features as encoding

    # logits = F.softmax(out, dim=-1).detach().squeeze()
    # pred_class = logits.argmax().item()
    pred_value = float(out[0])  # Convert to Python float for easier handling

    # assert pred_class == original_molecule.y.item()
    print(original_molecule)

    # original_molecule.smiles = mol_to_smiles(pyg_to_mol_battery(original_molecule))

    print(f'Molecule: {original_molecule.smiles}')

    # atoms_ = [
    #     x_map_tox21(e).name
    #     for e in np.unique(
    #         [x.tolist().index(1) for x in original_molecule.x.numpy()]
    #     )
    # ]
    atoms_ = np.unique(
        [x.GetSymbol() for x in mol_from_smiles(original_molecule.smiles).GetAtoms()]
    )

    params = {
        # General-purpose params
        **general_params,
        'init_mol': original_molecule.smiles,
        'atom_types': set(atoms_),
        # Task-specific params
        'original_molecule': original_molecule,
        'model_to_explain': model_to_explain,
        'weight_sim': 0.2,
        'similarity_measure': 'tanimoto'
    }

    cf_queue = SortedQueue(num_counterfactuals, sort_predicate=lambda mol: mol['reward'])
    cf_env = CF_Battery(**params)
    cf_env.initialize()

    def action_encoder(action):
        return morgan_bit_fingerprint(action, args['fp_length'], args['fp_radius']).numpy()

    meg_train(writer,
              action_encoder,
              args['fp_length'],
              cf_env,
              cf_queue,
              marker="cf",
              tb_name="battery",
              id_function=lambda action: action,
              args=args)

    overall_queue = []
    overall_queue.append({
        'pyg': original_molecule,
        'marker': 'og',
        'smiles': original_molecule.smiles,
        'encoding': original_encoding.numpy(),
        'prediction': {
            'type': 'regression',
            'output': pred_value,
            'for_explanation': original_molecule.y.item(),
            'class': original_molecule.y.item()
        }
    })
    overall_queue.extend(cf_queue.data_)

    save_results(base_path, overall_queue, args)

def battery_2(general_params,
        base_path,
        writer,
        num_counterfactuals,
        original_molecule,
        model_to_explain,
        **args):
    
    print('Running MEG on ESOL dataset...')
    original_molecule.x = original_molecule.x.float()

    og_prediction, original_encoding = model_to_explain(original_molecule.x, original_molecule.edge_index)
    print(f'Molecule: {original_molecule.smiles}')

    atoms_ = np.unique(
        [x.GetSymbol() for x in mol_from_smiles(original_molecule.smiles).GetAtoms()]
    )

    params = {
        # General-purpose params
        **general_params,
        'init_mol': original_molecule.smiles,
        'atom_types': set(atoms_),
        # Task-specific params
        'model_to_explain': model_to_explain,
        'original_molecule': original_molecule,
        'weight_sim': 0.2,
        'similarity_measure': 'tanimoto',
    }

    cf_queue = SortedQueue(num_counterfactuals, sort_predicate=lambda mol: mol['reward'])
    cf_env = CF_Esol(**params)
    cf_env.initialize()

    def action_encoder(action):
        return morgan_bit_fingerprint(action, args['fp_length'], args['fp_radius']).numpy()

    meg_train(writer,
              action_encoder,
              args['fp_length'],
              cf_env,
              cf_queue,
              marker="cf",
              tb_name="battery",
              id_function=lambda action: action,
              args=args)

    overall_queue = []
    overall_queue.append({
        'pyg': original_molecule,
        'marker': 'og',
        'smiles': original_molecule.smiles,
        'encoding': original_encoding,
        # 'encoding': original_encoding.numpy(),
        'prediction': {
            'type': 'regression',
            'output': og_prediction.squeeze().detach().numpy().tolist(),
            'for_explanation': og_prediction.squeeze().detach().numpy().tolist()
        }
    })
    overall_queue.extend(cf_queue.data_)

    save_results(base_path, overall_queue, args)
    
def tox21(general_params,
          base_path,
          writer,
          num_counterfactuals,
          original_molecule,
          model_to_explain,
          **args):
    print('Running MEG on Tox21 dataset...')
    out, (_, original_encoding) = model_to_explain(original_molecule.x,
                                                   original_molecule.edge_index)

    logits = F.softmax(out, dim=-1).detach().squeeze()
    pred_class = logits.argmax().item()

    assert pred_class == original_molecule.y.item()

    original_molecule.smiles = mol_to_smiles(pyg_to_mol_tox21(original_molecule))

    print(f'Molecule: {original_molecule.smiles}')

    atoms_ = [
        x_map_tox21(e).name
        for e in np.unique(
            [x.tolist().index(1) for x in original_molecule.x.numpy()]
        )
    ]

    params = {
        # General-purpose params
        **general_params,
        'init_mol': original_molecule.smiles,
        'atom_types': set(atoms_),
        # Task-specific params
        'original_molecule': original_molecule,
        'model_to_explain': model_to_explain,
        'weight_sim': 0.2,
        'similarity_measure': 'combined'
    }

    cf_queue = SortedQueue(num_counterfactuals, sort_predicate=lambda mol: mol['reward'])
    cf_env = CF_Tox21(**params)
    cf_env.initialize()

    def action_encoder(action):
        return morgan_bit_fingerprint(action, args['fp_length'], args['fp_radius']).numpy()

    meg_train(writer,
              action_encoder,
              args['fp_length'],
              cf_env,
              cf_queue,
              marker="cf",
              tb_name="tox21",
              id_function=lambda action: action,
              args=args)

    overall_queue = []
    overall_queue.append({
        'pyg': original_molecule,
        'marker': 'og',
        'smiles': original_molecule.smiles,
        'encoding': original_encoding.numpy(),
        'prediction': {
            'type': 'bin_classification',
            'output': logits.numpy().tolist(),
            'for_explanation': original_molecule.y.item(),
            'class': original_molecule.y.item()
        }
    })
    overall_queue.extend(cf_queue.data_)

    save_results(base_path, overall_queue, args)

def esol(general_params,
         base_path,
         writer,
         num_counterfactuals,
         original_molecule,
         model_to_explain,
         **args):
    print('Running MEG on ESOL dataset...')
    original_molecule.x = original_molecule.x.float()

    og_prediction, original_encoding = model_to_explain(original_molecule.x, original_molecule.edge_index)
    print(f'Molecule: {original_molecule.smiles}')

    atoms_ = np.unique(
        [x.GetSymbol() for x in mol_from_smiles(original_molecule.smiles).GetAtoms()]
    )

    params = {
        # General-purpose params
        **general_params,
        'init_mol': original_molecule.smiles,
        'atom_types': set(atoms_),
        # Task-specific params
        'model_to_explain': model_to_explain,
        'original_molecule': original_molecule,
        'weight_sim': 0.2,
        'similarity_measure': 'combined',
    }

    cf_queue = SortedQueue(num_counterfactuals, sort_predicate=lambda mol: mol['reward'])
    cf_env = CF_Esol(**params)
    cf_env.initialize()

    def action_encoder(action):
        return morgan_bit_fingerprint(action, args['fp_length'], args['fp_radius']).numpy()

    meg_train(writer,
              action_encoder,
              args['fp_length'],
              cf_env,
              cf_queue,
              marker="cf",
              tb_name="esol",
              id_function=lambda action: action,
              args=args)

    overall_queue = []
    overall_queue.append({
        'pyg': original_molecule,
        'marker': 'og',
        'smiles': original_molecule.smiles,
        'encoding': original_encoding,
        # 'encoding': original_encoding.numpy(),
        'prediction': {
            'type': 'regression',
            'output': og_prediction.squeeze().detach().numpy().tolist(),
            'for_explanation': og_prediction.squeeze().detach().numpy().tolist()
        }
    })
    overall_queue.extend(cf_queue.data_)

    save_results(base_path, overall_queue, args)

def meg_train(writer,
              action_encoder,
              n_input,
              environment,
              queue,
              marker,
              tb_name,
              id_function,
              args):
    print('Training MEG...')
    device = torch.device("cpu")
    agent = Agent(n_input + 1, 1, device, args['lr'], args['replay_buffer_size'])

    eps = 1.0
    batch_losses = []
    episode = 0
    it = 0

    while episode < args['epochs']:
        steps_left = args['max_steps_per_episode'] - environment.num_steps_taken
        valid_actions = list(environment.get_valid_actions())

        # Check if there are valid actions available
        if not valid_actions:
            print(f"No valid actions available at episode {episode}, reinitializing environment")
            environment.initialize()
            episode += 1  # Count this as a completed episode
            continue      # Skip to the next iteration
        
        observations = np.vstack(
            [
                np.append(action_encoder(action), steps_left)
                for action in valid_actions
            ]
        )

        observations = torch.as_tensor(observations).float()
        a = agent.action_step(observations, eps)
        action = valid_actions[a]

        result = environment.step(action)

        action_embedding = np.append(
            action_encoder(action),
            steps_left
        )

        _, out, done = result

        writer.add_scalar(f'{tb_name}/reward', out['reward'], it)
        writer.add_scalar(f'{tb_name}/prediction', out['reward_pred'], it)
        writer.add_scalar(f'{tb_name}/similarity', out['reward_sim'], it)

        steps_left = args['max_steps_per_episode'] - environment.num_steps_taken

        # Check if there are valid actions after taking a step
        valid_next_actions = list(environment.get_valid_actions())
        if valid_next_actions:
            action_embeddings = np.vstack(
                [
                    np.append(action_encoder(action), steps_left)
                    for action in valid_next_actions
                ]
            )
        else:
            # If no valid actions, create a dummy action embedding with zeros
            # This allows training to continue but signals that no further actions are possible
            print(f"No valid next actions available at episode {episode}, step {environment.num_steps_taken}")
            action_embeddings = np.zeros((1, n_input + 1))
            done = True  # Force episode to end
        
        # action_embeddings = np.vstack(
        #     [
        #         np.append(action_encoder(action), steps_left)
        #         for action in valid_actions
        #     ]
        # )

        agent.replay_buffer.push(
            torch.as_tensor(action_embedding).float(),
            torch.as_tensor(out['reward']).float(),
            torch.as_tensor(action_embeddings).float(),
            float(result.terminated)
        )

        if it % args['update_interval'] == 0 and len(agent.replay_buffer) >= args['batch_size']:
            loss = agent.train_step(
                args['batch_size'],
                args['gamma'],
                args['polyak']
            )
            loss = loss.item()
            batch_losses.append(loss)

        it += 1

        if done:
            episode += 1

            print(f'({args["sample"]}) Episode {episode}> Reward = {out["reward"]:.4f} (pred: {out["reward_pred"]:.4f}, sim: {out["reward_sim"]:.4f})')
            
            # For battery dataset, create encoding from features if available
            if 'features' in out:
                encoding = np.array(out['features'])
            else:
                # Use action_encoder to get encoding
                encoding = action_encoder(action)
                
            queue.insert({
                'marker': marker,
                'id': id_function(action),
                'encoding': encoding,  # Add encoding field
                **out
            })
            # queue.insert({
            #     'marker': marker,
            #     'id': id_function(action),
            #     **out
            # })


            eps *= 0.9987
            # eps = max(eps, 0.05)

            batch_losses = []
            environment.initialize()


def save_results(base_path, queue, args):
    print('Saving results...')
    output_dir = base_path + f"/meg_output/{args['sample']}"
    embedding_dir = output_dir + "/embeddings"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(embedding_dir)

    # Convert queue items to JSON-serializable format
    json_queue = []
    for i, molecule in enumerate(queue):
        # Create a copy to avoid modifying the original
        molecule_copy = molecule.copy()
        
        # Save and remove encoding
        if 'encoding' in molecule_copy:
            np.save(embedding_dir + f"/{i}", molecule_copy.pop('encoding'))
        
        # Remove PyG object
        if 'pyg' in molecule_copy:
            molecule_copy.pop('pyg')
            
        # Convert NumPy values to Python types
        json_molecule = {}
        for key, value in molecule_copy.items():
            if isinstance(value, np.ndarray):
                json_molecule[key] = value.tolist()
            elif isinstance(value, np.number):
                json_molecule[key] = value.item()
            elif isinstance(value, dict):
                # Handle nested dictionaries (like prediction)
                json_nested = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_nested[k] = v.tolist()
                    elif isinstance(v, np.number):
                        json_nested[k] = v.item()
                    else:
                        json_nested[k] = v
                json_molecule[key] = json_nested
            else:
                json_molecule[key] = value
        
        json_queue.append(json_molecule)

    with open(output_dir + "/seed", "w") as outf:
        json.dump(args['seed'], outf)

    with open(output_dir + "/data.json", "w") as outf:
        # Fix: Use json_queue instead of queue
        json.dump(json_queue, outf, indent=2)

def main(dataset: str,
         experiment_name: str = typer.Argument("test"),
         sample: int = typer.Option(0),
         epochs: int = typer.Option(100), # 5000
         max_steps_per_episode: int = typer.Option(2), # 1
         num_counterfactuals: int = typer.Option(10),
         fp_length: int = typer.Option(1024),
         fp_radius: int = typer.Option(2),
         lr: float = typer.Option(1e-4),
         polyak: float = typer.Option(0.995),
         gamma: float = typer.Option(0.95),
         discount: float = typer.Option(0.9),
         replay_buffer_size: int = typer.Option(10000),
         batch_size: int = typer.Option(1),
         update_interval: int = typer.Option(1),
         allow_no_modification: bool = typer.Option(False),
         allow_removal: bool = typer.Option(True),
         allow_node_addition: bool = typer.Option(True),
         allow_edge_addition: bool = typer.Option(True),
         allow_bonds_between_rings: bool = typer.Option(True),
         seed: int = typer.Option(random.randint(0, 2**12))
):

    general_params = {
        # General-purpose params
        'discount_factor': discount,
        'allow_removal': allow_removal,
        'allow_no_modification': allow_no_modification,
        'allow_bonds_between_rings': allow_bonds_between_rings,
        'allow_node_addition': allow_node_addition,
        'allow_edge_addition': allow_edge_addition,
        'allowed_ring_sizes': set([5, 6]),
        'max_steps': max_steps_per_episode,
        'fp_len': fp_length,
        'fp_rad': fp_radius
    }

    dataset = dataset.lower()
    if dataset == 'tox21':
        meg = tox21
    elif dataset == 'esol':
        meg = esol
    elif dataset == 'battery':
        meg = battery

    torch.manual_seed(seed)

    base_path = f'./runs_meg/{dataset.lower()}/{experiment_name}'

    print('num samples test: ', len(get_split(dataset.lower(), 'test', experiment_name)))
    print('num samples: train', len(get_split(dataset.lower(), 'train', experiment_name)))
    print('num samples: val', len(get_split(dataset.lower(), 'val', experiment_name)))
    print('Running meg on dataset: ', dataset)
    meg(general_params,
        base_path,
        SummaryWriter(f'{base_path}/plots'),
        num_counterfactuals,
        get_split(dataset.lower(), 'test', experiment_name)[sample],
        model_to_explain=get_dgn(dataset.lower(), experiment_name),
        experiment_name=experiment_name,
        sample=sample,
        epochs=epochs,
        max_steps_per_episode=max_steps_per_episode,
        fp_length=fp_length,
        fp_radius=fp_radius,
        lr=lr,
        polyak=polyak,
        gamma=gamma,
        discount=discount,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        update_interval=update_interval,
        seed=seed)


if __name__ == '__main__':
    typer.run(main)
