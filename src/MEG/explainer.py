import random
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.tensorboard import SummaryWriter

from src.MEG.agent import MegAgent
from src.MEG.environment import SortedQueue
from src.MEG.regression_environment import RegressionEnvironment

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)


class MegRegressionExplainer:
    """
    A class to explain the predictions of a regression model using MEG
    """
    def __init__(
            self,
            writer: SummaryWriter,
            target_diff: float,
            transition_point: float,
            env_params: dict,
            agent_params: dict,
            samples: int = 10,
            epochs: int = 10,
    ):
        self.writer = writer
        self.target_diff = target_diff
        self.samples = samples
        self.env_params = env_params
        self.agent_params = agent_params
        self.transition_point = transition_point
        self.epochs = epochs

    def action_encoder(self, action: Any):
        return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(action), self.env_params['fp_rad'], self.env_params['fp_len'], bitInfo=None)

    def train(self, environment: RegressionEnvironment, queue: SortedQueue, sample: int):
        """
        Train the MEG explainer.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        agent = MegAgent(
            num_input=self.env_params['fp_len'] + 1,
            num_output=1,
            lr=self.agent_params.get('lr', 1e-3),
            replay_buffer_size=self.agent_params.get('replay_buffer_size', 10),
            device=device
        )
        eps = 1.0
        batch_losses = []
        episode = 0
        it = 0

        while episode < self.epochs:
            steps_left = self.agent_params['max_steps_per_episodes'] - environment.counter
            valid_actions = list(environment.get_valid_actions())

            if not valid_actions:
                print(f"No valid actions available at episode {episode}, reinitializing environment")
                environment.initialize()
                episode += 1  # Count this as a completed episode
                continue  # Skip to the next iteration

            observations = np.vstack(
                [
                    np.append(self.action_encoder(action), steps_left)
                    for action in valid_actions
                ]
            )
            observations = torch.as_tensor(observations).float()
            a = agent.action_step(observations, eps)
            action = valid_actions[a]

            result = environment.step(action)

            action_embedding = np.append(
                self.action_encoder(action),
                steps_left
            )
            _, out, done = result

            self.writer.add_scalar(f'{self.agent_params['exp_name']}/reward', out['reward'], it)
            self.writer.add_scalar(f'{self.agent_params['exp_name']}/prediction', out['reward_pred'], it)
            self.writer.add_scalar(f'{self.agent_params['exp_name']}/similarity', out['reward_sim'], it)

            steps_left = self.agent_params['max_steps_per_episodes'] - environment.counter

            valid_next_actions = list(environment.get_valid_actions())
            if valid_next_actions:
                action_embeddings = np.vstack(
                    [
                        np.append(self.action_encoder(action), steps_left)
                        for action in valid_next_actions
                    ]
                )
            else:
                # If no valid actions, create a dummy action embedding with zeros
                # This allows training to continue but signals that no further actions are possible
                print(f"No valid next actions available at episode {episode}, step {environment.counter}")
                action_embeddings = np.zeros((1, self.env_params['fp_len'] + 1))
                done = True  # Force episode to end
            agent.replay_buffer.push(
                torch.as_tensor(action_embedding).float(),
                torch.as_tensor(out['reward']).float(),
                torch.as_tensor(action_embeddings).float(),
                float(result.terminated)
            )
            if it % self.agent_params['update_interval'] == 0 and len(agent.replay_buffer) >= self.agent_params['batch_size']:
                loss = agent.train_step(
                    self.agent_params['batch_size'],
                    self.agent_params['gamma'],
                    self.agent_params['polyak']
                )
                loss = loss.item()
                batch_losses.append(loss)

            it += 1

            if done:
                episode += 1

                print(
                    f'({sample}) Epoch {episode}> Reward = {out["reward"]:.4f} (pred: {out["reward_pred"]:.4f}, sim: {out["reward_sim"]:.4f})')

                queue.insert({
                    'marker': 'cf',
                    'id': action,
                    **out
                })

                eps *= 0.9987

                batch_losses = []
                environment.initialize()


    def explain(self, model_to_explain: Any, smiles: str, sample: int) -> list:
        """
        Generate explanations for the given SMILES string.
        :param model_to_explain: The regression model to explain.
        :param smiles: The SMILES representation of the molecule to explain.
        :return: Explanations for the model's prediction.
        """

        mol = Chem.MolFromSmiles(smiles)
        atoms_unique = np.unique([atom.GetSymbol() for atom in mol.GetAtoms()])
        cf_queue = SortedQueue(self.samples, sort_predicate=lambda x: x['reward'])
        env_params = {
            **self.env_params,
            'transition_point': self.transition_point,
            'atom_types': atoms_unique,
            'model_to_explain': model_to_explain,
            'target_diff': self.target_diff,
            'original_molecule': smiles
        }
        cf_env = RegressionEnvironment(**env_params)
        cf_env.initialize()

        self.train(cf_env, queue=cf_queue, sample=sample)
        overall_queue = [{
            'molecule': cf_env.original_molecule_dict,
            'marker': 'og',
            'smiles': smiles,
            'encoding': cf_env.original_molecule_dict['x'],
            'prediction': {
                'type': 'regression',
                'output': cf_env.origin_pred,
                'original': cf_env.origin_pred,
                'difference': 0,
                'target_difference': cf_env.target_diff
            },
        }]
        overall_queue.extend(cf_queue.data_)
        return overall_queue

