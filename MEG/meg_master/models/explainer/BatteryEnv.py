import torch
import numpy as np
import utils
from sklearn.metrics.pairwise import cosine_similarity
from collections import namedtuple
# from models.explainer.utils import StepResult

# Define StepResult locally since it's not available in utils module
StepResult = namedtuple('StepResult', ['state', 'out', 'terminated'])

# Define a custom environment for Battery dataset
class CF_Battery:
    def __init__(self, **kwargs):
        self.original_molecule = kwargs['original_molecule']
        self.model_to_explain = kwargs['model_to_explain']
        self.weight_sim = kwargs['weight_sim']
        self.max_steps = kwargs['max_steps']
        self.num_steps_taken = 0
        self.feature_names = kwargs['feature_names']
        self.current_features = None
        self.molecule_id = getattr(self.original_molecule, 'battery_id', kwargs.get('init_mol', 'battery_sample'))  # Get molecule ID from params
        self.smiles = getattr(self.original_molecule, 'smiles', None)  # Get SMILES from params
        
    def initialize(self):
        self.current_features = self.original_molecule.x.clone()
        self.num_steps_taken = 0
        return self.current_features
        
    def get_valid_actions(self):
        # For battery dataset, valid actions are perturbations of features
        # Here we just create small variations of the current features
        actions = []
        base_features = self.current_features.reshape(-1).numpy()
        
        # Generate 10 random perturbations
        for _ in range(10):
            # Create a random perturbation
            perturb = np.random.normal(0, 0.1, size=base_features.shape)
            # Only modify fingerprint features (boolean 0/1 values)
            perturb = np.round(base_features + perturb).clip(0, 1)
            actions.append(perturb)
        
        return actions
        
    def step(self, action):
        # Update the current features
        self.current_features = torch.tensor(action).float().reshape(self.original_molecule.x.shape)
        self.num_steps_taken += 1
        
        # Get prediction for the new features
        features_flat = self.current_features.reshape(1, -1)
        pred = self.model_to_explain.predict(features_flat.numpy())[0]
        
        # Calculate similarity
        sim = self.calculate_tanimoto_similarity()
        # sim = self.caluclate_euclidean_dist_similarity()
        
        # Calculate reward 
        # For regression, we want predictions that differ significantly
        original_pred = self.model_to_explain.predict(
            self.original_molecule.x.reshape(1, -1).numpy()
        )[0]
        
        # Reward based on prediction difference and similarity
        pred_diff = abs(pred - original_pred)
        reward_pred = np.tanh(pred_diff)  # Normalize to 0-1 range
        reward_sim = sim
        
        reward = (1 - self.weight_sim) * reward_pred + self.weight_sim * reward_sim
        
        # Check if done
        done = (self.num_steps_taken >= self.max_steps)
        
        result = StepResult(
            state=self.current_features,
            out={
                'reward': reward,
                'reward_pred': reward_pred,
                'reward_sim': reward_sim,
                'dataset_id': self.molecule_id,  # Use the ID since we don't have SMILES
                'smiles': self.smiles,
                'pred': float(pred),
                'features': self.current_features.reshape(-1).numpy().tolist()
            },
            terminated=done
        )
        
        return result
    
    def calculate_tanimoto_similarity(self):
        # Calculate Tanimoto similarity
        # For binary vectors like MACCS fingerprints, this is a suitable measure
        current_features_flat = self.current_features.reshape(1, -1).numpy()
        original_features_flat = self.original_molecule.x.reshape(1, -1).numpy()
        
        # Tanimoto coefficient calculation:
        # T(A,B) = |A ∩ B| / |A ∪ B| = (A • B) / (|A|² + |B|² - A • B)
        intersection = np.sum(current_features_flat * original_features_flat)
        union = np.sum(current_features_flat) + np.sum(original_features_flat) - intersection
        
        # Avoid division by zero
        if union == 0:
            tanimoto = 1.0  # If both vectors are all zeros, they're identical
        else:
            tanimoto = intersection / union
        
        # Use tanimoto as similarity (ranges from 0 to 1)
        return tanimoto
    
    def caluclate_euclidean_dist_similarity(self):
        return 1.0 / (1.0 + np.linalg.norm(
            self.current_features.reshape(-1).numpy() - 
            self.original_molecule.x.reshape(-1).numpy()
        ))