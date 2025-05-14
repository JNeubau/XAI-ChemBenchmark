### output/data.json

- marker: 'og' for original, 'cf' for counterfactuals
- smiles: The SMILES string representation of the molecule
- prediction: Prediction information including the model output
- reward: For counterfactuals, the reward value (higher is better)
- reward_pred: Component of reward based on prediction difference
- reward_sim: Component of reward based on similarity
- features: The feature values (MACCS fingerprints)