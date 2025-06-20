import re
import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to your txt file
text_file = '2_7_terminam_output.txt'
base_path = os.path.join(os.getcwd(), 'results', 'final')
text_path = os.path.join(base_path, text_file)
# print(text_path)

# Regular expression to parse each line
pattern = re.compile(
    r"\((\d+)\) Episode (\d+)> Reward = ([\d\.\-e]+) \(pred: ([\d\.\-e]+), sim: ([\d\.\-e]+)\)"
)

data = []

# Read and parse the file
with open(text_path, 'r') as f:
    for line in f:
        match = pattern.match(line.strip())
        if match:
            run, episode, reward, pred, sim = match.groups()
            data.append({
                'run': int(run),
                'episode': int(episode),
                'reward': float(reward),
                'pred': float(pred),
                'sim': float(sim)
            })

# Create DataFrame
df = pd.DataFrame(data)

# Plot: Typical RL plot is reward vs episode
plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['reward'], label='Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(base_path + f'/reward_per_episode_{df.iloc[0]["run"]}.png')

plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['sim'], label='Similiarity')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(base_path + f'/similarity_per_episode_{df.iloc[0]["run"]}.png')