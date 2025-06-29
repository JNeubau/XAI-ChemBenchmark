import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['fast'])

# Data for the first plot (from the image you shared)
features1 = [
    'maccsfingerprint25', 'maccsfingerprint41', 'maccsfingerprint80', 
    'maccsfingerprint140', 'maccsfingerprint62', 'maccsfingerprint150', 
    'maccsfingerprint148', 'maccsfingerprint78', 'maccsfingerprint53', 
    'maccsfingerprint119'
]

importances1 = [
    0.085, 0.083, 0.075, 0.056, 0.043, 
    0.033, 0.028, 0.027, 0.025, 0.021
]

# Data for the second plot (different features with similar distribution)
features2 = [
    'maccsfingerprint25', 'maccsfingerprint41', 'maccsfingerprint80', 
    'maccsfingerprint140', 'maccsfingerprint152','maccsfingerprint50', 
    'maccsfingerprint150', 'maccsfingerprint157', 'maccsfingerprint78', 
    'maccsfingerprint79'
]

importances2 = [
    0.115, 0.107, 0.080, 0.062, 0.060, 
    0.045, 0.044, 0.044, 0.032, 0.030
]

# Create a unified set of features
all_features = list(set(features1 + features2))

# Create dictionaries mapping features to importances
importance_dict1 = {feature: importance for feature, importance in zip(features1, importances1)}
importance_dict2 = {feature: importance for feature, importance in zip(features2, importances2)}

# Calculate maximum importance (max of both importances) for sorting
max_importance = {}
for feature in all_features:
    imp1 = importance_dict1.get(feature, 0)
    imp2 = importance_dict2.get(feature, 0)
    max_importance[feature] = max(imp1, imp2)

# Sort features by maximum importance (descending)
sorted_features = sorted(all_features, key=lambda x: max_importance[x], reverse=True)

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))
barWidth = 0.35
positions = np.arange(len(sorted_features))

# Create empty lists to store values for plotting
values1 = []
values2 = []

# Fill in the values, using 0 for missing features
for feature in sorted_features:
    values1.append(importance_dict1.get(feature, 0))
    values2.append(importance_dict2.get(feature, 0))

# Create bars
bars1 = ax.bar(positions - barWidth/2, [v if v > 0 else np.nan for v in values1], 
               width=barWidth, label='Python 3.7')
bars2 = ax.bar(positions + barWidth/2, [v if v > 0 else np.nan for v in values2], 
               width=barWidth, label='Python 3.10')

# Add importance values as text labels on each bar
# for i, bar in enumerate(bars1):
#     if values1[i] > 0:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
#                 f'{values1[i]:.3f}',
#                 ha='center', va='bottom', rotation=90, fontsize=8)

# for i, bar in enumerate(bars2):
#     if values2[i] > 0:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
#                 f'{values2[i]:.3f}',
#                 ha='center', va='bottom', rotation=90, fontsize=8)

# Add labels and styling
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Importance', fontsize=12)
ax.set_title('Combined Feature Importances Comparison', fontsize=16)
ax.set_xticks(positions)
ax.set_xticklabels(all_features, rotation=90)

# Adjust y-limit
ax.set_ylim(0, max(max(values1), max(values2)) * 1.15)

# Add a legend
ax.legend(fontsize=12)

# Add grid lines for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('combined_feature_importance.png', dpi=300)
plt.show()

# # Create a combined plot showing both datasets side by side for each feature position
# fig, ax = plt.subplots(figsize=(12, 6))

# # Set width of bars
# barWidth = 0.35
# positions1 = np.arange(len(importances1))
# positions2 = [p + barWidth for p in positions1]

# # Create bars
# ax.bar(positions1, importances1, width=barWidth, label='Pyhon 3.7')
# ax.bar(positions2, importances2, width=barWidth, label='Python 3.10')

# # Add labels
# ax.set_xlabel('Feature Rank')
# ax.set_ylabel('Importance')
# ax.set_title('Comparison of Top 10 Feature Importances Between Datasets')
# ax.set_xticks([p + barWidth/2 for p in positions1])
# ax.set_xticklabels([f'#{i+1}' for i in range(len(positions1))])

# # Add a legend
# ax.legend()

# plt.tight_layout()
# plt.savefig('feature_importance_side_by_side.png', dpi=300)
# plt.show()