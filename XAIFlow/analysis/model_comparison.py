import pandas as pd
import os
from datetime import datetime
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_explanation_values(df):
    """Normalize explanation values for each model to make them comparable."""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # For each model, normalize the explanation values
    for model in df['Model'].unique():
        model_mask = df['Model'] == model
        values = df.loc[model_mask, 'Explanation_value']
        
        abs_values = abs(values)
        sum_value = abs_values.sum()
        
        df.loc[model_mask, 'Explanation_value'] = values / sum_value
        # print(f"Values after normalization: {df.loc[model_mask, 'Explanation_value'].describe()}")
        # print(f"Values after normalization: {df.loc[model_mask, 'Explanation_value']}")

    
    return df

def process_feature_key(feature_key):
    """Extract feature name from string or tuple representations like "('xxx',)"."""
    if isinstance(feature_key, str) and feature_key.startswith("('") and feature_key.endswith("',)"):
        # Extract the string inside the tuple representation
        return feature_key[2:-3]
    elif isinstance(feature_key, tuple):
        # Join tuple elements with 'x' for interactions, or return single element
        return 'x'.join(str(f) for f in feature_key)
    else:
        # Return as is for other cases
        return str(feature_key)

def load_model_results(results_dir):
    """Load all Excel files from results directory."""
    all_data = []
    files = list(Path(results_dir).glob('**/*molecule_results_with_highlights*.xlsx'))
    
    if not files:
        raise ValueError(f"No Excel files found in directory: {results_dir}\nPlease check if the path is correct and contains files matching pattern '*molecule_results_with_highlights*.xlsx'")
    
    for file in files:
        print(f"Loading file: {file}")
        df = pd.read_excel(file)
        if not df.empty:
            all_data.append(df)
        else:
            print(f"Warning: Empty file found: {file}")
    
    if not all_data:
        raise ValueError("No valid data found in any of the Excel files")
    
    print(f"Data:\n{all_data}\n")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Process feature keys for SHAPIQ model
    combined_data['Feature_key'] = combined_data['Feature_key'].apply(process_feature_key)
    
    # Normalize the explanation values
    normalize_combined_data = normalize_explanation_values(combined_data)
    
    return combined_data,normalize_combined_data

def calculate_model_rankings(df):
    """Calculate various rankings and metrics for each model."""
    # Create empty lists to store data
    data = []
    
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        print(f"Processing model: {model}, Number of rows: {len(model_data)}")
        model_data = model_data[model_data['Explanation_value'].notna()]

        for feature_key in model_data['Feature_key'].unique():
            feature_data = model_data[model_data['Feature_key'] == feature_key]
            
            # Create a row for each feature
            # Extract numeric value from 'Explanation_sign' (e.g., "Positive|0.4010989010989011")
            # def parse_corr_value(val):
            #     if isinstance(val, str) and '|' in val:
            #         try:
            #             return float(val.split('|')[1])
            #         except Exception:
            #             return np.nan
            #     return np.nan

            # corr_values = feature_data['Explanation_sign'].apply(parse_corr_value)
            row = {
                'Model': model,
                'Feature': feature_key,
                'Average_Explanation_Value': abs(feature_data['Explanation_value']).mean(),
                'Std_Dev': feature_data['Explanation_value'].std(),
                'Min_Value': feature_data['Explanation_value'].min(),
                'Max_Value': feature_data['Explanation_value'].max(),
                'Fold_Count': feature_data['Fold_No'].nunique(),
                # 'Corr_value': corr_values.mean(),
            }
            data.append(row)
    
    # Create DataFrame from the collected data
    rankings_df = pd.DataFrame(data)
    # Set Model and Feature as index for better organization
    rankings_df = rankings_df.set_index(['Model', 'Feature'])
    
    return rankings_df

def compare_feature_importance(df):
    """Compare feature importance across models."""
    pivot_table = pd.pivot_table(
        df,
        values='Explanation_value',
        index='Feature_key',
        columns='Model',
        aggfunc='mean'
    )
    
    # Calculate correlation between model explanations
    correlation_matrix = pivot_table.corr()
    
    return pivot_table, correlation_matrix

def create_ranking_plots(rankings_df, output_dir, timestamp):
    """Create visualizations for model rankings."""
    # Set style
    plt.style.use('default')  # Use default matplotlib style instead of seaborn
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Common plot settings
    plt.rcParams.update({
        'figure.autolayout': True,
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    # Get top 10 features for each model
    top_features_by_model = {}
    common_features = set()
    first_model = True
    
    for model in rankings_df.index.get_level_values('Model').unique():
        model_data = rankings_df.xs(model)
        top_10 = set(model_data.nlargest(10, 'Average_Explanation_Value').index)
        top_features_by_model[model] = top_10
        if first_model:
            common_features = top_10
            first_model = False
        else:
            common_features = common_features.union(top_10)
    
    # Filter the dataframe to include only common top features
    plot_data = rankings_df.reset_index()
    plot_data = plot_data[plot_data['Feature'].isin(common_features)]
    plot_data = plot_data.sort_values('Average_Explanation_Value', ascending=False)
    
    # 1. Bar plot of top features by average explanation value
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(data=plot_data, x='Feature', y='Average_Explanation_Value', hue='Model', palette='deep')
    plt.xticks(rotation=90, ha='right')  # Rotated labels for better readability
    plt.title('Top Features Across All Models by Average Explanation Value', pad=20)
    plt.xlabel('Feature', labelpad=10)
    plt.ylabel('Average Explanation Value', labelpad=10)
    # Add legend outside of plot to avoid overlap
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'common_top_features_bar_{timestamp}.png'), dpi=300, bbox_inches='tight', 
                pad_inches=0.5)  # Added padding for legend
    plt.close()

    # # 1b. Bar plot: Average Explanation Value with Sign for Every Feature
    # plt.figure(figsize=(15, 8))
    # signed_data = rankings_df.reset_index()
    # # Use Corr_value as the sign (positive/negative correlation)
    # signed_data['Signed_Avg_Explanation_Value'] = signed_data['Average_Explanation_Value'] * np.sign(signed_data['Corr_value'].fillna(1))
    # sns.barplot(
    #     data=signed_data,
    #     x='Feature',
    #     y='Signed_Avg_Explanation_Value',
    #     hue='Model',
    #     palette='deep'
    # )
    # plt.xticks(rotation=90, ha='right')
    # plt.title('Average Explanation Value (with Sign) for Every Feature', pad=20)
    # plt.xlabel('Feature', labelpad=10)
    # plt.ylabel('Signed Average Explanation Value', labelpad=10)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'signed_avg_expl_value_{timestamp}.png'), dpi=300, bbox_inches='tight', pad_inches=0.5)
    # plt.close()

    # 1c. Bump chart: Feature ranking changes across models
    # Prepare data for bump chart
    bump_data = rankings_df.reset_index()
    # Rank features within each model (1 = highest average explanation value)
    bump_data['Rank'] = bump_data.groupby('Model')['Average_Explanation_Value'].rank(ascending=False, method='min')
    
    # Fix for deprecation warning: handle top N features selection differently
    N = 5
    top_features = []
    for model in bump_data['Model'].unique():
        model_data = bump_data[bump_data['Model'] == model].copy()
        top_n = model_data.nsmallest(N, 'Rank')[['Feature']]
        top_features.extend(top_n['Feature'].tolist())
    
    # Get unique top features across all models
    top_features_set = set(top_features)
    bump_data = bump_data[bump_data['Feature'].isin(top_features_set)]

    # Pivot for bump chart: rows=Feature, columns=Model, values=Rank
    bump_pivot = bump_data.pivot(index='Feature', columns='Model', values='Rank')

    # Sort features by their average rank for plotting order
    avg_rank = bump_pivot.mean(axis=1).sort_values()
    bump_pivot = bump_pivot.loc[avg_rank.index]

    # Assign a unique color to each feature
    feature_colors = sns.color_palette('tab20', n_colors=len(bump_pivot.index))
    color_map = dict(zip(bump_pivot.index, feature_colors))

    plt.figure(figsize=(15, 8))
    for feature in bump_pivot.index:
        plt.plot(
            bump_pivot.columns,
            bump_pivot.loc[feature],
            marker='o',
            label=feature,
            linewidth=2,
            color=color_map[feature]
        )
    plt.gca().invert_yaxis()  # Rank 1 at the top
    plt.title('Bump Chart: Feature Ranking Across Models', pad=20)
    plt.xlabel('Model', labelpad=10)
    plt.ylabel('Feature Rank (1 = Most Important)', labelpad=10)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Feature', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'bump_chart_{timestamp}.png'), dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    # 1d. Bump chart: Feature ranking changes across models for 5 worst features
    # Prepare data for bump chart (worst features)
    bump_data_worst = rankings_df.reset_index()
    # Rank features within each model (1 = highest, so worst = largest rank)
    bump_data_worst['Rank'] = bump_data_worst.groupby('Model')['Average_Explanation_Value'].rank(ascending=False, method='min')
    N_worst = 5
    worst_features = []
    for model in bump_data_worst['Model'].unique():
        model_data = bump_data_worst[bump_data_worst['Model'] == model].copy()
        worst_n = model_data.nlargest(N_worst, 'Rank')[['Feature']]
        print(f"Model: {model}, Worst Features: {worst_n['Feature'].tolist()}")
        worst_features.extend(worst_n['Feature'].tolist())
    # Get unique worst features across all models
    worst_features_set = set(worst_features)
    bump_data_worst = bump_data_worst[bump_data_worst['Feature'].isin(worst_features_set)]
    # Pivot for bump chart: rows=Feature, columns=Model, values=Rank
    bump_pivot_worst = bump_data_worst.pivot(index='Feature', columns='Model', values='Rank')
    # Sort features by their average rank for plotting order (worst at top)
    avg_rank_worst = bump_pivot_worst.mean(axis=1).sort_values(ascending=False)
    bump_pivot_worst = bump_pivot_worst.loc[avg_rank_worst.index]

    # Assign a unique color to each feature for the worst bump chart
    feature_colors_worst = sns.color_palette('tab20', n_colors=len(bump_pivot_worst.index))
    color_map_worst = dict(zip(bump_pivot_worst.index, feature_colors_worst))

    plt.figure(figsize=(15, 8))
    for feature in bump_pivot_worst.index:
        plt.plot(
            bump_pivot_worst.columns,
            bump_pivot_worst.loc[feature],
            marker='o',
            label=feature,
            linewidth=2,
            color=color_map_worst[feature]
        )
    plt.gca().invert_yaxis()  # Rank 1 at the top
    plt.title('Bump Chart: Worst Feature Ranking Across Models', pad=20)
    plt.xlabel('Model', labelpad=10)
    plt.ylabel('Feature Rank (1 = Most Important)', labelpad=10)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Feature', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'bump_chart_worst_{timestamp}.png'), dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    
    # 5. Heatmap of correlation between models
    # pivot_table, correlation_matrix = compare_feature_importance(rankings_df.reset_index().rename(columns={'Feature': 'Feature_key'}))
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", cbar_kws={'label': 'Correlation'})
    # plt.title('Correlation Between Models (Feature Importance)', pad=20)
    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'model_correlation_heatmap_{timestamp}.png'), dpi=300, bbox_inches='tight')
    # plt.close()


    # 2. Box plot showing value distribution
    # plt.figure(figsize=(15, 8))
    # plot_data = rankings_df.reset_index()
    # plot_data['Range'] = plot_data['Max_Value'] - plot_data['Min_Value']
    # sns.boxplot(data=plot_data, x='Model', y='Range', hue='Model', legend=False)
    # plt.title('Distribution of Explanation Value Ranges by Model', pad=20)
    # plt.xlabel('Model', labelpad=10)
    # plt.ylabel('Value Range', labelpad=10)
    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'value_distribution_{timestamp}.png'), dpi=300, bbox_inches='tight')
    # plt.close()

    # 3. Heatmap of metrics
    # plt.figure(figsize=(12, 8))
    # metrics = ['Average_Explanation_Value', 'Std_Dev', 'Fold_Count']
    # heatmap_data = rankings_df[metrics].groupby('Model').mean()
    # sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'Value'})
    # plt.title('Model Metrics Heatmap', pad=20)
    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'metrics_heatmap_{timestamp}.png'), dpi=300, bbox_inches='tight')
    # plt.close()

    # 4. Scatter plot of Average vs Std Dev
    # plt.figure(figsize=(10, 8))
    # colors = sns.color_palette('deep', n_colors=len(rankings_df.index.get_level_values('Model').unique()))
    # for idx, model in enumerate(rankings_df.index.get_level_values('Model').unique()):
    #     model_data = rankings_df.xs(model)
    #     plt.scatter(model_data['Average_Explanation_Value'], 
    #                model_data['Std_Dev'], 
    #                label=model, 
    #                alpha=0.6,
    #                c=[colors[idx]])
    # plt.xlabel('Average Explanation Value', labelpad=10)
    # plt.ylabel('Standard Deviation', labelpad=10)
    # plt.title('Average Explanation Value vs Standard Deviation', pad=20)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'avg_vs_std_{timestamp}.png'), dpi=300, bbox_inches='tight')
    # plt.close()

def generate_comparison_report(results_dir, output_dir):
    """Generate comprehensive comparison report."""
    try:
        # Load and process data
        print(f"Searching for files in: {results_dir}")
        combined_data,normalize_combined_data = load_model_results(results_dir)
        # print(f"Found {len(combined_data)} total rows of data")
        # print(f"Models found: {combined_data['Model'].unique()}")
        
        # Calculate rankings
        model_rankings = calculate_model_rankings(normalize_combined_data)
        
        # Compare feature importance
        feature_importance, model_correlation = compare_feature_importance(combined_data)
        norm_feature_importance, norm_model_correlation = compare_feature_importance(normalize_combined_data)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create plots
        create_ranking_plots(model_rankings, output_dir, timestamp)
        
        # Save results
        output_file = os.path.join(output_dir, f'model_comparison_{timestamp}.xlsx')
        with pd.ExcelWriter(output_file) as writer:
            model_rankings.to_excel(writer, sheet_name='Model_Rankings')
            feature_importance.to_excel(writer, sheet_name='Feature_Importance')
            norm_feature_importance.to_excel(writer, sheet_name='Normalized_Feature_Importance')
            model_correlation.to_excel(writer, sheet_name='Model_Correlation')
            
            # Additional analysis: Top features per model
            for model in combined_data['Model'].unique():
                model_data = combined_data[combined_data['Model'] == model]
                norm_model_data = normalize_combined_data[normalize_combined_data['Model'] == model]
                
                # Aggregate by Feature_key
                agg_model_data = model_data.groupby('Feature_key').agg({
                    'SMARTS': 'first',
                    'Explanation_value': 'mean',
                    'Number_where_important': 'first'
                }).reset_index()
                agg_norm_model_data = norm_model_data.groupby('Feature_key').agg({
                    'Explanation_value': 'mean'
                }).reset_index().rename(columns={'Explanation_value': 'Normalized_Explanation_value'})
                
                # Merge normalized values
                agg_features = pd.merge(
                    agg_model_data,
                    agg_norm_model_data,
                    on='Feature_key',
                    how='left'
                )
                
                # Get top 10 features by mean Explanation_value
                top_features = agg_features.nlargest(10, 'Explanation_value')
                top_features.to_excel(writer, sheet_name=f'{model}_TF', index=False)
        
        return output_file
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Define directories
        parent_dir = Path(__file__).parents[2]  # Go up to XAI-experiments directory
        if len(sys.argv) < 2:
            print("Usage: python model_comparison.py <results_dir>")
            exit(1)
        results_dir = Path(sys.argv[1]).resolve()  # Get absolute path
        print(f"Processing directory: {results_dir}")
        if not results_dir.exists():
            print(f"Error: Directory does not exist: {results_dir}")
            exit(1)
        output_dir = parent_dir / 'results' / 'model_comparison'
        
        # Generate report
        output_file = generate_comparison_report(results_dir, output_dir)
        print(f"Comparison report generated: {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
