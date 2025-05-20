import pandas as pd
import os
from datetime import datetime
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    return pd.concat(all_data, ignore_index=True)

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
            row = {
                'Model': model,
                'Feature': feature_key,
                'Average_Explanation_Value': abs(feature_data['Explanation_value']).mean(),
                'Std_Dev': feature_data['Explanation_value'].std(),
                'Min_Value': feature_data['Explanation_value'].min(),
                'Max_Value': feature_data['Explanation_value'].max(),
                'Fold_Count': feature_data['Fold_No'].nunique()
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
        combined_data = load_model_results(results_dir)
        print(f"Found {len(combined_data)} total rows of data")
        print(f"Models found: {combined_data['Model'].unique()}")
        
        # Calculate rankings
        model_rankings = calculate_model_rankings(combined_data)
        
        # Compare feature importance
        feature_importance, model_correlation = compare_feature_importance(combined_data)
        
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
            model_correlation.to_excel(writer, sheet_name='Model_Correlation')
            
            # Additional analysis: Top features per model
            for model in combined_data['Model'].unique():
                model_data = combined_data[combined_data['Model'] == model]
                top_features = model_data.nlargest(10, 'Explanation_value')[
                    ['Feature_key', 'SMARTS', 'Explanation_value', 'Number_where_important']
                ]
                top_features.to_excel(writer, sheet_name=f'{model}_Top_Features')
        
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
