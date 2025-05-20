import pandas as pd
import os
from datetime import datetime
import numpy as np
from pathlib import Path
import sys

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
