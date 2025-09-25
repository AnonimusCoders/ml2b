import numpy as np
import pandas as pd

# aggregation function
def agg_stats(x):
    x = np.array(x)
    return pd.Series({
        "median": np.median(x),
        "q1": np.percentile(x, 25),
        "q3": np.percentile(x, 75),
        "iqr": np.percentile(x, 75) - np.percentile(x, 25),
        "top10_frac": np.mean(x >= 0.90),
        "below50_frac": np.mean(x <= 0.50)
    })

def read_xlsx(path):
    df = pd.read_excel(path)
    
    # Get model columns (exclude metadata columns)
    model_cols = [col for col in df.columns if not col.startswith('Unnamed') and 
                  col not in ['№', 'comp_name', 'competition', 'data_type', 'metric_type', 'domain', 'task_type', 'competition_link']]
    
    # Define model groups and their column ranges
    model_groups = {
        'gpt-oss-120': (11, 24),  # columns L-X (11-23)
        'gemini-2.5-flash': (24, 37),  # columns Y-AK (24-36)
        'gpt-4.1-mini': (37, 50),  # columns AL-BW (37-49)
        'qwen2.5-coder': (50, 63),  # columns BX-CK (50-62)
        'gpt-4.1-mini + deepseek-r1 (ml-master)': (63, 76),  # columns CL-DC (63-75)
        'gpt-oss:120b + qwen3-coder:30b (ml-master)': (76, 89)  # columns DD-EP (76-88)
    }
    
    # Create long format dataframe
    result_data = []
    
    for model_name, (start_col, end_col) in model_groups.items():
        # Get language names from first row
        languages = df.iloc[0, start_col:end_col].dropna().tolist()
        
        # Process each language column for this model
        for i, col_idx in enumerate(range(start_col, end_col)):
            if i < len(languages):
                language = languages[i]
                # Get numeric values from the data rows (skip first row which has language names)
                values = pd.to_numeric(df.iloc[1:, col_idx], errors='coerce').dropna()
                
                # Get task types from column J (index 9)
                task_types = df.iloc[1:, 9].dropna().tolist()  # Column J is index 9
                
                for j, value in enumerate(values):
                    if not pd.isna(value) and j < len(task_types):
                        task_type = task_types[j]
                        result_data.append({
                            'language': language,
                            'model': model_name,
                            'task_type': task_type,
                            'percentile': value
                        })
    
    result_df = pd.DataFrame(result_data)
    print("Sample data:")
    print(result_df.head(10))
    print(f"\nTotal records: {len(result_df)}")
    print(f"Languages: {sorted(result_df['language'].unique())}")
    print(f"Models: {sorted(result_df['model'].unique())}")
    print(f"Task types: {sorted(result_df['task_type'].unique())}")
    
    return result_df

def main():
    # Read and process the data
    df = read_xlsx("final_version_percentiles.xlsx")

    # Group by model, task type, and language to calculate statistics
    agg = df.groupby(["model", "task_type", "language"])["percentile"].apply(agg_stats).unstack().reset_index()

    print("\nAggregated statistics by model, task type, and language:")
    print(agg)

    # Create median dataframe (pivot table with models and task types as rows and languages as columns)
    median_df = df.groupby(["model", "task_type", "language"])["percentile"].median().unstack(fill_value=np.nan)
    print("\nMedian values by Model, Task Type, and Language:")
    print(median_df)

    # Create IQR dataframe (pivot table with models and task types as rows and languages as columns)
    iqr_df = df.groupby(["model", "task_type", "language"])["percentile"].apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25)).unstack(fill_value=np.nan)
    print("\nIQR values by Model, Task Type, and Language:")
    print(iqr_df)

    # Create the final dataframe with the required structure: model | task type | Arab | Belarus | ...
    # Reset index to make model and task_type regular columns
    final_median_df = median_df.reset_index()
    print("\nFinal median dataframe with required structure:")
    print(final_median_df)

    # Save to CSV files
    final_median_df.to_csv('results_2/median_by_model_task_type_language.csv', index=False)
    iqr_df.reset_index().to_csv('results_2/iqr_by_model_task_type_language.csv', index=False)
    print("\nResults saved to median_by_model_task_type_language.csv and iqr_by_model_task_type_language.csv")

if __name__ == "__main__":
    main()