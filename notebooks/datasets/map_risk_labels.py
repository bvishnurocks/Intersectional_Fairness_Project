# map_risk_labels.py

import pandas as pd
import sys

def map_risk_labels(input_path, output_path):
    try:
        # Load the dataset
        df = pd.read_csv(input_path)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # Verify current labels
        print("Current Risk labels:", df['Risk'].unique())
        
        # Define the mapping
        label_mapping = {1: 0, 2: 1}
        
        # Apply the mapping
        df['Risk'] = df['Risk'].map(label_mapping)
        
        # Verify the mapping
        print("Mapped Risk labels:", df['Risk'].unique())
        
        # Check for any unmapped values
        if df['Risk'].isnull().any():
            print("Warning: Some Risk values could not be mapped and are set to NaN.")
            df = df.dropna(subset=['Risk'])
            print("Dropped rows with NaN Risk values. New dataset size:", df.shape)
        
        # Convert Risk to integer type
        df['Risk'] = df['Risk'].astype(int)
        
        # Save the updated dataset
        df.to_csv(output_path, index=False)
        print(f"Mapped dataset saved to '{output_path}'.")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    input_csv = 'datasets/german_processed.csv'
    output_csv = 'datasets/german_processed_mapped.csv'
    map_risk_labels(input_csv, output_csv)

