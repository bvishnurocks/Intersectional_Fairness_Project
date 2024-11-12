import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
import argparse

def apply_fair_smote(input_csv, output_csv):
    """Apply Fair SMOTE to the processed dataset"""
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(input_csv)
    
    # Verify columns exist
    required_cols = ['sex', 'race', 'income']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset")
    
    # Verify data is numeric
    non_numeric = df.select_dtypes(include=['object']).columns
    if len(non_numeric) > 0:
        raise ValueError(f"Non-numeric columns found: {list(non_numeric)}")
    
    print("Converting to BinaryLabelDataset format...")
    dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df,
        label_names=['income'],
        protected_attribute_names=['sex', 'race'],
        privileged_protected_attributes=[[1], [1]]
    )
    
    print("Applying Reweighing transformation...")
    RW = Reweighing(
        unprivileged_groups=[{'sex': 0, 'race': 0}],
        privileged_groups=[{'sex': 1, 'race': 1}]
    )
    
    dataset_transformed = RW.fit_transform(dataset)
    
    print("Converting back to DataFrame...")
    df_transformed = dataset_transformed.convert_to_dataframe()[0]
    
    print(f"Saving transformed dataset to {output_csv}")
    df_transformed.to_csv(output_csv, index=False)
    
    print("Transformation complete!")
    return df_transformed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply Fair SMOTE algorithm to dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to input dataset CSV.')
    parser.add_argument('--output', type=str, required=True, help='Path to output transformed CSV.')
    
    args = parser.parse_args()
    
    try:
        apply_fair_smote(args.dataset, args.output)
    except Exception as e:
        print(f"Error: {str(e)}")
