import argparse
import pandas as pd
import numpy as np
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
import sys
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply Reweighing to mitigate bias in the dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the input dataset.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the REW-processed dataset.')
    parser.add_argument('--label_column', type=str, required=True, help='The target variable column name.')
    parser.add_argument('--protected_attributes', type=str, required=True, help='Comma-separated list of protected attribute column names.')
    return parser.parse_args()

def preprocess_categorical(df, categorical_cols):
    """
    Properly handle categorical variables by converting them to numeric format.
    """
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    return df

def apply_reweighting(input_path, output_path, label_col, protected_attrs):
    try:
        # Print current working directory for debugging
        print(f"Current working directory: {os.getcwd()}")

        # Load the dataset
        df = pd.read_csv(input_path)
        print(f"Loaded dataset with {df.shape[0]} samples and {df.shape[1]} features.")
        print("Available columns:", df.columns.tolist())

        # Define protected attributes
        original_protected_attributes = [attr.strip() for attr in protected_attrs.split(',')]
        
        # Verify protected attributes exist in the dataset
        missing_attrs = [attr for attr in original_protected_attributes if attr not in df.columns]
        if missing_attrs:
            raise ValueError(f"Protected attributes {missing_attrs} not found in dataset.")

        # Handle 'Personal_status_and_sex' by extracting gender
        if 'Personal_status_and_sex' in original_protected_attributes:
            print("Mapping 'Personal_status_and_sex' to binary 'sex' attribute.")
            df['sex'] = df['Personal_status_and_sex'].map({
                'A91': 1,  # male
                'A93': 1,  # male
                'A94': 1,  # male
                'A92': 0,  # female
                'A95': 0   # female
            })
            # Remove 'Personal_status_and_sex' and use 'sex' as the protected attribute
            protected_attributes = ['sex']
        else:
            protected_attributes = original_protected_attributes

        # Ensure no missing values in 'sex'
        if 'sex' in df.columns:
            missing_sex = df['sex'].isnull().sum()
            if missing_sex > 0:
                print(f"Dropping {missing_sex} rows with missing 'sex' values.")
                df = df.dropna(subset=['sex'])

        # Identify categorical columns (excluding protected attributes and label)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in protected_attributes + [label_col]]

        # Handle categorical variables
        if categorical_cols:
            print(f"Converting categorical features: {categorical_cols}")
            df = preprocess_categorical(df, categorical_cols)

        # Convert protected attributes to binary if needed
        for attr in protected_attributes:
            if df[attr].dtype != 'int64' and df[attr].dtype != 'float64':
                df[attr] = pd.to_numeric(df[attr], errors='coerce')
            unique_values = df[attr].dropna().unique()
            if len(unique_values) > 2:
                print(f"Warning: Protected attribute '{attr}' has more than two unique values.")
                df[attr] = (df[attr] == df[attr].max()).astype(int)
            elif not set(unique_values).issubset({0, 1}):
                df[attr] = (df[attr] == df[attr].max()).astype(int)

        # Ensure label is binary
        if not set(df[label_col].unique()).issubset({0, 1}):
            print(f"Converting label column '{label_col}' to binary format.")
            df[label_col] = (df[label_col] == df[label_col].max()).astype(int)

        # Drop any rows with NaN values
        df.dropna(inplace=True)
        print(f"Final dataset shape after dropping NaNs: {df.shape}")

        # Create BinaryLabelDataset
        dataset = BinaryLabelDataset(
            df=df,
            label_names=[label_col],
            protected_attribute_names=protected_attributes,
            favorable_label=1,
            unfavorable_label=0
        )

        # Define privileged and unprivileged groups
        privileged_groups = [{attr: 1 for attr in protected_attributes}]
        unprivileged_groups = [{attr: 0 for attr in protected_attributes}]

        # Initialize and apply Reweighing
        RW = Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )

        # Fit and transform the dataset
        dataset_transf = RW.fit_transform(dataset)
        print("Reweighting completed successfully.")

        # Extract weights and add to original dataframe
        df['sample_weights'] = dataset_transf.instance_weights

        # Save the processed dataset
        df.to_csv(output_path, index=False)
        print(f"Reweighted dataset saved to '{output_path}'.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_arguments()
    apply_reweighting(args.dataset, args.output, args.label_column, args.protected_attributes)

