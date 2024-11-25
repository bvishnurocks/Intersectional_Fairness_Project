#!/usr/bin/env python

import argparse
import pandas as pd
import sys
import numpy as np
from aif360.algorithms.preprocessing import LFR
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import StandardScaler

def apply_lfr(df, label, protected_attribute, output_path):
    # The protected attribute is already encoded in the main function
    # Define unprivileged and privileged groups
    # Assuming 0 is unprivileged and 1 is privileged
    unprivileged_groups = [{protected_attribute: 0}]
    privileged_groups = [{protected_attribute: 1}]

    # Convert to BinaryLabelDataset
    try:
        dataset = BinaryLabelDataset(
            df=df,
            label_names=[label],
            protected_attribute_names=[protected_attribute]
        )
    except Exception as e:
        print(f"Error converting to BinaryLabelDataset: {e}")
        sys.exit(1)

    # Apply LFR
    try:
        lfr = LFR(unprivileged_groups=unprivileged_groups,
                  privileged_groups=privileged_groups,
                  verbose=1)
        dataset_transf = lfr.fit_transform(dataset)
    except Exception as e:
        print(f"Error applying LFR on '{protected_attribute}': {e}")
        sys.exit(1)

    # Convert back to DataFrame
    df_transf = dataset_transf.convert_to_dataframe()[0]

    # Save the transformed dataset
    try:
        output_file = f"{output_path.rstrip('.csv')}_{protected_attribute}.csv"
        df_transf.to_csv(output_file, index=False)
        print(f"Transformed dataset saved to {output_file}")
    except Exception as e:
        print(f"Error saving transformed dataset: {e}")
        sys.exit(1)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Apply FairMask using LFR.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the input dataset CSV file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the transformed dataset CSV file prefix.')
    parser.add_argument('--label_column', type=str, required=True, help='Name of the label column.')
    parser.add_argument('--protected_attributes', type=str, required=True, help='Comma-separated protected attributes.')
    args = parser.parse_args()

    # Load dataset with semicolon delimiter
    try:
        df = pd.read_csv(args.dataset, delimiter=';', encoding='utf-8', on_bad_lines='error')
        print("Dataset loaded successfully with semicolon delimiter.")
    except FileNotFoundError:
        print(f"Error: Dataset file '{args.dataset}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print(f"Dataset shape before processing: {df.shape}")

    # Handle duplicate columns if any
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        print("Duplicate columns found. Renaming them to make them unique.")
        df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)

    # Rename specific duplicate columns (if needed)
    df = df.rename(columns={
        'decile_score': 'decile_score_risk',
        'decile_score.1': 'decile_score_violence',
        'priors_count': 'priors_count_risk',
        'priors_count.1': 'priors_count_violence'
    })

    # Define label and protected attributes
    label = args.label_column
    protected_attributes = args.protected_attributes.split(',')

    # Check if label and protected attributes exist in the dataset
    missing_columns = [col for col in [label] + protected_attributes if col not in df.columns]
    if missing_columns:
        print(f"Error: The following columns are missing in the dataset: {missing_columns}")
        sys.exit(1)

    # Check for missing values
    print(f"Dataset contains {df.isna().sum().sum()} missing values.")

    # Handle missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Option A: Drop columns with too many missing values
    missing_values = df.isnull().sum()
    threshold = len(df) * 0.5  # Drop columns with more than 50% missing values
    cols_to_drop = missing_values[missing_values > threshold].index.tolist()

    if cols_to_drop:
        print(f"\nDropping columns with more than {threshold} missing values:")
        print(cols_to_drop)
        df.drop(columns=cols_to_drop, inplace=True)
    else:
        print("\nNo columns to drop based on missing value threshold.")

    # Option B: Impute missing values
    # Impute numerical columns with mean
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    # Impute categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Verify that there are no missing values left
    print(f"\nDataset after handling missing values contains {df.isna().sum().sum()} missing values.")

    if df.isna().sum().sum() > 0:
        print("Error: There are still missing values after handling them.")
        sys.exit(1)

    # Encode all categorical columns, including label and protected attributes
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        print(f"Encoding categorical columns: {categorical_cols}")
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes

    # Ensure the label is binary and encoded as 0 and 1
    if df[label].dtype not in [np.int64, np.float64]:
        print(f"Encoding label column '{label}'")
        df[label] = df[label].astype('category').cat.codes

    # Ensure the protected attributes are properly encoded
    for attr in protected_attributes:
        if df[attr].dtype not in [np.int64, np.float64]:
            print(f"Encoding protected attribute '{attr}'")
            df[attr] = df[attr].astype('category').cat.codes

    # Ensure all features are numerical
    features = df.drop(columns=[label] + protected_attributes)

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # Ensure df features are float64 to match scaled features
    for col in features.columns:
        df[col] = df[col].astype('float64')

    df.update(features_scaled)

    # Apply LFR separately for each protected attribute
    for attr in protected_attributes:
        print(f"\nApplying LFR for protected attribute: {attr}")
        apply_lfr(df.copy(), label, attr, args.output)

if __name__ == "__main__":
    main()

