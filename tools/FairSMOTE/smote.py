#!/usr/bin/env python

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import argparse
import os
import sys

def apply_fair_smote(input_csv, output_csv, label_column, protected_attributes):
    """Apply SMOTE within each protected group to the dataset."""
    try:
        # Load the dataset
        print("Loading dataset...")
        df = pd.read_csv(input_csv)
        print(f"Dataset loaded. Shape: {df.shape}")

        # Verify columns exist
        required_cols = protected_attributes + [label_column]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")

        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])

        # Separate features and label
        X = df.drop(columns=[label_column])
        y = df[label_column]

        # Combine features and label for stratification
        df_combined = X.copy()
        df_combined[label_column] = y
        df_combined['protected_group'] = df_combined[protected_attributes].astype(str).agg('-'.join, axis=1)

        # Perform SMOTE within each protected group
        df_resampled = pd.DataFrame(columns=df_combined.columns)
        protected_groups = df_combined['protected_group'].unique()
        print(f"Protected groups: {protected_groups}")

        for group in protected_groups:
            print(f"\nProcessing group: {group}")
            df_group = df_combined[df_combined['protected_group'] == group]
            X_group = df_group.drop(columns=[label_column, 'protected_group'])
            y_group = df_group[label_column]

            if len(y_group.unique()) < 2:
                print(f"Not enough classes to resample for group {group}. Skipping SMOTE.")
                df_resampled = pd.concat([df_resampled, df_group], axis=0)
                continue

            sm = SMOTE(random_state=42)
            X_resampled, y_resampled = sm.fit_resample(X_group, y_group)

            df_group_resampled = pd.DataFrame(X_resampled, columns=X_group.columns)
            df_group_resampled[label_column] = y_resampled
            df_group_resampled['protected_group'] = group

            df_resampled = pd.concat([df_resampled, df_group_resampled], axis=0)

        # Drop the 'protected_group' column
        df_resampled = df_resampled.drop(columns=['protected_group'])

        # Save the resampled dataset
        df_resampled.to_csv(output_csv, index=False)
        print(f"\nResampled dataset saved to {output_csv}")
        print("Transformation complete!")
        return df_resampled

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply SMOTE within each protected group to dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to input dataset CSV.')
    parser.add_argument('--output', type=str, required=True, help='Path to output transformed CSV.')
    parser.add_argument('--label_column', type=str, required=True, help='Name of the label column.')
    parser.add_argument('--protected_attributes', type=str, required=True, help='Comma-separated list of protected attribute column names.')

    args = parser.parse_args()

    # Parse protected attributes into a list
    protected_attributes = [attr.strip() for attr in args.protected_attributes.split(',')]

    apply_fair_smote(args.dataset, args.output, args.label_column, protected_attributes)

