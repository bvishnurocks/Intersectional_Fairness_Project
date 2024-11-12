import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
import argparse
import sys
import os

def apply_fair_mask(input_csv, output_csv, label_column, protected_attributes):
    try:
        # Load dataset
        print("Loading dataset...")
        df = pd.read_csv(input_csv)
        print(f"Dataset loaded. Shape: {df.shape}")

        # Verify the label column exists
        if label_column not in df.columns:
            print(f"Error: Label column '{label_column}' not found in the dataset.")
            sys.exit(1)

        # Verify protected attributes exist
        for attr in protected_attributes:
            if attr not in df.columns:
                print(f"Error: Protected attribute '{attr}' not found in the dataset.")
                sys.exit(1)

        # Convert categorical columns to numeric codes if necessary
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype.name == 'category':
                print(f"Converting column '{col}' to numeric codes.")
                df[col] = df[col].astype('category').cat.codes

        # Ensure label is binary
        label_values = df[label_column].unique()
        if set(label_values) != {0, 1}:
            print(f"Converting label column '{label_column}' to binary values.")
            df[label_column] = df[label_column].apply(lambda x: 1 if x == df[label_column].max() else 0)

        # Create BinaryLabelDataset
        print("Creating BinaryLabelDataset...")
        dataset = BinaryLabelDataset(
            df=df,
            label_names=[label_column],
            protected_attribute_names=protected_attributes,
            favorable_label=1,
            unfavorable_label=0
        )

        # Apply Disparate Impact Remover
        print("Applying Disparate Impact Remover...")
        DIR = DisparateImpactRemover(repair_level=1.0)
        dataset_transf = DIR.fit_transform(dataset)

        # Convert back to DataFrame
        print("Converting transformed dataset back to DataFrame...")
        df_transf = dataset_transf.convert_to_dataframe()[0]

        # Save the transformed dataset
        print(f"Saving transformed dataset to {output_csv}...")
        df_transf.to_csv(output_csv, index=False)

        print("FairMask applied successfully!")
        print(f"Transformed dataset saved to {output_csv}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply FairMask Disparate Impact Remover.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to input dataset CSV.')
    parser.add_argument('--output', type=str, required=True, help='Path to output transformed CSV.')
    parser.add_argument('--label_column', type=str, required=True, help='Name of the label column.')
    parser.add_argument('--protected_attributes', type=str, required=True, help='Comma-separated list of protected attribute column names.')

    args = parser.parse_args()

    # Parse protected attributes into a list
    protected_attributes = [attr.strip() for attr in args.protected_attributes.split(',')]

    apply_fair_mask(args.dataset, args.output, args.label_column, protected_attributes)

