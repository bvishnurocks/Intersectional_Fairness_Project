# fair_mask.py

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
import argparse

def apply_fair_mask(input_csv, output_csv):
    # Load dataset
    df = pd.read_csv(input_csv)
    
    # Convert to AIF360's BinaryLabelDataset
    dataset = BinaryLabelDataset(
        df=df,
        label_names=['Two_yr_Recidivism'],  # Ensure this matches your dataset
        protected_attribute_names=['sex', 'race'],
        privileged_protected_attributes=[[1, 1]]  # Assuming 'sex'=1 (Male), 'race'=1 (White) are privileged
    )
    
    # Apply Disparate Impact Remover
    DIR = DisparateImpactRemover(repair_level=1.0)
    dataset_transf = DIR.fit_transform(dataset)
    
    # Convert back to DataFrame
    df_transf = dataset_transf.convert_to_dataframe()[0]
    
    # Save the transformed dataset
    df_transf.to_csv(output_csv, index=False)
    
    print(f"FairMask applied. Transformed dataset saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply FairMask Disparate Impact Remover.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to input dataset CSV.')
    parser.add_argument('--output', type=str, required=True, help='Path to output transformed CSV.')
    
    args = parser.parse_args()
    
    apply_fair_mask(args.dataset, args.output)

