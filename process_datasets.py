# process_datasets.py

import pandas as pd
import os

# Define the datasets configuration
datasets = {
    'adult': {
        'input_path': 'datasets/adult_processed.csv',         # Updated input path
        'output_path': 'datasets/adult_final.csv',            # New output path
        'label_column': 'income',
        'label_mapping': {},  # Assuming already numeric
        'protected_attributes': ['sex', 'race'],
        'attribute_mappings': {
            'sex': lambda x: 1 if str(x).strip().lower() in ['male', 'm'] else 0,
            'race': lambda x: 1 if str(x).strip().lower() == 'white' else 0
        },
        'categorical_columns': ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    },
    'compas': {
        'input_path': 'datasets/compas_processed.csv',        # Updated input path
        'output_path': 'datasets/compas_final.csv',           # New output path
        'label_column': 'two_year_recid',
        'label_mapping': {},  # Assuming already numeric
        'protected_attributes': ['sex', 'race'],
        'attribute_mappings': {
            'sex': lambda x: 1 if str(x).strip().lower() in ['male', 'm'] else 0,
            'race': lambda x: 1 if str(x).strip().lower() == 'white' else 0
        },
        'categorical_columns': [
            'age_cat_age_cat_Greater than 45', 'age_cat_age_cat_Less than 25',
            'c_charge_degree_c_charge_degree_M',
            'c_charge_desc_c_charge_desc_Accessory After the Fact',
            'c_charge_desc_c_charge_desc_Agg Abuse Elderlly/Disabled Adult',
            'c_charge_desc_c_charge_desc_Agg Assault Law Enforc Officer',
            # Add other relevant categorical columns as needed
        ]
    },
    'german': {
        'input_path': 'datasets/german_processed_encoded.csv',  # Use the encoded file
        'output_path': 'datasets/german_final.csv',             # New output path
        'label_column': 'Risk',                                 # Correct label column with capital 'R'
        'label_mapping': {},                                    # Assuming already numeric
        'protected_attributes': ['Personal_status_and_sex_Marital_Status_Male'],  # Adjust based on your dataset
        'attribute_mappings': {
            'Personal_status_and_sex_Marital_Status_Male': lambda x: 1 if x == 1 else 0
        },
        'categorical_columns': []  # Already encoded
    }
}

def process_dataset(name, config):
    print(f"\n[{name}] Processing dataset...")

    input_path = config['input_path']
    output_path = config['output_path']
    label_column = config['label_column']
    label_mapping = config['label_mapping']
    protected_attributes = config['protected_attributes']
    attribute_mappings = config['attribute_mappings']
    categorical_columns = config['categorical_columns']

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' does not exist. Skipping dataset '{name}'.")
        return

    # Load the dataset
    try:
        df = pd.read_csv(input_path)
        print(f"[{name}] Loaded dataset with columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"[{name}] Error loading '{input_path}': {e}")
        return

    # Map protected attributes
    for attr in protected_attributes:
        if attr not in df.columns:
            print(f"[{name}] Warning: Protected attribute '{attr}' not found in dataset. Skipping mapping for this attribute.")
            continue
        try:
            df[attr] = df[attr].apply(attribute_mappings[attr])
            print(f"[{name}] Encoded '{attr}' column.")
        except Exception as e:
            print(f"[{name}] Error encoding '{attr}': {e}")

    # Handle label mapping if any
    if label_mapping:
        if label_column not in df.columns:
            print(f"[{name}] Warning: Label column '{label_column}' not found in dataset.")
        else:
            try:
                df[label_column] = df[label_column].map(label_mapping)
                print(f"[{name}] Mapped label column '{label_column}'.")
            except Exception as e:
                print(f"[{name}] Error mapping label column '{label_column}': {e}")

    # Ensure all columns are numeric
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        print(f"[{name}] Warning: The following columns are non-numeric and will be converted to numeric if possible: {non_numeric_cols}")
        for col in non_numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    print(f"[{name}] Error: Column '{col}' contains non-numeric values that could not be converted.")
                else:
                    print(f"[{name}] Successfully converted column '{col}' to numeric.")
            except Exception as e:
                print(f"[{name}] Error converting column '{col}' to numeric: {e}")

    # Verify all columns are numeric
    remaining_non_numeric = df.select_dtypes(exclude=['number']).columns.tolist()
    if remaining_non_numeric:
        print(f"[{name}] Error: The following columns are still non-numeric after conversion: {remaining_non_numeric}")
    else:
        print(f"[{name}] All columns are numeric.")

    # Save the processed dataset
    try:
        df.to_csv(output_path, index=False)
        print(f"[{name}] Processed dataset saved to '{output_path}'.")
    except Exception as e:
        print(f"[{name}] Error saving processed dataset to '{output_path}': {e}")

def main():
    for name, config in datasets.items():
        process_dataset(name, config)
    print("\nAll datasets have been processed.")

if __name__ == "__main__":
    main()

