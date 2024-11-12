# Intersectional Fairness Project

A project aimed at mitigating biases in machine learning datasets and models using advanced fairness techniques from the AI Fairness 360 (AIF360) toolkit.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Datasets](#datasets)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [Installing Dependencies](#installing-dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [FairSMOTE on Adult Dataset](#fairsmote-on-adult-dataset)
  - [FairMask on COMPAS Dataset](#fairmask-on-compas-dataset)
  - [Reweighting (REW) on German Credit Dataset](#reweighting-rew-on-german-credit-dataset)
  - [Automating Fairness Techniques](#automating-fairness-techniques)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

The Intersectional Fairness Project implements three key fairness preprocessing methods using the AIF360 toolkit:

- **FairSMOTE**: An extension of SMOTE that considers fairness constraints while generating synthetic samples
- **FairMask**: A fairness-aware preprocessing technique that modifies datasets while preserving utility
- **Reweighting (REW)**: Assigns weights to instances to balance the influence of different groups

## Features

- 🔧 **Preprocessing for Fairness**: Apply advanced fairness algorithms to balance datasets and reduce bias
- 📊 **Support for Multiple Datasets**: Works with Adult, COMPAS, and German Credit datasets
- 🔄 **Automated Workflow**: Streamlined scripts to apply multiple fairness techniques seamlessly
- ✅ **Comprehensive Verification**: Tools to inspect and verify fairness transformations

## Datasets

The project works with the following datasets:

- **Adult Dataset** (`adult_final.csv`)
  - Purpose: Predict income exceeding $50K/year
  - Protected Attributes: sex, race

- **COMPAS Dataset** (`compas_processed.csv`)
  - Purpose: Predict recidivism within two years
  - Protected Attributes: race, gender

- **German Credit Dataset** (`german_processed_mapped.csv`)
  - Purpose: Assess credit risk
  - Protected Attributes: sex, race

Place these datasets in the `datasets/` directory.

## Installation

### Prerequisites

- Operating System: macOS, Linux, or Windows
- Python: Version 3.9
- Conda: Package and environment management system

### Environment Setup

1. Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Create the Conda Environment:
```bash
conda create -n aif360_env python=3.9
```

3. Activate the Environment:
```bash
conda activate aif360_env
```

4. (Optional) Prevent Auto-Activation of Base Environment:
```bash
conda config --set auto_activate_base false
```

### Installing Dependencies

With the `aif360_env` activated:
```bash
pip install --upgrade pip
pip install pandas scikit-learn aif360 imbalanced-learn joblib 'aif360[inFairness]'
```

## Project Structure

```
Intersectional_Fairness_Project/
├── apply_fairness_techniques.sh
├── datasets/
│   ├── adult_final.csv
│   ├── compas_processed.csv
│   └── german_processed_mapped.csv
├── results/
│   ├── fair_mask_compas.csv
│   ├── fair_smote_adult.csv
│   └── rew_german_credit.csv
└── tools/
    ├── FairMask/
    │   └── fair_mask.py
    ├── FairSMOTE/
    │   └── fair_smote.py
    └── REW/
        └── reweighting.py
```

## Usage

### FairSMOTE on Adult Dataset

```bash
cd tools/FairSMOTE/
python fair_smote.py \
    --dataset ../../datasets/adult_final.csv \
    --output ../../results/fair_smote_adult.csv \
    --label_column income \
    --protected_attributes sex,race
```

### FairMask on COMPAS Dataset

```bash
cd ../FairMask/
python fair_mask.py \
    --dataset ../../datasets/compas_processed_clean.csv \
    --output ../../results/fair_mask_compas.csv \
    --label_column two_year_recid \
    --protected_attributes race,gender
```

### Reweighting (REW) on German Credit Dataset

```bash
cd ../REW/
python reweighting.py \
    --dataset ../../datasets/german_processed_mapped.csv \
    --output ../../results/rew_german_credit.csv \
    --label_column Risk \
    --protected_attributes sex,race
```

### Automating Fairness Techniques

Run all techniques using the master script:
```bash
chmod +x apply_fairness_techniques.sh
./apply_fairness_techniques.sh
```

## Verification

Verify outputs using provided Python commands:

```python
# Example: Check Adult Dataset results
import pandas as pd
df = pd.read_csv('results/fair_smote_adult.csv')
print('Income distribution:\n', df['income'].value_counts())
print('Sample Weights Summary:\n', df['weight'].describe())
```

## Troubleshooting

Common issues and solutions:

1. **Environment Activation Issues**
```bash
conda deactivate
conda activate aif360_env
```

2. **Non-Numeric Columns**
```python
import pandas as pd
df = pd.read_csv('path/to/dataset.csv')
print(df.dtypes)
```

3. **Suppress Warnings**
```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```

For more help, visit the [AIF360 Documentation](https://aif360.readthedocs.io/) or open an issue.

## Contributing

1. Fork the Repository
2. Create a Feature Branch: `git checkout -b feature/YourFeature`
3. Commit Changes: `git commit -m "Add Your Feature"`
4. Push to Branch: `git push origin feature/YourFeature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [AI Fairness 360 (AIF360)](https://github.com/Trusted-AI/AIF360)
- [imbalanced-learn](https://imbalanced-learn.org/)
- [PyTorch](https://pytorch.org/)
- [pandas](https://pandas.pydata.org/)
