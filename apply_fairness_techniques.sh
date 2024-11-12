#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Fairness Techniques Application..."

# FairSMOTE on Adult Dataset
echo "Applying FairSMOTE to Adult Dataset..."
cd tools/FairSMOTE/
python fair_smote.py \
    --dataset ../../datasets/adult_processed.csv \
    --output ../../results/fair_smote_adult.csv \
    --label_column income \
    --protected_attributes sex,race
echo "FairSMOTE applied successfully."
cd ../../

# FairMask on COMPAS Dataset
echo "Applying FairMask to COMPAS Dataset..."
cd tools/FairMask/
python fair_mask.py \
    --dataset ../../datasets/compas_processed.csv \
    --output ../../results/fair_mask_compas.csv
echo "FairMask applied successfully."
cd ../../

# Reweighting (REW) on German Credit Dataset
echo "Applying Reweighting (REW) to German Credit Dataset..."
cd tools/REW/
python reweighting.py \
    --dataset ../../datasets/german_processed_mapped.csv \
    --output ../../results/rew_german_credit.csv \
    --label_column Risk \
    --protected_attributes Personal_status_and_sex_Marital_Status_Male
echo "Reweighting applied successfully."
cd ../../

echo "All Fairness Techniques Applied Successfully."

