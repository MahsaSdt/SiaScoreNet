# SiaScoreNet

### Ensemble Siamese Model for Predicting Binding Between HLA Class I Molecules and Peptides

## Overview

SiaScoreNet is a deep learning model that predicts the binding affinity between HLA class I proteins and peptides. It integrates sequence embeddings and ensemble scores from existing predictors, using a Siamese-like architecture.

---

## File Structure

SiaScoreNet-Project/
├── model.py # Model architecture
├── predict.py # Prediction pipeline
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── weights/
│ └── SiaScoreNet_trained_on_D1_model.h5
├── data/
│ └── test.csv
└── predictions.csv # Output prediction file

yaml
Copy
Edit

---

## How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
If using ESM embeddings:

bash
Copy
Edit
pip install fair-esm torch
2. Prepare Data
Place your test file at data/test.csv with the following columns:

peptide

HLA

HLA_sequence

label (optional)

Ensemble scores (first 9 columns after dropping the above)

Peptide embedding (320-dimensional vector)

HLA embedding (320-dimensional vector)

3. Run Prediction
bash
Copy
Edit
python predict.py
This will generate:

predicted_proba: Predicted binding probability

predicted_label: Binary prediction (0/1)

Saved in predictions.csv

Model Inputs
The model receives:

Peptide embedding (320 dims)

HLA embedding (320 dims)

9 ensemble predictor scores

Citation
