# SiaScoreNet

### Ensemble Siamese Model for Predicting Binding Between HLA Class I Molecules and Peptides

## Overview

***SiaScoreNet*** is a deep learning model that predicts the binding affinity between HLA class I proteins and peptides. It integrates sequence embeddings and ensemble scores from existing predictors, using a Siamese-like architecture.
<img width="539" alt="fig2" src="https://github.com/user-attachments/assets/f0e72473-9808-4e95-abf2-ca6975240fe5" />

---

## How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your test file at Data/test.csv with the following columns:
* peptide
* HLA


### 3. Run Prediction
```bash
python predict.py
```
This will generate:
* predicted_proba: Predicted binding probability
* predicted_label: Binary prediction (0/1)
Saved in predictions.csv
