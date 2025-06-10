# SiaScoreNet

### A Siamese neural network-based model integrating prediction scores for HLA-peptide interaction prediction

## Overview

***SiaScoreNet*** is a deep learning model that predicts the binding affinity between HLA class I proteins and peptides. It integrates sequence embeddings and ensemble scores from existing predictors, using a Siamese-like architecture.


<img width="527" alt="fig2" src="https://github.com/user-attachments/assets/7e12b798-c743-4a8f-88e3-4b39f90d2583" />



## Data availability
All datasets used for training, validation, and testing of the SiaScoreNet model are publicly available on **Zenodo Repository:** [https://doi.org/10.5281/zenodo.15606313](https://doi.org/10.5281/zenodo.15606313)


---

## How to Run
Follow these steps to perform binding prediction using SiaScoreNet:
### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Input File
Create a CSV file named test_example.csv with two columns:
* peptide
* HLA

Example:

peptide, HLA

LLFGYPVYV, HLA-A*02:01

NLVPMVATV, HLA-B*07:02

### 3. Run Feature Extraction
Extract embeddings and ensemble scores by running:

```bash
python feature_extraction.py --input test_example.csv --output features_extracted.csv
```
This will generate the file features_extracted.csv, which contains:
* 9 IEDB scores for ensemble
* ESM embeddings of the peptide
* ESM embeddings of the HLA sequence

### 4. Run Prediction
Use the extracted features to make predictions:

```bash
python predict.py --input features_extracted.csv --output predictions.csv
```
The result will be saved in predictions.csv, containing:

peptide,HLA,predicted_score,predicted_label

Where:
* predicted_score is the probability predicted by the model.
* predicted_label is the binary class (0 or 1) based on threshold 0.5.

