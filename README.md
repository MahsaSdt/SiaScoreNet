# SiaScoreNet

SiaScoreNet is a machine learning model that integrates prediction scores from state-of-the-art predictors with Siamese networks to provide superior performance in predicting HLA-peptide binding interactions. SiaScoreNet combines HLA embeddings, peptide embeddings, and prediction scores from multiple sources to provide a comprehensive solution. This tool is particularly useful for advancing research in immunotherapy, vaccine design, and precision medicine.


<img width="539" alt="fig2" src="https://github.com/user-attachments/assets/01c53b61-b387-4003-9629-fba87c9e1c02" />



## Features
- **Robust Prediction:** Combines the strengths of multiple predictors for enhanced accuracy.
- **Deep Learning Backbone:** Employs a Siamese network architecture for efficient similarity learning.
- **Versatile Input Support:** Handles a variety of HLA subtypes and peptides.


## Files

- `SiaScoreNet.ipynb`: Main script for loading test data, generating predictions, and calculating evaluation metrics.
- `SiaScoreNet.pkl`: Pre-trained SiaScoreNet model file.
- `ESM_test.csv`: Test dataset with peptide-HLA pairs with feature vector consist of IEDB scores and ESM embeddimgs.
- `label.csv`: Test dataset labels.
