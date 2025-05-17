import pandas as pd
import numpy as np
from model import SiaScoreNet
from tensorflow import keras

df = pd.read_csv('data/features_extracted.csv')


X_etest = df.drop(columns=["peptide", "label", "HLA", "HLA_sequence"])
X_etest_ens = X_etest.iloc[:, :9].values
X_etest_1 = X_etest.iloc[:, 9:329].values
X_etest_2 = X_etest.iloc[:, 329:].values

model = SiaScoreNet()
model.load_weights('weights/SiaScoreNet_trained_on_D1_model.h5')


y_pred_proba = model.predict([X_etest_1, X_etest_2, X_etest_ens])
y_pred = (y_pred_proba > 0.5).astype(int)


df['predicted_proba'] = y_pred_proba
df['predicted_label'] = y_pred
df.to_csv('predictions.csv', index=False)
