import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def load_ohio_data(filepath='data/ohio/2018/train/559-ws-training.csv', seq_len=20, pred_step=1):
    df = pd.read_csv(filepath)
    
    # Nur die gewünschten Spalten
    df = df[['glucose_level', 'bolus_dose', 'meal_carbs']]

    # Fehlende Werte auffüllen
    df = df.fillna(0)

    # Normalisieren
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(data_scaled) - seq_len - pred_step):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len+pred_step-1, 0])  # glucose_level vorhersagen

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    return X, y, scaler

if __name__ == "__main__":
    X, y, _ = load_ohio_data()
    print(X.shape, y.shape)
