import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variable: True = train all patients, False = train only the first CSV
TRAIN_ALL = False  

class GlucoseLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        return self.fc(x[:, -1, :]).squeeze(-1)

def load_ohio_data(filepath, seq_len=288):
    df = pd.read_csv(filepath)
    df = df[['glucose_level', 'bolus_dose', 'meal_carbs', 'exercise_intensity', 'basis_sleep_binary']].fillna(0)
    df['meal_indicator'] = (df['meal_carbs'] > 0).astype(float)
    df['glucose_change'] = df['glucose_level'].diff().fillna(0)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(data_scaled) - seq_len - 1):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len, 0])  # Nur ein Schritt weiter

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y, scaler, df

def _predict_iteratively_local(model, initial_sequence, steps, scaler):
    """Iterative 1-step repeated predictor. Returns array (steps,) in original scale if scaler given."""
    model.eval()
    seq = initial_sequence.clone().detach().to(device)
    preds_scaled = []
    with torch.no_grad():
        for _ in range(steps):
            out = model(seq.unsqueeze(0).to(device))  # (1,) on device
            preds_scaled.append(out.cpu().item())
            new_row = torch.zeros((1, seq.shape[1]), device=seq.device, dtype=seq.dtype)
            new_row[0, 0] = out.squeeze().to(seq.dtype)
            seq = torch.cat([seq[1:], new_row], dim=0)

    used_scaler = scaler
    if used_scaler is not None:
        zeros = np.zeros((len(preds_scaled), seq.shape[1]-1))
        preds_orig = used_scaler.inverse_transform(np.concatenate([np.array(preds_scaled).reshape(-1,1), zeros], axis=1))[:,0]
        return preds_orig
    else:
        return np.array(preds_scaled)

def train_lstm_and_iterative_eval(data_dir='data/ohio/2018/train_cleaned/', seq_len=50, batch_size=16, epochs=20, lr=0.001, target_file=None, max_horizon=12, out_dir="lstm/models_lstm"):
    """
    Train a simple LSTM on target_file (or all files if TRAIN_ALL True).
    After training, run iterative predictions up to max_horizon for each valid start index,
    save CSV (y_true_1..max, pred_1..max) and print metrics for horizons 1,3,12.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    if not csv_files:
        raise RuntimeError(f"No CSVs found in {data_dir}")

    if not TRAIN_ALL:
        csv_files = [csv_files[0]]

    for filename in csv_files:
        if target_file is not None and filename != target_file:
            continue

        filepath = os.path.join(data_dir, filename)
        print(f"\nTraining for: {filename}")

        X, y, scaler, df_raw = load_ohio_data(filepath, seq_len)
        if len(X) == 0:
            print("Not enough data for seq_len, skipping:", filename)
            continue

        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        model = GlucoseLSTM(X.shape[2]).to(device)
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                y_pred = model(xb)
                loss = criterion(y_pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")

        # After training: only save model + scaler (no evaluation here)
        single_dir = os.path.join(out_dir, "single")
        os.makedirs(single_dir, exist_ok=True)
        ckpt_path = os.path.join(single_dir, f"lstm_model_simple_{filename.replace('.csv','')}.pth")
        torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler}, ckpt_path)
        print("Saved model checkpoint (no evaluation):", ckpt_path)

if __name__ == '__main__':
    # minimal CLI for standalone use
    if os.path.exists('data/ohio/2018/train_cleaned/'):
        train_lstm_and_iterative_eval('data/ohio/2018/train_cleaned/', seq_len=50, epochs=1, max_horizon=12)
    else:
        print("Training directory not found.")