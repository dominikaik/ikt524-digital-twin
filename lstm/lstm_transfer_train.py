import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def load_ohio_data(filepath, seq_len=20):
    df = pd.read_csv(filepath)
    df = df[['glucose_level', 'bolus_dose', 'meal_carbs']].fillna(0)
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
    return X, y, scaler

def predict_iteratively(model, initial_sequence, steps, scaler):
    model.eval()
    sequence = initial_sequence.clone().detach().to(device)
    predictions = []

    with torch.no_grad():
        for _ in range(steps):
            input_seq = sequence.unsqueeze(0)
            # get prediction tensor (stays on device)
            pred_tensor = model(input_seq)  # shape (1,) on device
            # store float for scaler postprocessing
            pred_value = pred_tensor.cpu().item()
            predictions.append(pred_value)

            # create new_row on the same device and dtype as sequence
            new_row = torch.zeros((1, sequence.shape[1]), device=sequence.device, dtype=sequence.dtype)
            # assign predicted value (ensure scalar tensor)
            new_row[0, 0] = pred_tensor.squeeze()  # put prediction into first feature
            # append and slide window
            sequence = torch.cat([sequence[1:], new_row], dim=0)

    zeros = np.zeros((len(predictions), sequence.shape[1]-1))
    pred_orig = scaler.inverse_transform(np.concatenate([np.array(predictions).reshape(-1,1), zeros], axis=1))[:,0]
    return pred_orig

def train_baseline_and_finetune(data_dir='data/ohio/2018/train_cleaned/', 
                                seq_len=50, batch_size=16, 
                                epochs_baseline=1, epochs_finetune=1, lr=0.001,
                                finetune_file='559-ws-training.csv'):

    os.makedirs("models_lstm", exist_ok=True)
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    if finetune_file not in csv_files:
        print(f"Error: {finetune_file} not found in directory.")
        return

    baseline_files = [f for f in csv_files if f != finetune_file]
    target_file = finetune_file
    print(f"Baseline training files: {baseline_files}")
    print(f"Fine-tune target file: {target_file}")

    # Baseline training
    X_all, y_all = [], []
    for filename in baseline_files:
        print(f"Loading file: {filename}")
        X, y, scaler = load_ohio_data(os.path.join(data_dir, filename), seq_len)
        X_all.append(X)
        y_all.append(y)

    X_all = torch.cat(X_all).to(device)
    y_all = torch.cat(y_all).to(device)
    train_loader = DataLoader(TensorDataset(X_all, y_all), batch_size=batch_size, shuffle=True)

    model = GlucoseLSTM(X_all.shape[2]).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\nTraining baseline model ...")
    for epoch in tqdm(range(epochs_baseline), desc="Baseline Epochs"):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "models_lstm/baseline_model.pth")
    joblib.dump(scaler, "models_lstm/baseline_scaler.joblib")
    print("Baseline model saved.")

    # Fine-tuning
    print(f"\nFine-tuning on {target_file} ...")
    X_target, y_target, scaler_target = load_ohio_data(os.path.join(data_dir, target_file), seq_len)
    train_size = int(0.8 * len(X_target))
    X_train, X_test = X_target[:train_size].to(device), X_target[train_size:].to(device)
    y_train, y_test = y_target[:train_size].to(device), y_target[train_size:].to(device)

    train_loader_target = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    model.load_state_dict(torch.load("models_lstm/baseline_model.pth", map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.5)

    for epoch in tqdm(range(epochs_finetune), desc="Finetune Epochs"):
        model.train()
        for xb, yb in train_loader_target:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        zeros = np.zeros((len(y_test_np), X_target.shape[2]-1))
        y_test_orig = scaler_target.inverse_transform(np.concatenate([y_test_np.reshape(-1,1), zeros], axis=1))[:,0]
        y_pred_orig = scaler_target.inverse_transform(np.concatenate([y_pred.reshape(-1,1), zeros], axis=1))[:,0]

    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    print(f"Fine-tune RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler_target},
               f"models_lstm/lstm_model_finetuned_{target_file.replace('.csv','')}.pth")
    print("Fine-tuned model saved.")

    # Iterative prediction example
    print("\nIterative prediction (10 steps):")
    future_preds = predict_iteratively(model, X_test[-1], steps=10, scaler=scaler_target)
    print("Future predictions:", future_preds)

if __name__ == '__main__':
    data_path = 'data/ohio/2018/train_cleaned/'
    if os.path.exists(data_path):
        train_baseline_and_finetune(data_path, finetune_file='559-ws-training.csv')
    else:
        print("Training directory not found.")