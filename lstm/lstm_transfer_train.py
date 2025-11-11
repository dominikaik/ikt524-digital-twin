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
import argparse

# Hyperparameters
epochs_baseline=1
epochs_finetune=1
seq_len=50


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

def load_ohio_data(filepath, seq_len=seq_len):
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
                                seq_len=seq_len, batch_size=16,
                                epochs_baseline=epochs_baseline, epochs_finetune=epochs_finetune, lr=0.001,
                                finetune_file='559-ws-training.csv', out_dir="lstm/models_lstm"):

    # create run subfolders: base_line and fine_tuned
    os.makedirs(out_dir, exist_ok=True)
    base_dir = os.path.join(out_dir, "base_line")
    finetune_dir = os.path.join(out_dir, "fine_tuned")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(finetune_dir, exist_ok=True)
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

    # keep tensors on CPU for DataLoader workers, move to device inside the loop
    X_all = torch.cat(X_all)
    y_all = torch.cat(y_all)
    train_loader = DataLoader(TensorDataset(X_all, y_all), batch_size=batch_size, shuffle=True, num_workers=0)

    model = GlucoseLSTM(X_all.shape[2]).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\nTraining baseline model ...")
    for epoch in tqdm(range(epochs_baseline), desc="Baseline Epochs"):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(base_dir, "baseline_model.pth"))
    joblib.dump(scaler, os.path.join(base_dir, "baseline_scaler.joblib"))
    print("Baseline model saved ->", base_dir)

    # Fine-tuning
    print(f"\nFine-tuning on {target_file} ...")
    X_target, y_target, scaler_target = load_ohio_data(os.path.join(data_dir, target_file), seq_len)
    train_size = int(0.8 * len(X_target))
    X_train, X_test = X_target[:train_size].to(device), X_target[train_size:].to(device)
    y_train, y_test = y_target[:train_size].to(device), y_target[train_size:].to(device)

    # keep num_workers=0 to avoid CUDA init issues in worker processes
    train_loader_target = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)

    model.load_state_dict(torch.load(os.path.join(base_dir, "baseline_model.pth"), map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.5)

    for epoch in tqdm(range(epochs_finetune), desc="Finetune Epochs"):
        model.train()
        for xb, yb in train_loader_target:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    finetune_path = os.path.join(finetune_dir, f"lstm_model_finetuned_{target_file.replace('.csv','')}.pth")
    torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler_target}, finetune_path)
    print("Fine-tuned model saved ->", finetune_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train / finetune Glucose LSTM")
    parser.add_argument("--data_path", type=str, default="data/ohio/2018/train_cleaned/", help="Path to training CSV folder")
    parser.add_argument("--finetune_file", type=str, default="559-ws-training.csv", help="Filename in data_path to finetune on")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs_baseline", type=int, default=15, help="Epochs for baseline training")
    parser.add_argument("--epochs_finetune", type=int, default=10, help="Epochs for finetuning")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # Call training function with CLI-provided (or default) hyperparameters
    if os.path.exists(args.data_path):
        train_baseline_and_finetune(
            data_dir=args.data_path,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs_baseline=args.epochs_baseline,
            epochs_finetune=args.epochs_finetune,
            lr=args.lr,
            finetune_file=args.finetune_file
        )
    else:
        print("Training directory not found:", args.data_path)