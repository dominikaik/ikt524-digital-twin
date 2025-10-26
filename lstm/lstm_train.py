import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    df = df[['glucose_level', 'bolus_dose', 'meal_carbs']].fillna(0)
    df['meal_indicator'] = (df['meal_carbs'] > 0).astype(float)
    df['glucose_change'] = df['glucose_level'].diff().fillna(0)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(data_scaled) - seq_len - 1):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len, 0])  # Nur ein Schritt weiter

    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(y), dtype=torch.float32).to(device)
    return X, y, scaler

def train_lstm_for_all_patients(data_dir='data/ohio/2018/train_cleaned/', seq_len=50, batch_size=16, epochs=20, lr=0.001):
    os.makedirs("lstm/models_lstm", exist_ok=True)
    results = []
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    if not TRAIN_ALL and csv_files:
        csv_files = [csv_files[0]]
        print(f"TRAIN_ALL=False → only '{csv_files[0]}' will be trained.")

    for filename in csv_files:
        filepath = os.path.join(data_dir, filename)
        print(f"\nTraining for: {filename}")

        X, y, scaler = load_ohio_data(filepath, seq_len)
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

        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test).cpu().numpy()
            y_test_np = y_test.cpu().numpy()

        zeros = np.zeros((len(y_test_np), X.shape[2]-1))
        y_test_orig = scaler.inverse_transform(np.concatenate([y_test_np.reshape(-1,1), zeros], axis=1))[:,0]
        y_pred_orig = scaler.inverse_transform(np.concatenate([y_pred_test.reshape(-1,1), zeros], axis=1))[:,0]

        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        print(f"Test RMSE: {rmse:.2f} | MAE: {mae:.2f}")

        model_path = os.path.join("lstm/models_lstm", f"lstm_model_{filename.replace('.csv','')}.pth")
        torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler}, model_path)

        results.append({"patient": filename, "rmse": rmse, "mae": mae})

    df_results = pd.DataFrame(results)
    df_results.to_csv("lstm_evaluation_results.csv", index=False)
    print("Evaluation results saved to lstm_evaluation_results.csv")

if __name__ == '__main__':
    if os.path.exists('data/ohio/2018/train_cleaned/'):
        train_lstm_for_all_patients('data/ohio/2018/train_cleaned/')
    else:
        print("Training directory not found.")