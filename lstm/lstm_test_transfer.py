import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import datetime
import joblib

# Hyperparameters
seq_len = 288
pred_step = 3
input_size = 5

# Helper function to load data
def load_ohio_data(filepath, seq_len=50, pred_step=3):
    df = pd.read_csv(filepath)
    df = df[['glucose_level', 'bolus_dose', 'meal_carbs']].fillna(0)
    df['meal_indicator'] = (df['meal_carbs'] > 0).astype(float)
    df['glucose_change'] = df['glucose_level'].diff().fillna(0)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(data_scaled) - seq_len - pred_step):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len+pred_step-1, 0])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y, scaler, df

# LSTM model
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

# Model paths
model_paths = {
    "baseline": "lstm/models_lstm/baseline_model.pth",
    "finetuned": "lstm/models_lstm/lstm_model_finetuned_559-ws-training.pth",
    "original": "lstm/models_lstm/lstm_model_559-ws-training.pth"
}

models = {}
scalers = {}

# use device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# try load a baseline scaler sidecar if available (fallback)
baseline_scaler = None
baseline_scaler_path = "lstm/models_lstm/baseline_scaler.joblib"
if os.path.exists(baseline_scaler_path):
    try:
        baseline_scaler = joblib.load(baseline_scaler_path)
        print(f"Loaded baseline scaler from {baseline_scaler_path}")
    except Exception as e:
        print(f"Warning: could not load baseline scaler: {e}")

# Load checkpoints that may contain non-tensor objects (e.g. sklearn StandardScaler)
with torch.serialization.safe_globals([StandardScaler]):
    for key, path in model_paths.items():
        if not os.path.exists(path):
            print(f"Warning: model file not found: {path} (skipping {key})")
            continue

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        scaler = checkpoint.get('scaler', None) if isinstance(checkpoint, dict) else None
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

        model = GlucoseLSTM(input_size)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models[key] = model
        scalers[key] = scaler

# Load test data
X_test, y_test, _, df_raw = load_ohio_data(
    filepath='data/ohio/2018/test_cleaned/559-ws-testing.csv',
    seq_len=seq_len,
    pred_step=pred_step
)

# Generate predictions (device + scaler fixes)
X_test_device = X_test.to(device)
y_preds = {}

with torch.no_grad():
    for key, model in models.items():
        model.to(device)
        model.eval()
        # forward on device, then bring preds to CPU/numpy
        out = model(X_test_device)  # tensor on device
        y_pred = out.cpu().numpy()
        zeros_placeholder = np.zeros((len(y_test), input_size - 1))

        # choose scaler (model scaler -> baseline fallback -> None)
        scaler_for_model = scalers.get(key, None)
        if scaler_for_model is None:
            scaler_for_model = baseline_scaler

        if scaler_for_model is None:
            print(f"Warning: no scaler for model '{key}', returning scaled predictions")
            y_pred_orig = y_pred.reshape(-1)
        else:
            y_pred_orig = scaler_for_model.inverse_transform(
                np.concatenate([y_pred.reshape(-1,1), zeros_placeholder], axis=1)
            )[:,0]

        y_preds[key] = y_pred_orig

# Inverse transform true values using baseline scaler (fallback)
zeros_placeholder = np.zeros((len(y_test), input_size - 1))
used_baseline = scalers.get('baseline') if scalers.get('baseline') is not None else baseline_scaler
if used_baseline is None:
    raise RuntimeError("No baseline scaler available to inverse-transform ground truth.")
y_test_orig = used_baseline.inverse_transform(
    np.concatenate([y_test.numpy().reshape(-1,1), zeros_placeholder], axis=1)
)[:,0]

# Plot directory
now = datetime.datetime.now()
timestamp = now.strftime("%y%m%d-%H%M")
plot_dir = f'lstm/plots_lstm/test_{timestamp}'
os.makedirs(plot_dir, exist_ok=True)

# Full comparison plot with markers
plt.figure(figsize=(12,6))
plt.plot(y_test_orig, label='Real Glucose', color='black', linewidth=1)
plt.scatter(range(len(y_test_orig)), y_test_orig, color='black', s=10)

plt.plot(y_preds['baseline'], label='Baseline', color='blue', linewidth=1)
plt.scatter(range(len(y_preds['baseline'])), y_preds['baseline'], color='blue', s=10)

plt.plot(y_preds['finetuned'], label='Fine-Tuned', color='red', linewidth=1)
plt.scatter(range(len(y_preds['finetuned'])), y_preds['finetuned'], color='red', s=10)

plt.plot(y_preds['original'], label='Original Training', color='green', linewidth=1)
plt.scatter(range(len(y_preds['original'])), y_preds['original'], color='green', s=10)

plt.xlabel('Time Index')
plt.ylabel('Glucose Level')
plt.title('Glucose Prediction Comparison')
plt.legend()
plt.ylim(50, 350)
plt.tight_layout()
plt.savefig(f'{plot_dir}/glucose_prediction_comparison.png')
plt.close()

# Segment plots (every 100 steps) with markers
pred_len = pred_step
context_len = 10

for i in range(0, len(y_test_orig) - (context_len + pred_len), 100):
    true_past = y_test_orig[i:i+context_len]
    true_future = y_test_orig[i+context_len:i+context_len+pred_len]

    plt.figure(figsize=(8,4))
    plt.plot(range(context_len), true_past, 'k-', linewidth=2, label='Last 10 Real')
    plt.scatter(range(context_len), true_past, color='k', s=30, zorder=3)

    x_future = range(context_len-1, context_len+pred_len)
    true_future_plot = np.concatenate([[true_past[-1]], true_future])
    plt.plot(x_future, true_future_plot, 'g-', linewidth=2, label='Next 3 Real')
    plt.scatter(range(context_len, context_len+pred_len), true_future, color='g', s=40, zorder=3)

    for key, color in zip(['baseline','finetuned','original'], ['b','r','c']):
        pred_future = y_preds[key][i+context_len:i+context_len+pred_len]
        pred_plot = np.concatenate([[true_past[-1]], pred_future])
        plt.plot(x_future, pred_plot, linestyle='--', color=color, label=f'{key} Prediction')
        plt.scatter(range(context_len, context_len+pred_len), pred_future, color=color, s=50, marker='o', zorder=4)

    plt.axvline(x=context_len-1, color='gray', linestyle='--')
    plt.xlabel('Relative Time Index')
    plt.ylabel('Glucose Level')
    plt.title(f'Segment {i}')
    plt.legend()
    plt.ylim(50, 350)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/segment_{i:05d}_comparison.png')
    plt.close()

# Save CSVs for segments
csv_dir = os.path.join(plot_dir, "segments_csv")
os.makedirs(csv_dir, exist_ok=True)

for i in range(0, len(y_test_orig) - (context_len + pred_len), 100):
    true_past_rows = df_raw.iloc[i : i + context_len]
    true_future_rows = df_raw.iloc[i + context_len : i + context_len + pred_len]

    data = []

    for j, row in enumerate(true_past_rows.itertuples(), start=0):
        data.append({
            'Type': 'Real_Past',
            'Relative_Index': j,
            'Bolus_Dose': row.bolus_dose,
            'Meal_Carbs': row.meal_carbs,
            'Real_Glucose': row.glucose_level,
            'Baseline_Pred': None,
            'FineTuned_Pred': None,
            'Original_Pred': None
        })

    for j, row in enumerate(true_future_rows.itertuples(), start=context_len):
        idx = i+context_len + j-context_len
        data.append({
            'Type': 'Comparison',
            'Relative_Index': j,
            'Bolus_Dose': row.bolus_dose,
            'Meal_Carbs': row.meal_carbs,
            'Real_Glucose': row.glucose_level,
            'Baseline_Pred': y_preds['baseline'][idx],
            'FineTuned_Pred': y_preds['finetuned'][idx],
            'Original_Pred': y_preds['original'][idx]
        })

    df_segment = pd.DataFrame(data)
    df_segment.to_csv(os.path.join(csv_dir, f'segment_{i:05d}_comparison.csv'), index=False)

print(f"All segment plots and CSVs saved in: {plot_dir}")
