import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import datetime
import joblib


# 1. Hyperparameters

seq_len = 288
pred_step = 3
input_size = 5  
output_size = 1


# 2. Helper function to load data

def load_ohio_data(filepath, seq_len=50, pred_step=3):
    df = pd.read_csv(filepath)
    df = df[['glucose_level', 'bolus_dose', 'meal_carbs']].fillna(0)

    # New features
    df['meal_indicator'] = (df['meal_carbs'] > 0).astype(float)
    df['glucose_change'] = df['glucose_level'].diff().fillna(0)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(data_scaled) - seq_len - pred_step):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len+pred_step-1, 0])  # predict glucose

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y, scaler


# 3. LSTM model (same as in training code)

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


# 4. Load model & scaler

model_path = 'models_lstm/lstm_model_559-ws-training.pth'  # adjust path if needed

# Safely load scaler from checkpoint
from sklearn.preprocessing import StandardScaler
with torch.serialization.safe_globals([StandardScaler]):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

scaler = checkpoint['scaler']

# Instantiate model & load weights
model = GlucoseLSTM(input_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# 5. Load test data

X_test, y_test, _ = load_ohio_data(
    filepath='data/ohio/2018/test_cleaned/559-ws-testing.csv',
    seq_len=seq_len,
    pred_step=pred_step
)


# 6. Generate predictions

with torch.no_grad():
    y_pred_test = model(X_test)


# 7. Inverse transform

zeros_placeholder = np.zeros((len(y_test), input_size - 1))
y_test_orig = scaler.inverse_transform(
    np.concatenate([y_test.numpy().reshape(-1, 1), zeros_placeholder], axis=1)
)[:, 0]

y_pred_orig = scaler.inverse_transform(
    np.concatenate([y_pred_test.numpy().reshape(-1, 1), zeros_placeholder], axis=1)
)[:, 0]


# 8. Create plot directory

now = datetime.datetime.now()
timestamp = now.strftime("%y%m%d-%H%M")
plot_dir = f'./plots_lstm/test_{timestamp}'
os.makedirs(plot_dir, exist_ok=True)


# 9. Full plot

plt.figure(figsize=(12, 6))
plt.plot(y_test_orig, label='Real Glucose')
plt.plot(y_pred_orig, label='Predicted Glucose')
plt.xlabel('Time Index')
plt.ylabel('Glucose Level')
plt.title('Glucose Prediction on Test Data (LSTM)')
plt.legend()
plt.ylim(50, 350)              # fixed glucose range
plt.tight_layout()
plt.savefig(f'{plot_dir}/glucose_prediction_full.png')
plt.close()
print(f"Full plot saved at: {plot_dir}/glucose_prediction_full.png")


# 10. Segment plots (every 100 steps)

pred_len = pred_step
context_len = 10

for i in range(0, len(y_test_orig) - (context_len + pred_len), 100):
    true_past = y_test_orig[i:i+context_len]
    true_future = y_test_orig[i+context_len:i+context_len+pred_len]
    pred_future = y_pred_orig[i+context_len:i+context_len+pred_len]

    plt.figure(figsize=(8, 4))
    plt.plot(range(context_len), true_past, 'b-', linewidth=2, label='Last 10 Real')
    plt.plot(range(context_len-1, context_len+pred_len), np.concatenate([[true_past[-1]], true_future]), 'g-', linewidth=2, label='Next 3 Real')
    plt.plot(range(context_len-1, context_len+pred_len), np.concatenate([[true_past[-1]], pred_future]), 'r--', linewidth=2, label='Next 3 Predicted')
    plt.scatter(range(context_len), true_past, color='b', zorder=3)
    plt.scatter(range(context_len, context_len+pred_len), true_future, color='g', zorder=3)
    plt.scatter(range(context_len, context_len+pred_len), pred_future, color='r', zorder=3)
    plt.axvline(x=context_len-1, color='gray', linestyle='--')
    plt.xlabel('Relative Time Index')
    plt.ylabel('Glucose Level')
    plt.title(f'Segment {i}')
    plt.legend()
    plt.ylim(50, 350)          # fixed glucose range for segment plots
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/segment_{i:05d}.png')
    plt.close()

print(f"Segment plots saved in: {plot_dir}")


# 11. Save segment CSVs (optional)

csv_dir = os.path.join(plot_dir, "segments_csv")
os.makedirs(csv_dir, exist_ok=True)

df_raw = pd.read_csv('data/ohio/2018/test_cleaned/559-ws-testing.csv')
df_raw = df_raw[['glucose_level', 'bolus_dose', 'meal_carbs']].fillna(0)

for i in range(0, len(y_test_orig) - (context_len + pred_len), 100):
    true_past = df_raw.iloc[i : i + context_len]
    true_future = df_raw.iloc[i + context_len : i + context_len + pred_len]
    pred_future_glucose = y_pred_orig[i + context_len : i + context_len + pred_len]

    data = []

    # past values
    for j, row in enumerate(true_past.itertuples(), start=0):
        data.append({
            'Type': 'Real_Past',
            'Relative_Index': j,
            'Bolus_Dose': row.bolus_dose,
            'Meal_Carbs': row.meal_carbs,
            'Real_Glucose': row.glucose_level,
            'Predicted_Glucose': None
        })

    # future comparison
    for j, (pred_val, row) in enumerate(zip(pred_future_glucose, true_future.itertuples()), start=context_len):
        data.append({
            'Type': 'Comparison',
            'Relative_Index': j,
            'Bolus_Dose': row.bolus_dose,
            'Meal_Carbs': row.meal_carbs,
            'Real_Glucose': row.glucose_level,
            'Predicted_Glucose': pred_val
        })

    df_segment = pd.DataFrame(data)
    csv_path = os.path.join(csv_dir, f'segment_{i:05d}_values.csv')
    df_segment.to_csv(csv_path, index=False)

print(f"CSV files for all 100-step segments saved in: {csv_dir}")
