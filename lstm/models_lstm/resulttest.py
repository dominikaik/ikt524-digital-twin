import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


fp = sys.argv[1] if len(sys.argv) > 1 else "lstm/models_lstm/eval_559-ws-testing.csv"
df = pd.read_csv(fp)

results = []
n_horizons = 12

for i in range(1, n_horizons + 1):
    y_col = f"y_true_{i}"
    p_col = f"pred_{i}"
    if y_col not in df.columns or p_col not in df.columns:
        print(f"Horizon {i} columns not found in CSV")
        continue

    sub = df[[y_col, p_col]].dropna()
    if sub.empty:
        print(f"No valid values for Horizon {i} — skipping")
        continue

    y_true = sub[y_col].values
    y_pred = sub[p_col].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    results.append({
        "horizon": i,
        "n": len(sub),
        "MAE": mae,
        "RMSE": rmse
    })

metrics_df = pd.DataFrame(results).set_index("horizon")
metrics_df["MAE"] = metrics_df["MAE"].round(4)
metrics_df["RMSE"] = metrics_df["RMSE"].round(4)

print(metrics_df)

out_path = fp.replace(".csv", "_metrics_per_horizon.csv")
metrics_df.to_csv(out_path)
print(f"Saved metrics: {out_path}")
