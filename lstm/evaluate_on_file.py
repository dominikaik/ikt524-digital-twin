import os
import sys
import torch
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Try to import helper/predict_iteratively and GlucoseLSTM from the training module.
# If running the script from project root, `lstm.lstm_transfer_train` should be importable.
try:
    from lstm.lstm_transfer_train import predict_iteratively, GlucoseLSTM
except Exception:
    # Fallback: add parent project path if called directly from this folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from lstm.lstm_transfer_train import predict_iteratively, GlucoseLSTM


def evaluate_on_file(model, file_path, scaler, seq_len, device=None, max_horizon=12, out_dir="lstm/models_lstm"):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    model.eval()

    df_raw = pd.read_csv(file_path)
    df = df_raw[['glucose_level', 'bolus_dose', 'meal_carbs']].fillna(0)
    df['meal_indicator'] = (df['meal_carbs'] > 0).astype(float)
    df['glucose_change'] = df['glucose_level'].diff().fillna(0)

    used_scaler = scaler
    if used_scaler is None:
        baseline_path = os.path.join(out_dir, "baseline_scaler.joblib")
        if os.path.exists(baseline_path):
            used_scaler = joblib.load(baseline_path)

    if used_scaler is None:
        raise RuntimeError("No scaler provided and no baseline_scaler found. Cannot evaluate in original scale.")

    data_scaled = used_scaler.transform(df)
    N = len(data_scaled)
    if N - seq_len - max_horizon + 1 <= 0:
        raise ValueError("Not enough data in file for the requested horizons/sequence length.")

    indices = range(0, N - seq_len - max_horizon + 1)
    all_preds = []
    truths_all = []

    for idx in tqdm(indices, desc="Iterative eval (per-start index)"):
        init_scaled = torch.tensor(data_scaled[idx:idx+seq_len], dtype=torch.float32)
        pred_seq_orig = predict_iteratively(model, init_scaled, steps=max_horizon, scaler=used_scaler)
        all_preds.append(pred_seq_orig)

        truths_row = []
        for h in range(1, max_horizon + 1):
            truth_row_scaled = data_scaled[idx + seq_len + h - 1]
            truth_orig = used_scaler.inverse_transform(truth_row_scaled.reshape(1, -1))[0, 0]
            truths_row.append(truth_orig)
        truths_all.append(truths_row)

    all_preds = np.array(all_preds)
    truths_all = np.array(truths_all)
    num = all_preds.shape[0]

    # compute metrics for selected horizons (if available)
    metrics = {}
    for h in (1, 3, 12):
        if h <= max_horizon:
            preds_h = all_preds[:, h-1]
            truths_h = truths_all[:, h-1]
            mse_h = mean_squared_error(truths_h, preds_h)
            rmse_h = float(np.sqrt(mse_h))
            mae_h = mean_absolute_error(truths_h, preds_h)
            metrics[h] = {"rmse": rmse_h, "mae": mae_h}
            print(f"Iterative {h}-step -> RMSE: {rmse_h:.2f} | MAE: {mae_h:.2f}")

    # Build CSV with y_true_1..y_true_max and pred_1..pred_max
    out_dict = {"start_index": np.arange(num)}
    for step in range(1, max_horizon + 1):
        out_dict[f"y_true_{step}"] = truths_all[:, step-1]
    for step in range(1, max_horizon + 1):
        out_dict[f"pred_{step}"] = all_preds[:, step-1]

    os.makedirs(out_dir, exist_ok=True)
    # include the out_dir's folder name (model/folder tag) in the file name
    model_tag = os.path.basename(os.path.normpath(out_dir))
    base_name = os.path.basename(file_path).replace('.csv', '')
    out_path = os.path.join(out_dir, f"eval_{base_name}_{model_tag}_all{max_horizon}.csv")

    pd.DataFrame(out_dict).to_csv(out_path, index=False)
    print(f"Saved full-horizon CSV with real values: {out_path}")

    return metrics


def _load_checkpoint_model(checkpoint_path, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt = torch.load(checkpoint_path, map_location=device)

    scaler = None
    state_dict = None

    # handle wrapper dict vs plain state_dict
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt or "state_dict" in ckpt or "scaler" in ckpt:
            scaler = ckpt.get("scaler", None)
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
        else:
            # likely a plain state_dict (mapping of parameter tensors)
            state_dict = ckpt
    else:
        state_dict = ckpt

    # if no scaler inside checkpoint, try to load a scaler file next to the ckpt
    if scaler is None:
        ckpt_dir = os.path.dirname(checkpoint_path)
        for cand in ("baseline_scaler.joblib", "scaler.joblib", "scaler.pkl"):
            cand_path = os.path.join(ckpt_dir, cand)
            if os.path.exists(cand_path):
                try:
                    scaler = joblib.load(cand_path)
                    break
                except Exception:
                    pass

    # determine input_size from scaler if possible
    input_size = 5
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        input_size = scaler.n_features_in_

    model = GlucoseLSTM(input_size)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    else:
        raise RuntimeError("Checkpoint did not contain a model state dict.")
    model.to(device)
    model.eval()
    return model, scaler


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model iteratively on a CSV (save preds y_true_1..12, pred_1..12)")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to model checkpoint (.pth) containing 'model_state_dict' and optional 'scaler'")
    parser.add_argument("--file", type=str, required=True, help="CSV file to evaluate")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--max_horizon", type=int, default=12)
    parser.add_argument("--out_dir", type=str, default="lstm/models_lstm")
    args = parser.parse_args()

    model, scaler = _load_checkpoint_model(args.model_ckpt)
    evaluate_on_file(model, args.file, scaler, seq_len=args.seq_len, max_horizon=args.max_horizon, out_dir=args.out_dir)