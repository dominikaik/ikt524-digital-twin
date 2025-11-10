#!/usr/bin/env python3
from pathlib import Path
import json
import os
import glob
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# Default paths
DEFAULT_JSON = Path("/home/coder/digital_twin/twin/simulation_data/json/block_00000.json")
DEFAULT_FUTURE_GLOB = "/home/coder/digital_twin/twin/simulation_data/json/future_block_*.json"
DEFAULT_CKPT = Path("/home/coder/digital_twin/twin/simulation_data/models/lstm_model_finetuned_559-ws-training.pth")
SEQ_LEN = 50
MAX_HORIZON = 12

# === DATA PREPARATION ===
def _prepare_df_from_list(data_list):
    df = pd.DataFrame(data_list)
    for c in ("glucose_level", "bolus_dose", "meal_carbs"):
        if c not in df.columns:
            df[c] = 0.0
    df["meal_indicator"] = (df["meal_carbs"] > 0).astype(float)
    df["glucose_change"] = df["glucose_level"].diff().fillna(0)
    return df[["glucose_level", "bolus_dose", "meal_carbs", "meal_indicator", "glucose_change"]]

# === ORIGINAL LSTM ARCHITEKTUR ===
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

# === MODEL LOADING ===
def _load_model_and_scaler(ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    scaler = None
    state_dict = None

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
        scaler = ckpt.get("scaler", ckpt.get("scaler_object", None))
        if state_dict is None and isinstance(ckpt.get("model", None), nn.Module):
            model = ckpt["model"]
            model.to(device)
            model.eval()
            return model, scaler
    else:
        state_dict = ckpt

    if state_dict is None:
        raise RuntimeError("Checkpoint does not contain a model state_dict.")

    # input size aus state dict ableiten
    input_size = 5  # default falls scaler unbekannt
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        input_size = scaler.n_features_in_

    model = GlucoseLSTM(input_size)
    try:
        model.load_state_dict(state_dict)
    except Exception:
        # Remap keys falls prefix "model." existiert
        remapped = {k[len("model."):]: v if k.startswith("model.") else v for k, v in state_dict.items()}
        model.load_state_dict(remapped, strict=False)

    model.to(device)
    model.eval()
    return model, scaler

# === ITERATIVE PREDICTION ===
def _predict_iteratively_local(model, initial_sequence, steps, scaler=None, future_covariates_scaled=None, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    seq = initial_sequence.clone().detach().to(device)
    preds_scaled = []

    with torch.no_grad():
        for t in range(steps):
            inp = seq.unsqueeze(0)
            out = model(inp)
            pred_scaled = out.cpu().item()
            preds_scaled.append(pred_scaled)

            n_feats = seq.shape[1]
            new_row = torch.zeros((1, n_feats), device=seq.device, dtype=seq.dtype)
            new_row[0, 0] = torch.tensor(pred_scaled, device=seq.device, dtype=seq.dtype)

            if future_covariates_scaled is not None and t < len(future_covariates_scaled):
                cov_t = torch.tensor(future_covariates_scaled[t], device=seq.device, dtype=seq.dtype)
                new_row[0, 1:1 + len(cov_t)] = cov_t

            last_glucose_scaled = seq[-1, 0]
            new_row[0, -1] = new_row[0, 0] - last_glucose_scaled
            seq = torch.cat([seq[1:], new_row], dim=0)

    if scaler is not None:
        zeros = np.zeros((len(preds_scaled), seq.shape[1] - 1))
        stacked = np.concatenate([np.array(preds_scaled).reshape(-1, 1), zeros], axis=1)
        preds_orig = scaler.inverse_transform(stacked)[:, 0]
        return preds_orig
    return np.array(preds_scaled)

# === FORECAST ===
def forecast(data, ckpt_path: str | Path = None, future_glob: str = DEFAULT_FUTURE_GLOB):
    if not isinstance(data, (list, tuple)) or len(data) == 0:
        return {"error": "input must be a non-empty list"}

    ckpt_path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    if not ckpt_path.exists():
        return {"error": f"checkpoint not found: {ckpt_path}"}

    df_block = _prepare_df_from_list(data)

    # Prüfe Länge der Sequence
    seq_len = min(len(df_block), SEQ_LEN)
    init_scaled = df_block[-seq_len:].copy()  # letzte seq_len Zeilen
    print(f"[DEBUG] Using sequence length: {seq_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lade Modell + Scaler
    try:
        model, scaler = _load_model_and_scaler(ckpt_path, device)
    except Exception as e:
        return {"error": f"failed to load model: {e}"}

    if scaler is None:
        for cand in ("baseline_scaler.joblib", "scaler.joblib", "scaler.pkl"):
            p = ckpt_path.parent / cand
            if p.exists():
                scaler = joblib.load(p)
                break
    if scaler is None:
        return {"error": "no scaler available"}

    data_scaled = scaler.transform(init_scaled)
    init_scaled_tensor = torch.tensor(data_scaled, dtype=torch.float32)

    # --- FUTURE COVARIATES ---
    future_cov_scaled = None
    future_candidates = sorted(glob.glob(future_glob))
    if future_candidates:
        try:
            with open(future_candidates[0], "r", encoding="utf-8") as f:
                future_data = json.load(f)
            df_future = _prepare_df_from_list(future_data)
            future_scaled_full = scaler.transform(df_future)
            future_cov_scaled = future_scaled_full[:, 1:4]  # bolus, meal_carbs, meal_indicator
            print(f"[DEBUG] Loaded {len(df_future)} future steps, future_cov_scaled shape={future_cov_scaled.shape}")
        except Exception as e:
            print(f"[DEBUG WARNING] could not load future JSON: {e}")
            future_cov_scaled = None

    # --- Vorhersagen ---
    preds_no_future = _predict_iteratively_local(
        model, init_scaled_tensor, steps=MAX_HORIZON, scaler=scaler, future_covariates_scaled=None, device=device
    )
    preds_with_future = _predict_iteratively_local(
        model, init_scaled_tensor, steps=MAX_HORIZON, scaler=scaler, future_covariates_scaled=future_cov_scaled, device=device
    )

    print("[DEBUG] Initial glucose (scaled):", init_scaled_tensor[:,0].numpy())
    if future_cov_scaled is not None:
        print("[DEBUG] Future covariates (scaled):", future_cov_scaled[:10])

    return {
        "model_checkpoint": str(ckpt_path),
        "device": str(device),
        "csv_no_future": [float(x) for x in preds_no_future],
        "csv_with_future": [float(x) for x in preds_with_future],
    }

# === CLI ENTRY ===
if __name__ == "__main__":
    path = DEFAULT_JSON
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    if not path.exists():
        print(json.dumps({"error": f"JSON not found: {path}"}))
        sys.exit(2)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out = forecast(data)
    print(json.dumps(out, ensure_ascii=False, indent=2))
