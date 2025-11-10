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

# ------------------ DEFAULT PATHS ------------------
DEFAULT_JSON = Path("/home/coder/digital_twin/twin/simulation_data/json/block_00000.json")
DEFAULT_FUTURE_GLOB = "/home/coder/digital_twin/twin/simulation_data/json/future_block_*.json"
DEFAULT_CKPT = Path("/home/coder/digital_twin/twin/simulation_data/models/lstm_model_finetuned_559-ws-training.pth")
SEQ_LEN = 50
MAX_HORIZON = 12

# ------------------ DATA PREPARATION ------------------
def _prepare_df_from_list(data_list):
    df = pd.DataFrame(data_list)
    for c in ("glucose_level", "bolus_dose", "meal_carbs"):
        if c not in df.columns:
            df[c] = 0.0
    df["meal_indicator"] = (df["meal_carbs"] > 0).astype(float)
    df["glucose_change"] = df["glucose_level"].diff().fillna(0)
    return df[["glucose_level", "bolus_dose", "meal_carbs", "meal_indicator", "glucose_change"]]

# ------------------ LSTM MODEL ------------------
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

# ------------------ MODEL LOADING ------------------
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

    input_size = 5
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        input_size = scaler.n_features_in_

    model = GlucoseLSTM(input_size)
    try:
        model.load_state_dict(state_dict)
    except Exception:
        remapped = {k[len("model."):]: v if k.startswith("model.") else v for k, v in state_dict.items()}
        model.load_state_dict(remapped, strict=False)

    model.to(device)
    model.eval()
    return model, scaler

# ------------------ ITERATIVE PREDICTION ------------------
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

# ------------------ DIGITAL TWIN FUNCTIONS ------------------
def _tri_kernel(t_min, peak_at=60, total=180):
    t = np.asarray(t_min, dtype=float)
    w = np.zeros_like(t)
    rise = (t >= 0) & (t <= peak_at)
    fall = (t > peak_at) & (t <= total)
    w[rise] = (t[rise] / max(peak_at, 1e-6))
    w[fall] = (total - t[fall]) / max(total - peak_at, 1e-6)
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else w

def _make_profile(length_steps, at_step, magnitude, kernel_minutes, dt_min):
    prof = np.zeros(length_steps, dtype=float)
    if at_step is None or magnitude == 0:
        return prof
    grid = np.arange(0, length_steps) * dt_min
    w = _tri_kernel(grid - at_step * dt_min, peak_at=kernel_minutes[0], total=kernel_minutes[1])
    if w.sum() > 0:
        w = w * (magnitude / w.sum())
    return w

# --- Residual (Abklingprofil) Erweiterung ---
def _make_residual_profile_from_past(df_now, feature, dt_min, peak_min, total_min, magnitude_scale=1.0):
    """Erzeugt ein abklingendes Restprofil für vergangene Ereignisse."""
    prof = np.zeros(MAX_HORIZON, dtype=float)
    idx_arr = df_now[feature].to_numpy().nonzero()[0]
    if len(idx_arr) == 0:
        return prof

    last_idx = idx_arr[-1]  # letzter bekannter Zeitpunkt
    steps_since = len(df_now) - 1 - last_idx
    if steps_since < 0:
        return prof

    grid = np.arange(0, total_min + dt_min, dt_min)
    w = _tri_kernel(grid, peak_at=peak_min, total=total_min)
    remaining = w[steps_since:]
    if remaining.sum() > 0:
        remaining = remaining * (df_now[feature].iloc[last_idx] * magnitude_scale / remaining.sum())
    return remaining[:MAX_HORIZON]

def simulate_meal_insulin_hybrid(
    model, scaler, df_now, feature_names,
    steps=12, seq_len=36, dt_min=5,
    meal_at_step=None, meal_grams=0.0,
    bolus_at_step=None, bolus_units=0.0,
    carb_peak_min=60, carb_total_min=180,
    ins_peak_min=75, ins_total_min=300,
    ISF_mgdl_per_U=50.0, CR_g_per_U=10.0,
    hold_inputs="last",
):
    df_now = df_now.copy()
    for col in feature_names:
        if col not in df_now.columns:
            df_now[col] = 0.0
    df_now = df_now[feature_names]

    F = len(feature_names)
    glucose_idx = feature_names.index("glucose_level")
    meal_idx = feature_names.index("meal_carbs") if "meal_carbs" in feature_names else None
    bolus_idx = feature_names.index("bolus_dose") if "bolus_dose" in feature_names else None

    device = next(model.parameters()).device
    arr_scaled = scaler.transform(df_now.values)
    seq = arr_scaled[-seq_len:].copy()
    base_cov_orig = df_now.values[-1, :].astype(np.float32)
    if hold_inputs == "zero":
        base_cov_orig = np.zeros_like(base_cov_orig)

    carb_profile_feat  = _make_profile(steps, meal_at_step, meal_grams, (carb_peak_min, carb_total_min), dt_min)
    bolus_profile_feat = _make_profile(steps, bolus_at_step, bolus_units, (0.1, dt_min), dt_min)

    # --- Residual (Abklingprofil) hinzufügen ---
    carb_residual = _make_residual_profile_from_past(df_now, "meal_carbs", dt_min, carb_peak_min, carb_total_min, 1.0)
    bolus_residual = _make_residual_profile_from_past(df_now, "bolus_dose", dt_min, ins_peak_min, ins_total_min, 1.0)
    carb_profile_feat  = carb_profile_feat + carb_residual[:steps]
    bolus_profile_feat = bolus_profile_feat + bolus_residual[:steps]

    total_rise_mgdl = (meal_grams / max(CR_g_per_U, 1e-6)) * ISF_mgdl_per_U
    carb_effect = _make_profile(steps, meal_at_step, total_rise_mgdl, (carb_peak_min, carb_total_min), dt_min)
    total_fall_mgdl = bolus_units * ISF_mgdl_per_U
    insulin_effect = _make_profile(steps, bolus_at_step, total_fall_mgdl, (ins_peak_min, ins_total_min), dt_min)

    preds_mgdl = []
    for step in range(steps):
        x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            y_scaled = model(x).cpu().numpy().ravel()[0]

        tmp = np.zeros((1, F), dtype=np.float32)
        tmp[:, glucose_idx] = y_scaled
        glu_mgdl = scaler.inverse_transform(tmp)[0, glucose_idx]

        # physiologische Effekte (inkl. Abklingprofile)
        glu_mgdl = glu_mgdl + carb_effect[step] - insulin_effect[step]
        glu_mgdl = glu_mgdl + carb_residual[step] * ISF_mgdl_per_U / max(CR_g_per_U, 1e-6)
        glu_mgdl = glu_mgdl - bolus_residual[step] * ISF_mgdl_per_U

        preds_mgdl.append(glu_mgdl)

        feat_orig = base_cov_orig.copy()
        feat_orig[glucose_idx] = glu_mgdl
        if meal_idx is not None:
            feat_orig[meal_idx] = carb_profile_feat[step]
        if bolus_idx is not None:
            feat_orig[bolus_idx] = bolus_profile_feat[step]

        feat_scaled = scaler.transform(feat_orig.reshape(1, -1))[0]
        seq = np.vstack([seq[1:], feat_scaled])

    return pd.DataFrame({
        "glucose_pred":   np.array(preds_mgdl, dtype=float),
        "carb_effect":    carb_effect.astype(float),
        "insulin_effect": insulin_effect.astype(float),
        "carb_profile":   carb_profile_feat.astype(float),
        "bolus_profile":  bolus_profile_feat.astype(float),
    })

# ------------------ FORECAST ------------------
def forecast(data, ckpt_path: str | Path = None, future_glob: str = DEFAULT_FUTURE_GLOB):
    if not isinstance(data, (list, tuple)) or len(data) == 0:
        return {"error": "input must be a non-empty list"}

    ckpt_path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    if not ckpt_path.exists():
        return {"error": f"checkpoint not found: {ckpt_path}"}

    df_block = _prepare_df_from_list(data)
    seq_len = min(len(df_block), SEQ_LEN)
    init_scaled = df_block[-seq_len:].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler = _load_model_and_scaler(ckpt_path, device)

    if scaler is None:
        for cand in ("baseline_scaler.joblib", "scaler.joblib", "scaler.pkl"):
            p = ckpt_path.parent / cand
            if p.exists():
                scaler = joblib.load(p)
                break
    if scaler is None:
        return {"error": "no scaler available"}

    init_scaled_tensor = torch.tensor(scaler.transform(init_scaled), dtype=torch.float32)

    # --- FUTURE COVARIATES & Digital-Twin Parameter ---
    future_cov_scaled = None
    meal_at_step = None
    meal_grams = 0
    bolus_at_step = None
    bolus_units = 0

    future_candidates = sorted(glob.glob(future_glob))
    if future_candidates:
        try:
            with open(future_candidates[0], "r", encoding="utf-8") as f:
                future_data = json.load(f)
            df_future = _prepare_df_from_list(future_data)
            future_cov_scaled = scaler.transform(df_future)[:, 1:4]

            meal_idx_arr = df_future["meal_carbs"].to_numpy().nonzero()[0]
            if len(meal_idx_arr) > 0:
                meal_at_step = int(meal_idx_arr[0])
                meal_grams = float(df_future["meal_carbs"].iloc[meal_at_step])
            bolus_idx_arr = df_future["bolus_dose"].to_numpy().nonzero()[0]
            if len(bolus_idx_arr) > 0:
                bolus_at_step = int(bolus_idx_arr[0])
                bolus_units = float(df_future["bolus_dose"].iloc[bolus_at_step])
        except Exception:
            future_cov_scaled = None

    preds_no_future = _predict_iteratively_local(model, init_scaled_tensor, steps=MAX_HORIZON, scaler=scaler, future_covariates_scaled=None, device=device)
    preds_with_future = _predict_iteratively_local(model, init_scaled_tensor, steps=MAX_HORIZON, scaler=scaler, future_covariates_scaled=future_cov_scaled, device=device)

    feature_names = ["glucose_level", "bolus_dose", "meal_carbs", "meal_indicator", "glucose_change"]
    digital_twin_df = simulate_meal_insulin_hybrid(
        model, scaler, df_block, feature_names,
        steps=MAX_HORIZON, seq_len=seq_len, dt_min=5,
        meal_at_step=meal_at_step, meal_grams=meal_grams,
        bolus_at_step=bolus_at_step, bolus_units=bolus_units,
        ISF_mgdl_per_U=50, CR_g_per_U=10
    )

    return {
        "model_checkpoint": str(ckpt_path),
        "device": str(device),
        "csv_no_future": [float(x) for x in preds_no_future],
        "csv_with_future": [float(x) for x in preds_with_future],
        "digital_twin": digital_twin_df["glucose_pred"].tolist()
    }

# ------------------ CLI ENTRY ------------------
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
