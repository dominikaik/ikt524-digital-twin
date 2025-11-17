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

def _align_df_to_scaler(df: pd.DataFrame, scaler):
    """
    Return a DataFrame with columns ordered like scaler.feature_names_in_ (if present).
    Missing columns are added with zeros. Extra columns are dropped.
    """
    if scaler is None:
        return df
    feat = getattr(scaler, "feature_names_in_", None)
    if feat is None:
        # fallback: if scaler knows n_features_in_ and df already has that many cols, return df
        n = getattr(scaler, "n_features_in_", None)
        if n is None or df.shape[1] == n:
            return df
        # else try to be conservative: keep df as-is
        return df
    feat = list(feat)
    out = df.copy()
    for c in feat:
        if c not in out.columns:
            out[c] = 0.0
    return out[feat]

# ------------------ DEFAULT PATHS ------------------
DEFAULT_JSON = Path("/home/coder/digital_twin/twin/simulation_data/json/block_00000.json")
DEFAULT_FUTURE_GLOB = "/home/coder/digital_twin/twin/simulation_data/json/future_block_*.json"
DEFAULT_CKPT = Path("/home/coder/digital_twin/twin/simulation_data/models/lstm_model_finetuned_559-ws-training.pth")
SEQ_LEN = 50
MAX_HORIZON = 12

# ------------------ DATA PREPARATION ------------------
def _prepare_df_from_list(data_list):
    df = pd.DataFrame(data_list)
    for c in ("glucose_level", "bolus_dose", "meal_carbs", "exercise_intensity", "basis_sleep_binary"):
        if c not in df.columns:
            df[c] = 0.0
    # keep only the base features in the training order
    return df[["glucose_level", "bolus_dose", "meal_carbs", "exercise_intensity", "basis_sleep_binary"]]

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

# --- Residual (decay profile) extension ---
def _make_residual_profile_from_past(df_now, feature, dt_min, peak_min, total_min, magnitude_scale=1.0):
    """Create a decaying residual profile for past events.
    Always returns an array of length MAX_HORIZON (padded with zeros)."""
    prof = np.zeros(MAX_HORIZON, dtype=float)
    idx_arr = df_now[feature].to_numpy().nonzero()[0]
    if len(idx_arr) == 0:
        return prof

    last_idx = idx_arr[-1]
    steps_since = len(df_now) - 1 - last_idx
    if steps_since < 0:
        return prof

    grid = np.arange(0, total_min + dt_min, dt_min)
    w = _tri_kernel(grid, peak_at=peak_min, total=total_min)
    remaining = w[steps_since:]
    if remaining.size == 0:
        return prof
    total = remaining.sum()
    if total > 0:
        remaining = remaining * (df_now[feature].iloc[last_idx] * magnitude_scale / total)
    L = min(len(remaining), MAX_HORIZON)
    prof[:L] = remaining[:L]
    return prof

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
    # transform using a DataFrame aligned to scaler feature order
    df_now_aligned = _align_df_to_scaler(df_now, scaler)
    arr_scaled = scaler.transform(df_now_aligned)
    seq = arr_scaled[-seq_len:].copy()
    base_cov_orig = df_now.values[-1, :].astype(np.float32)
    if hold_inputs == "zero":
        base_cov_orig = np.zeros_like(base_cov_orig)

    carb_profile_feat  = _make_profile(steps, meal_at_step, meal_grams, (carb_peak_min, carb_total_min), dt_min)
    bolus_profile_feat = _make_profile(steps, bolus_at_step, bolus_units, (0.1, dt_min), dt_min)

    # --- Add residual (decay) profiles ---
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

        # build inverse-transform row matching scaler feature layout
        n_feats_scaler = getattr(scaler, "n_features_in_", seq.shape[1])
        tmp = np.zeros((1, n_feats_scaler), dtype=np.float32)
        # index of glucose in scaler's ordering (fallback to 0)
        try:
            g_idx_scaler = list(getattr(scaler, "feature_names_in_", ["glucose_level"])).index("glucose_level")
        except Exception:
            g_idx_scaler = 0
        tmp[:, g_idx_scaler] = y_scaled
        glu_mgdl = scaler.inverse_transform(tmp)[0, g_idx_scaler]

        # physiological effects (including residual decay profiles)
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

        # create a single-row DataFrame with feature_names then align to scaler before transform
        row_df = pd.DataFrame([feat_orig], columns=feature_names)
        row_aligned = _align_df_to_scaler(row_df, scaler)
        feat_scaled = scaler.transform(row_aligned)[0]
        seq = np.vstack([seq[1:], feat_scaled])

    return pd.DataFrame({
        "glucose_pred":   np.array(preds_mgdl, dtype=float),
        "carb_effect":    carb_effect.astype(float),
        "insulin_effect": insulin_effect.astype(float),
        "carb_profile":   carb_profile_feat.astype(float),
        "bolus_profile":  bolus_profile_feat.astype(float),
    })

# --- Erweiterte Digital-Twin-Simulation mit Schlaf- und Bewegungseffekten ---
def simulate_meal_insulin_hybrid_sleep_exercise(
    model, scaler, df_now, feature_names,
    *,
    steps=36, seq_len=36, dt_min=5,
    # ---- Meal & Insulin
    meal_at_step=None, meal_grams=45.0,
    bolus_at_step=None, bolus_units=3.0,
    carb_peak_min=60, carb_total_min=180,
    ins_peak_min=75, ins_total_min=300,
    ISF_mgdl_per_U=50.0, CR_g_per_U=10.0,
    # ---- Exercise
    ex_at_step=None, ex_duration_steps=0, ex_intensity=0.0,
    k_ex_drop_per_hr=40.0,
    ex_ISF_gain=0.30,
    ex_drop_peak_min=30, ex_drop_total_min=180,
    ex_sens_peak_min=90, ex_sens_total_min=360,
    # ---- Sleep
    sleep_at_step=None, sleep_duration_steps=0,
    k_sleep_drop_per_hr=8.0,
    sleep_ISF_gain=0.10,
    sleep_drop_peak_min=120, sleep_drop_total_min=480,
    # ---- Feature roll strategy
    hold_inputs="last",
    ex_schedule=None,
    sleep_schedule=None,
    clip_glucose_to=(40, 400)
):
    """
    Glucose rollout including:
      - Residual profiles from the past for carbs, bolus, exercise, sleep
      - Optional future inputs for carbs, bolus, exercise, sleep
      - Insulin-sensitivity and drop profiles for exercise/sleep
    """
    df_now = df_now.copy()
    for col in feature_names:
        if col not in df_now.columns:
            df_now[col] = 0.0
    df_now = df_now[feature_names]

    F = len(feature_names)
    device = next(model.parameters()).device
    g_idx   = feature_names.index("glucose_level")
    meal_idx  = feature_names.index("meal_carbs")         if "meal_carbs" in feature_names else None
    bolus_idx = feature_names.index("bolus_dose")         if "bolus_dose" in feature_names else None
    ex_idx    = feature_names.index("exercise_intensity") if "exercise_intensity" in feature_names else None
    sleep_idx = feature_names.index("basis_sleep_binary") if "basis_sleep_binary" in feature_names else None

    arr_scaled = scaler.transform(df_now.values)
    seq = arr_scaled[-seq_len:].copy()
    base_cov_orig = df_now.values[-1, :].astype(np.float32)
    if hold_inputs == "zero":
        base_cov_orig = np.zeros_like(base_cov_orig)

    # --- FUTURE MEAL / INSULIN profiles ---
    carb_profile_feat  = _make_profile(steps, meal_at_step, meal_grams,
                                       (carb_peak_min, carb_total_min), dt_min)
    bolus_profile_feat = _make_profile(steps, bolus_at_step, bolus_units,
                                       (0.1, dt_min), dt_min)
    total_rise_mgdl  = (meal_grams / max(CR_g_per_U, 1e-6)) * ISF_mgdl_per_U
    carb_effect      = _make_profile(steps, meal_at_step, total_rise_mgdl,
                                     (carb_peak_min, carb_total_min), dt_min)
    total_fall_mgdl  = bolus_units * ISF_mgdl_per_U
    insulin_effect   = _make_profile(steps, bolus_at_step, total_fall_mgdl,
                                     (ins_peak_min, ins_total_min), dt_min)

    # --- Residual profiles from the past ---
    carb_residual  = _make_residual_profile_from_past(df_now, "meal_carbs", dt_min, carb_peak_min, carb_total_min, 1.0)
    bolus_residual = _make_residual_profile_from_past(df_now, "bolus_dose", dt_min, ins_peak_min, ins_total_min, 1.0)
    ex_residual    = _make_residual_profile_from_past(df_now, "exercise_intensity", dt_min, ex_drop_peak_min, ex_drop_total_min, 1.0)
    sleep_residual = _make_residual_profile_from_past(df_now, "basis_sleep_binary", dt_min, sleep_drop_peak_min, sleep_drop_total_min, 1.0)




    # --- FUTURE EXERCISE / SLEEP schedules ---
    ex_profile_feat = np.zeros(steps)
    sleep_profile_feat = np.zeros(steps)

    ex_drop_profile = np.zeros(steps)
    ex_ISF_profile  = np.zeros(steps)
        
    ex_mask = np.asarray(ex_schedule, float).clip(0,1)
    ex_drop_profile = _make_profile(steps, at_step=np.argmax(ex_mask>0),
                                    magnitude=k_ex_drop_per_hr * (ex_mask.sum()*dt_min/60),
                                    kernel_minutes=(ex_drop_peak_min, ex_drop_total_min),
                                    dt_min=dt_min) * ex_mask
    ex_ISF_profile = _make_profile(steps, at_step=np.argmax(ex_mask>0),
                                    magnitude=ex_ISF_gain,
                                    kernel_minutes=(ex_sens_peak_min, ex_sens_total_min),
                                    dt_min=dt_min) * ex_mask
    ex_drop_profile += ex_profile_feat * k_ex_drop_per_hr * dt_min/60
    ex_ISF_profile  += ex_profile_feat * ex_ISF_gain

    sleep_drop_profile = np.zeros(steps)
    sleep_ISF_profile = np.zeros(steps)
    
    sleep_mask = np.asarray(sleep_schedule, float).clip(0,1)
    sleep_drop_profile = _make_profile(steps, at_step=np.argmax(sleep_mask>0),
                                        magnitude=k_sleep_drop_per_hr * (sleep_mask.sum()*dt_min/60),
                                        kernel_minutes=(sleep_drop_peak_min, sleep_drop_total_min),
                                        dt_min=dt_min) * sleep_mask
    sleep_ISF_profile = _make_profile(steps, at_step=np.argmax(sleep_mask>0),
                                        magnitude=sleep_ISF_gain,
                                        kernel_minutes=(sleep_drop_peak_min, sleep_drop_total_min),
                                        dt_min=dt_min) * sleep_mask
    sleep_drop_profile += sleep_profile_feat * k_sleep_drop_per_hr * dt_min/60
    sleep_ISF_profile  += sleep_profile_feat * sleep_ISF_gain

    ISF_mult = (1.0 + ex_ISF_profile) * (1.0 + sleep_ISF_profile)

    preds_mgdl = []
    ex_feat_track, sleep_feat_track = [], []

    lo, hi = clip_glucose_to if clip_glucose_to else (None, None)

    # Residu
    carb_profile_feat  = carb_profile_feat + carb_residual[:steps]
    bolus_profile_feat = bolus_profile_feat + bolus_residual[:steps]
    ex_profile_feat    = ex_profile_feat + ex_residual[:steps]
    sleep_profile_feat = sleep_profile_feat + sleep_residual[:steps]

    for step in range(steps):
        x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            y_scaled = model(x).cpu().numpy().ravel()[0]
        tmp = np.zeros((1, F), dtype=np.float32)
        tmp[:, g_idx] = y_scaled
        glu_mgdl = scaler.inverse_transform(tmp)[0, g_idx]

        # Physiological effects
        glu_mgdl = (
            glu_mgdl
            + carb_effect[step]  # future meal
            - insulin_effect[step] * ISF_mult[step]  # future bolus modified by sleep/exercise
        )

        # Residuals of past events
        glu_mgdl += carb_residual[step] * ISF_mgdl_per_U / max(CR_g_per_U, 1e-6)
        glu_mgdl -= bolus_residual[step] * ISF_mgdl_per_U

        # Sleep and exercise effects from residuals / drop profiles
        glu_mgdl -= ex_drop_profile[step]
        glu_mgdl -= sleep_drop_profile[step]
        if lo is not None and hi is not None:
            glu_mgdl = float(np.clip(glu_mgdl, lo, hi))

        preds_mgdl.append(glu_mgdl)
        feat_orig = base_cov_orig.copy()
        feat_orig[g_idx] = glu_mgdl
        if meal_idx  is not None:  feat_orig[meal_idx]  = carb_profile_feat[step]
        if bolus_idx is not None: feat_orig[bolus_idx] = bolus_profile_feat[step]
        if ex_idx    is not None: feat_orig[ex_idx]    = ex_profile_feat[step]
        if sleep_idx is not None: feat_orig[sleep_idx] = sleep_profile_feat[step]

        feat_scaled = scaler.transform(feat_orig.reshape(1, -1))[0]
        seq = np.vstack([seq[1:], feat_scaled])

        ex_feat_track.append(feat_orig[ex_idx] if ex_idx is not None else 0.0)
        sleep_feat_track.append(feat_orig[sleep_idx] if sleep_idx is not None else 0.0)

    
    return pd.DataFrame({
        "glucose_pred":      np.array(preds_mgdl, dtype=float),
        "carb_effect":       carb_effect.astype(float),
        "insulin_effect":    (insulin_effect * ISF_mult).astype(float),
        "exercise_add_drop": ex_drop_profile.astype(float),
        "sleep_add_drop":    sleep_drop_profile.astype(float),
        "ex_ISF_mult":       ISF_mult.astype(float),
        "exercise_feat":     np.array(ex_feat_track, dtype=float),
        "sleep_feat":        np.array(sleep_feat_track, dtype=float),
    })



# ------------------ FORECAST ------------------
def forecast(data, ckpt_path: str | Path = None, future_glob: str = DEFAULT_FUTURE_GLOB, future_json: list | None = None):
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

    init_scaled_aligned = _align_df_to_scaler(init_scaled, scaler)
    init_scaled_tensor = torch.tensor(scaler.transform(init_scaled_aligned), dtype=torch.float32)

    # --- FUTURE COVARIATES & Digital-Twin Parameters ---
    future_cov_scaled = None
    meal_at_step = None
    meal_grams = 0
    bolus_at_step = None
    bolus_units = 0
    ex_schedule = None
    sleep_schedule = None

    # prefer explicit future_json (passed as Python list) over glob lookup
    future_data = None
    if future_json is not None:
        # accept a list/tuple of dicts or a JSON string
        if isinstance(future_json, (list, tuple)):
            future_data = list(future_json)
        else:
            try:
                future_data = json.loads(future_json) if isinstance(future_json, str) else None
            except Exception:
                future_data = None

    # fallback: search filesystem using future_glob
    if future_data is None:
        future_candidates = sorted(glob.glob(future_glob))
        if future_candidates:
            try:
                with open(future_candidates[0], "r", encoding="utf-8") as f:
                    future_data = json.load(f)
            except Exception:
                future_data = None

    # if we have future data (from argument or file), prepare covariates and schedules
    if future_data:
        try:
            df_future = _prepare_df_from_list(future_data)
            df_future_aligned = _align_df_to_scaler(df_future, scaler)
            # covariates used by the iterative predictor -> columns bolus, meal, exercise
            future_cov_scaled = scaler.transform(df_future_aligned)[:, 1:4]

            meal_idx_arr = df_future["meal_carbs"].to_numpy().nonzero()[0]
            if len(meal_idx_arr) > 0:
                meal_at_step = int(meal_idx_arr[0])
                meal_grams = float(df_future["meal_carbs"].iloc[meal_at_step])
            bolus_idx_arr = df_future["bolus_dose"].to_numpy().nonzero()[0]
            if len(bolus_idx_arr) > 0:
                bolus_at_step = int(bolus_idx_arr[0])
                bolus_units = float(df_future["bolus_dose"].iloc[bolus_at_step])

            # read sleep/exercise schedules from future block if present
            try:
                df_future_full = pd.DataFrame(future_data)
                if "exercise_intensity" in df_future_full.columns:
                    ex_schedule = pd.to_numeric(df_future_full["exercise_intensity"].fillna(0.0)).to_numpy(dtype=float)
                if "basis_sleep_binary" in df_future_full.columns:
                    sleep_schedule = pd.to_numeric(df_future_full["basis_sleep_binary"].fillna(0.0)).to_numpy(dtype=float)
            except Exception:
                ex_schedule = None
                sleep_schedule = None

        except Exception:
            future_cov_scaled = None

    preds_no_future = _predict_iteratively_local(model, init_scaled_tensor, steps=MAX_HORIZON, scaler=scaler, future_covariates_scaled=None, device=device)
    preds_with_future = _predict_iteratively_local(model, init_scaled_tensor, steps=MAX_HORIZON, scaler=scaler, future_covariates_scaled=future_cov_scaled, device=device)

    # use scaler's feature order when available (matches your retrained columns)
    feature_names = list(getattr(scaler, "feature_names_in_", ["glucose_level", "bolus_dose", "meal_carbs", "exercise_intensity", "basis_sleep_binary"]))
    digital_twin_bolus_carbs = simulate_meal_insulin_hybrid(
        model, scaler, df_block, feature_names,
        steps=MAX_HORIZON, seq_len=seq_len, dt_min=5,
        meal_at_step=meal_at_step, meal_grams=meal_grams,
        bolus_at_step=bolus_at_step, bolus_units=bolus_units,
        ISF_mgdl_per_U=50, CR_g_per_U=10
    )
    # build feature list for sleep/exercise run (include cols if present / needed)
    feat_names_se = feature_names.copy()
    # include columns if present in raw df_block or if schedules were supplied
    if ex_schedule is not None or "exercise_intensity" in pd.DataFrame(data).columns:
        if "exercise_intensity" not in feat_names_se:
            feat_names_se.append("exercise_intensity")
    if sleep_schedule is not None or "basis_sleep_binary" in pd.DataFrame(data).columns:
        if "basis_sleep_binary" not in feat_names_se:
            feat_names_se.append("basis_sleep_binary")

    digital_twin_bolus_carbs_sleep_exercise = simulate_meal_insulin_hybrid_sleep_exercise(
        model, scaler, df_block, feat_names_se,
        steps=MAX_HORIZON, seq_len=seq_len, dt_min=5,
        meal_at_step=meal_at_step, meal_grams=meal_grams,
        bolus_at_step=bolus_at_step, bolus_units=bolus_units,
        ex_schedule=ex_schedule, sleep_schedule=sleep_schedule,
        k_sleep_drop_per_hr=8.0,
        sleep_ISF_gain=0.10,
        sleep_drop_peak_min=120, sleep_drop_total_min=480
    )


    return {
        "model_checkpoint": str(ckpt_path),
        "device": str(device),
        "csv_no_future": [float(x) for x in preds_no_future],
        "csv_with_future": [float(x) for x in preds_with_future],
        "d_t_bolus_carbs": digital_twin_bolus_carbs["glucose_pred"].tolist(),
        "d_t_bolus_carbs_sleep_exercise": digital_twin_bolus_carbs_sleep_exercise["glucose_pred"].tolist()
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
