import os
import sys
import torch
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
# use Agg backend for headless environments (ensures plt.savefig works)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# Physiology helpers (triangular kernel + profile maker)
# -------------------------
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

def _make_profile(length_steps, at_step, magnitude, kernel_minutes=(60,180), dt_min=5):
    prof = np.zeros(length_steps, dtype=float)
    if at_step is None or magnitude == 0:
        return prof
    grid = np.arange(length_steps) * dt_min
    w = _tri_kernel(grid - at_step * dt_min, peak_at=kernel_minutes[0], total=kernel_minutes[1])
    if w.sum() > 0:
        w = w * (magnitude / w.sum())
    return w

try:
    from lstm.lstm_transfer_train import predict_iteratively, GlucoseLSTM
except Exception:
    # Fallback: add parent project path if called directly from this folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from lstm.lstm_transfer_train import predict_iteratively, GlucoseLSTM


def _predict_iteratively_local(model, initial_sequence, steps, scaler=None, future_covariates_scaled=None, device=None):
    """
    Iterative 1-step repeated predictor.
    - initial_sequence: torch.tensor(seq_len, n_features) scaled
    - future_covariates_scaled: optional numpy array (steps, n_covariates) already scaled
        where covariates are ONLY the exogenous columns (bolus_dose, meal_carbs, meal_indicator)
        NOT including glucose_change or glucose itself.
    Behavior:
    - compute one-step prediction, append it
    - build next input row using predicted glucose (not real glucose) and:
        * provided future covariates for this step (if given)
        * glucose_change := pred - last_glucose (in scaled space)
    - repeat for `steps` times
    Returns predictions in original glucose units (inverse-transformed with scaler).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval()
    seq = initial_sequence.clone().detach().to(device)  # scaled tensor
    preds_scaled = []

    with torch.no_grad():
        for t in range(steps):
            # single-step forward
            out = model(seq.unsqueeze(0).to(device))  # assume output is scaled glucose
            pred_scaled = out.cpu().item()
            preds_scaled.append(pred_scaled)

            # build next row (all in scaled space)
            n_feats = seq.shape[1]
            new_row = torch.zeros((1, n_feats), device=seq.device, dtype=seq.dtype)
            # predicted glucose as first feature (scaled)
            new_row[0, 0] = torch.tensor(pred_scaled, device=seq.device, dtype=seq.dtype)

            # exogenous covariates: use provided future covariates (if any) else zeros
            if future_covariates_scaled is not None and t < len(future_covariates_scaled):
                cov = future_covariates_scaled[t]
                # expect cov shape matches columns 1:4 (bolus, meal_carbs, meal_indicator) if present
                cov_t = torch.tensor(cov, device=seq.device, dtype=seq.dtype)
                # place cov in positions 1..(1+len(cov)-1)
                new_row[0, 1:1+len(cov_t)] = cov_t

            # glucose_change (last column): compute from predicted and last glucose in window (scaled)
            last_glucose_scaled = seq[-1, 0]
            new_row[0, -1] = new_row[0, 0] - last_glucose_scaled

            # shift window and append new_row
            seq = torch.cat([seq[1:], new_row], dim=0)

    # inverse transform preds to original glucose scale using scaler (if given)
    if scaler is not None:
        # build placeholder matrix with preds in col 0 and zeros for other features, then inverse_transform
        zeros = np.zeros((len(preds_scaled), seq.shape[1]-1))
        stacked = np.concatenate([np.array(preds_scaled).reshape(-1,1), zeros], axis=1)
        preds_orig = scaler.inverse_transform(stacked)[:, 0]
        return preds_orig

    return np.array(preds_scaled)


def evaluate_on_file(model, file_path, scaler, seq_len, device=None, max_horizon=12, out_dir="lstm/models_lstm"):
    """
    Evaluate file: performs iterative multi-start evaluation.
    For each start index:
      - predicts `max_horizon` steps twice:
          a) no_future: future covariates = None (exogenous zeros)
          b) with_future: future covariates taken from file but WITHOUT using future glucose values.
               only columns [bolus_dose, meal_carbs, meal_indicator] are passed as future covariates.
    The glucose_change feature for future rows is computed from predictions (no leakage).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    model.eval()

    df_raw = pd.read_csv(file_path)
    # build features; ensure the ordering matches training (glucose, bolus, meal_carbs, meal_indicator, glucose_change)
    df = df_raw[['glucose_level', 'bolus_dose', 'meal_carbs']].fillna(0)
    df['meal_indicator'] = (df['meal_carbs'] > 0).astype(float)
    df['glucose_change'] = df['glucose_level'].diff().fillna(0)

    used_scaler = scaler
    if used_scaler is None:
        # try baseline scaler in out_dir
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

    preds_no_future = []
    preds_with_future = []
    preds_physio_only = []
    preds_model_plus_physio_list = []   # collect per-start arrays here
    truths_all = []

    # prepare original exogenous arrays for physio-only predictions
    # use the processed df (contains meal_indicator) so the column always exists
    exog_orig = df[['bolus_dose', 'meal_carbs', 'meal_indicator']].fillna(0).values
    # physiology defaults (can be tuned)
    CR_g_per_U = 10.0
    ISF_mgdl_per_U = 50.0
    dt_min = 5

    for idx in tqdm(indices, desc="Iterative eval (per-start index)"):
        init_scaled = torch.tensor(data_scaled[idx:idx+seq_len], dtype=torch.float32)
        future_slice = None
        if (idx + seq_len + max_horizon) <= N:
            future_slice = data_scaled[idx+seq_len: idx+seq_len+max_horizon, 1:4]  # shape (steps, 3)
        else:
            # if remaining shorter than max_horizon, take what's available
            future_slice = data_scaled[idx+seq_len:, 1:4]

        # prediction without future covariates (all exogenous zeros, glucose_change computed from preds)
        preds_nf = _predict_iteratively_local(model, init_scaled, steps=max_horizon, scaler=used_scaler, future_covariates_scaled=None, device=device)
        preds_no_future.append(preds_nf)

        # prediction with future exogenous covariates (but NOT future glucose)
        preds_wf = _predict_iteratively_local(model, init_scaled, steps=max_horizon, scaler=used_scaler, future_covariates_scaled=future_slice, device=device)
        preds_with_future.append(preds_wf)

        # --- physio-only prediction (no future glucose to model) ---
        # compute physiological carb/bolus effect from future exogenous (original units)
        # last observed glucose (original units)
        last_glu = df['glucose_level'].iloc[idx + seq_len - 1]
        # slice future exog (original units) aligned to same indices used above
        future_exog_orig = exog_orig[idx+seq_len: idx+seq_len+max_horizon]
        # build aggregated carb/insulin profile over the horizon
        carb_effect = np.zeros(max_horizon, dtype=float)
        ins_effect  = np.zeros(max_horizon, dtype=float)
        # for each future step where there is a meal/bolus, add a profile
        for s, (bolus_u, meal_g, meal_ind) in enumerate(future_exog_orig):
            if meal_g > 0:
                total_rise = (meal_g / max(CR_g_per_U, 1e-6)) * ISF_mgdl_per_U
                carb_effect += _make_profile(max_horizon, at_step=s, magnitude=total_rise, kernel_minutes=(60,180), dt_min=dt_min)
            if bolus_u > 0:
                total_fall = bolus_u * ISF_mgdl_per_U
                ins_effect += _make_profile(max_horizon, at_step=s, magnitude=total_fall, kernel_minutes=(75,300), dt_min=dt_min)
        # cumulative effect up to each horizon step
        net_step_effect = carb_effect - ins_effect
        preds_physio = last_glu + np.cumsum(net_step_effect)
        # ensure length max_horizon
        if len(preds_physio) < max_horizon:
            preds_physio = np.pad(preds_physio, (0, max_horizon - len(preds_physio)), constant_values=(preds_physio[-1] if len(preds_physio)>0 else last_glu))
        preds_physio_only.append(preds_physio)

        # --- model + physio: model gets NO future exog, physio effects are computed separately and ADDED ---
        # preds_nf are model outputs in original glucose units; add cumulative net effect
        preds_model_plus = preds_nf.copy()
        # if preds_model_plus shorter (edge cases) pad/truncate to max_horizon
        if len(preds_model_plus) < max_horizon:
            preds_model_plus = np.pad(preds_model_plus, (0, max_horizon - len(preds_model_plus)), mode='edge')
        # add cumulative physiologic effect on top of model predictions
        per_start_model_plus_physio = preds_model_plus + np.cumsum(net_step_effect)[:len(preds_model_plus)]
        per_start_model_plus_physio = per_start_model_plus_physio[:max_horizon]
        per_start_model_plus_physio = np.array(per_start_model_plus_physio, dtype=float)
        per_start_model_plus_physio = np.pad(per_start_model_plus_physio, (0, max_horizon - len(per_start_model_plus_physio)), constant_values=per_start_model_plus_physio[-1] if len(per_start_model_plus_physio)>0 else last_glu)
        preds_model_plus_physio_list.append(per_start_model_plus_physio)

        # truths (original glucose values)
        truths_row = []
        for h in range(1, max_horizon + 1):
            truth_row_scaled = data_scaled[idx + seq_len + h - 1]
            truth_orig = used_scaler.inverse_transform(truth_row_scaled.reshape(1, -1))[0, 0]
            truths_row.append(truth_orig)
        truths_all.append(truths_row)

    preds_no_future = np.array(preds_no_future)
    preds_with_future = np.array(preds_with_future)
    preds_model_plus_physio = np.array(preds_model_plus_physio_list)
    truths_all = np.array(truths_all)
    num = preds_no_future.shape[0]

    # compute metrics for selected horizons for both variants
    metrics = {"no_future": {}, "with_future": {}}
    metrics["physio_only"] = {}
    metrics["model_plus_physio"] = {}
    for variant, arr in (("no_future", preds_no_future), ("with_future", preds_with_future)):
        for h in (1, 3, 12):
            if h <= max_horizon:
                preds_h = arr[:, h-1]
                truths_h = truths_all[:, h-1]
                mse_h = mean_squared_error(truths_h, preds_h)
                rmse_h = float(np.sqrt(mse_h))
                mae_h = mean_absolute_error(truths_h, preds_h)
                metrics[variant][h] = {"rmse": rmse_h, "mae": mae_h}
                print(f"[{variant}] Iterative {h}-step -> RMSE: {rmse_h:.2f} | MAE: {mae_h:.2f}")

    # physio-only metrics
    if len(preds_physio_only) > 0:
        arr = np.array(preds_physio_only)
        for h in (1, 3, 12):
            if h <= max_horizon:
                preds_h = arr[:, h-1]
                truths_h = truths_all[:, h-1]
                mse_h = mean_squared_error(truths_h, preds_h)
                rmse_h = float(np.sqrt(mse_h))
                mae_h = mean_absolute_error(truths_h, preds_h)
                metrics["physio_only"][h] = {"rmse": rmse_h, "mae": mae_h}
                print(f"[physio_only] Iterative {h}-step -> RMSE: {rmse_h:.2f} | MAE: {mae_h:.2f}")

    # model + physio metrics
    if preds_model_plus_physio.size > 0:
         for h in (1, 3, 12):
             if h <= max_horizon:
                 preds_h = preds_model_plus_physio[:, h-1]
                 truths_h = truths_all[:, h-1]
                 mse_h = mean_squared_error(truths_h, preds_h)
                 rmse_h = float(np.sqrt(mse_h))
                 mae_h = mean_absolute_error(truths_h, preds_h)
                 metrics["model_plus_physio"][h] = {"rmse": rmse_h, "mae": mae_h}
                 print(f"[model_plus_physio] Iterative {h}-step -> RMSE: {rmse_h:.2f} | MAE: {mae_h:.2f}")

    # Prepare names/dirs for plots and CSVs BEFORE plotting/saving
    base_name = os.path.basename(file_path).replace('.csv', '')
    model_tag = os.path.basename(os.path.normpath(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Comparison plot: mean trajectories + RMSE per horizon ----------
    out_plot = None
    try:
        variants = {}
        if preds_no_future.size > 0:
            variants['no_future'] = preds_no_future
        if preds_with_future.size > 0:
            variants['with_future'] = preds_with_future
        if len(preds_physio_only) > 0:
            variants['physio_only'] = np.array(preds_physio_only)
        if preds_model_plus_physio.size > 0:
            variants['model_plus_physio'] = preds_model_plus_physio

        if variants:
            x = np.arange(1, max_horizon + 1)
            mean_truth = truths_all.mean(axis=0)

            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax = axes[0]
            ax.plot(x, mean_truth, color='k', lw=2, label='True (mean)')
            for name, arr in variants.items():
                mean_pred = np.mean(arr, axis=0)
                ax.plot(x, mean_pred, lw=1.6, label=name)
            ax.set_ylabel("Glucose (mg/dL)")
            ax.set_title(f"Mean multi-step preds (file={os.path.basename(file_path)})")
            ax.grid(alpha=0.25)
            ax.legend()

            # RMSE per horizon
            ax2 = axes[1]
            for name, arr in variants.items():
                rmses = []
                for h in range(arr.shape[1]):
                    rmses.append(np.sqrt(mean_squared_error(truths_all[:, h], arr[:, h])))
                ax2.plot(x, rmses, marker='o', label=name)
            ax2.set_xlabel("Horizon step")
            ax2.set_ylabel("RMSE (mg/dL)")
            ax2.grid(alpha=0.25)
            ax2.legend()

            # save both a run-specific eval plot and a per-model-type comparison plot
            os.makedirs(out_dir, exist_ok=True)
            out_plot = os.path.join(out_dir, f"eval_{base_name}_{model_tag}_comparison.png")
            plt.tight_layout()
            plt.savefig(out_plot, dpi=200)
            print("Saved comparison plot (per-file):", out_plot)

            # also save into a run-level comparison_metrics folder named by model type
            # place comparison plots next to the per-model folders (sibling of base_line/fine_tuned/single)
            comp_dir = os.path.join(os.path.dirname(out_dir), "comparison_metrics")
            os.makedirs(comp_dir, exist_ok=True)
            comp_path = os.path.join(comp_dir, f"{model_tag}_comparison_mae.png")
            plt.savefig(comp_path, dpi=200)
            plt.close(fig)
            print("Saved comparison plot (by modeltype):", comp_path)
    except Exception as e:
        print("Warning: failed to produce comparison plot:", e)
    # end plotting try

    # Save CSVs for both variants
    def _save_variant(out_dir_variant, preds_array, suffix):
        out_dict = {"start_index": np.arange(num)}
        for step in range(1, max_horizon + 1):
            out_dict[f"y_true_{step}"] = truths_all[:, step-1]
        for step in range(1, max_horizon + 1):
            out_dict[f"pred_{step}"] = preds_array[:, step-1]
        out_path = os.path.join(out_dir_variant, f"eval_{base_name}_{model_tag}_{suffix}_all{max_horizon}.csv")
        pd.DataFrame(out_dict).to_csv(out_path, index=False)
        print(f"Saved full-horizon CSV with real values: {out_path}")
        return out_path

    out_path_no = _save_variant(out_dir, preds_no_future, "no_future")
    out_path_with = _save_variant(out_dir, preds_with_future, "with_future")
    if preds_physio_only:
        out_path_physio = _save_variant(out_dir, np.array(preds_physio_only), "physio_only")
    else:
        out_path_physio = None
    if preds_model_plus_physio.size > 0:
        out_path_model_physio = _save_variant(out_dir, preds_model_plus_physio, "model_plus_physio")
    else:
        out_path_model_physio = None

    # print confirmation about comparison plot (if any)
    if out_plot is not None and os.path.exists(out_plot):
        print("Comparison plot available at:", out_plot)
    else:
        print("No comparison plot produced for this run.")

    return {
        "metrics": metrics,
        "csv_no_future": out_path_no,
        "csv_with_future": out_path_with,
        "csv_physio_only": out_path_physio,
        "csv_model_plus_physio": out_path_model_physio,
        "comparison_plot": out_plot,
    }


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