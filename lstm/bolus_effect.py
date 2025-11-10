import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from lstm import evaluate_on_file as eval_mod
def build_features(df_raw):
    df = df_raw[['glucose_level', 'bolus_dose', 'meal_carbs']].fillna(0)
    df['meal_indicator'] = (df['meal_carbs'] > 0).astype(float)
    df['glucose_change'] = df['glucose_level'].diff().fillna(0)
    return df

def run_for_events(ckpt, input_csv, seq_len=50, max_horizon=12, out_dir="lstm/event_effects"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, scaler = eval_mod._load_checkpoint_model(ckpt, device=device)
    df_raw = pd.read_csv(input_csv)
    df = build_features(df_raw)
    data_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(data_scaled, columns=df.columns)

    bolus_idxs = list(df.index[df['bolus_dose'] > 0])
    carbs_idxs = list(df.index[df['meal_carbs'] > 0])
    all_event_idxs = sorted(set(bolus_idxs + carbs_idxs))
    print(f"Found {len(bolus_idxs)} bolus and {len(carbs_idxs)} carbs events in {input_csv}")

    results = []
    for ev_idx in all_event_idxs:
        if ev_idx < seq_len - 1:
            print(f"Skipping event at index {ev_idx} (not enough history)")
            continue
        start = ev_idx - seq_len + 1
        init_scaled = df_scaled.iloc[start:ev_idx+1].copy()

        scenarios = {}
        scenarios['actual'] = init_scaled.copy()

        s = init_scaled.copy()
        # safe assignment without chained indexing
        s.loc[s.index[-1], 'bolus_dose'] = 0.0
        scenarios['no_bolus'] = s

        s2 = init_scaled.copy()
        s2.loc[s2.index[-1], 'meal_carbs'] = 0.0
        s2.loc[s2.index[-1], 'meal_indicator'] = 0.0
        scenarios['no_carbs'] = s2

        s3 = init_scaled.copy()
        s3.loc[s3.index[-1], 'bolus_dose'] = 0.0
        s3.loc[s3.index[-1], 'meal_carbs'] = 0.0
        s3.loc[s3.index[-1], 'meal_indicator'] = 0.0
        scenarios['no_bolus_no_carbs'] = s3

        event_preds = {}
        for name, seq_scaled in scenarios.items():
            init_t = torch.tensor(seq_scaled.values, dtype=torch.float32)
            preds = eval_mod._predict_iteratively_local(model, init_t, steps=max_horizon, scaler=scaler, future_covariates_scaled=None, device=device)
            event_preds[name] = preds

        horizons = np.arange(1, max_horizon+1)
        df_event = pd.DataFrame({'horizon': horizons})
        for name, preds in event_preds.items():
            df_event[f"{name}_pred"] = preds

        true_vals = []
        for h in horizons:
            idx_true = ev_idx + h
            if idx_true < len(df_raw):
                true_vals.append(df_raw.iloc[idx_true]['glucose_level'])
            else:
                true_vals.append(np.nan)
        df_event['y_true'] = true_vals

        df_event["diff_actual_minus_no_bolus"] = df_event["actual_pred"] - df_event["no_bolus_pred"]
        df_event["diff_actual_minus_no_carbs"] = df_event["actual_pred"] - df_event["no_carbs_pred"]
        df_event["diff_actual_minus_no_bolus_no_carbs"] = df_event["actual_pred"] - df_event["no_bolus_no_carbs_pred"]

        timestamp = df_raw.loc[ev_idx, 'timestamp'] if 'timestamp' in df_raw.columns else None
        out_filename = os.path.join(out_dir, f"event_idx{ev_idx}.csv")
        df_event.to_csv(out_filename, index=False)
        print(f"Saved event {ev_idx} results -> {out_filename}")

        # --- Combined plot: last 10 observed points + all predicted steps for each scenario + ground truth ---
        try:
            hist_len = 10
            # get last hist_len observed glucose values ending at ev_idx
            start_hist = max(0, ev_idx - hist_len + 1)
            y_hist = df_raw['glucose_level'].iloc[start_hist:ev_idx+1].values
            hist_len_actual = len(y_hist)

            # x axes: history from -hist_len_actual+1 .. 0, predictions 1..max_horizon
            x_hist = np.arange(-hist_len_actual + 1, 1)  # e.g. -9..0
            x_pred = np.arange(1, max_horizon + 1)

            plt.figure(figsize=(10,5))
            ax = plt.gca()

            # plot history (solid black, stärker)
            ax.plot(x_hist, y_hist, marker='o', color='k', linewidth=2, label='history (last {})'.format(hist_len_actual))

            # plot each scenario only for prediction horizon (connect last observed -> first pred)
            colors = {'actual':'C0','no_bolus':'C1','no_carbs':'C2','no_bolus_no_carbs':'C3'}
            last_obs = y_hist[-1] if hist_len_actual > 0 else np.nan
            for name, preds in event_preds.items():
                # connector from last observed point (t=0) to first prediction (t=1)
                x_conn = np.concatenate([[0], x_pred])
                y_conn = np.concatenate([[last_obs], preds])
                ax.plot(x_conn, y_conn, marker='o', linestyle='-', color=colors.get(name,'C7'), label=f'{name} pred')
                # emphasize prediction points
                ax.plot(x_pred, preds, linestyle='', marker='o', color=colors.get(name,'C7'))

            # plot ground-truth future values if available (dashed) and connect to last observed point
            y_true_future = df_event['y_true'].values
            if np.any(~np.isnan(y_true_future)):
                y_true_hist = np.full(hist_len_actual, np.nan)
                if hist_len_actual > 0:
                    y_true_hist[-1] = y_hist[-1]
                y_true_comb = np.concatenate([y_true_hist, y_true_future])
                x_all = np.concatenate([x_hist, x_pred])
                ax.plot(x_all, y_true_comb, marker='x', linestyle='--', color='C4', label='y_true (future)')

            ax.axvline(0, color='gray', linestyle=':', linewidth=1)  # event moment
            ax.set_xlabel('relative step (0 = event last observed point)')
            ax.set_ylabel('glucose (mg/dL)')
            ax.set_title(f"Event idx {ev_idx}" + (f" ts={timestamp}" if timestamp is not None else ""))
            ax.grid(True)
            ax.legend(loc='upper left')

            # secondary axis: bolus/carbs bars aligned to prediction steps (x_pred)
            ax2 = ax.twinx()
            bolus_vals = []
            carbs_vals = []
            for h in x_pred:
                idx_true = ev_idx + int(h)
                if 0 <= idx_true < len(df_raw):
                    bolus_vals.append(df_raw.iloc[idx_true].get('bolus_dose', 0.0))
                    carbs_vals.append(df_raw.iloc[idx_true].get('meal_carbs', 0.0))
                else:
                    bolus_vals.append(0.0)
                    carbs_vals.append(0.0)

            width = 0.35
            ax2.bar(x_pred - width/2, bolus_vals, width=width, alpha=0.6, color='C3', label='bolus (U)')
            ax2.bar(x_pred + width/2, carbs_vals, width=width, alpha=0.4, color='C5', label='carbs (g)')
            ax2.set_ylabel('bolus (U) / carbs (g)')
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(handles2, labels2, loc='upper right')

            plt.tight_layout()
            plot_path = os.path.join(out_dir, f"event_idx{ev_idx}_combined.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved combined event plot -> {plot_path}")
        except Exception as e:
            print("Failed to create combined plot for event", ev_idx, ":", e)

        results.append((ev_idx, timestamp, out_filename))

    manifest = pd.DataFrame([{'event_index': r[0], 'timestamp': r[1], 'csv': r[2]} for r in results])
    manifest.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    print("Done. Manifest saved.")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute effect of bolus/carbs at bolus or carbs events")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input_csv", required=True, help="CSV to scan for bolus and carbs events")
    parser.add_argument("--seq_len", type=int, default=50, help="Length of input sequence history")
    parser.add_argument("--max_horizon", type=int, default=12, help="Number of prediction steps")
    parser.add_argument("--out_dir", type=str, default="lstm/event_effects", help="Output directory for results")
    args = parser.parse_args()

    run_for_events(
        ckpt=args.ckpt,
        input_csv=args.input_csv,
        seq_len=args.seq_len,
        max_horizon=args.max_horizon,
        out_dir=args.out_dir
    )