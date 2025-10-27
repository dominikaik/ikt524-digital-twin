import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def _metrics_from_csv(fp, n_horizons=12):
    df = pd.read_csv(fp)
    results = []
    for i in range(1, n_horizons + 1):
        y_col = f"y_true_{i}"
        p_col = f"pred_{i}"
        if y_col not in df.columns or p_col not in df.columns:
            results.append({"horizon": i, "n": 0, "MAE": np.nan, "RMSE": np.nan})
            continue
        sub = df[[y_col, p_col]].dropna()
        if sub.empty:
            results.append({"horizon": i, "n": 0, "MAE": np.nan, "RMSE": np.nan})
            continue
        y_true = sub[y_col].values
        y_pred = sub[p_col].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        results.append({"horizon": i, "n": len(sub), "MAE": mae, "RMSE": rmse})
    metrics_df = pd.DataFrame(results).set_index("horizon")
    metrics_df["MAE"] = metrics_df["MAE"].round(4)
    metrics_df["RMSE"] = metrics_df["RMSE"].round(4)
    return metrics_df


def results(file_paths, n_horizons=12, out_path=None):
    """
    file_paths: list of eval CSVs (preferably [baseline, fine_tuned, single])
    Produces comparison.csv with columns:
      horizon, baseline_mae, baseline_rmse, finetune_mae, finetune_rmse, single_mae, single_rmse
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    file_paths = [p for p in file_paths if p and os.path.exists(p)]
    if not file_paths:
        raise FileNotFoundError("No existing eval CSV files provided.")

    # name mapping: if exactly 3 files, use baseline/finetuned/single
    if len(file_paths) == 3:
        names = ["baseline", "finetune", "single"]
    else:
        names = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]

    per_metrics = {}
    for name, fp in zip(names, file_paths):
        per_metrics[name] = _metrics_from_csv(fp, n_horizons=n_horizons)

    horizons = range(1, n_horizons + 1)
    comp_df = pd.DataFrame(index=horizons)

    # for each model name add mae then rmse columns (lowercase names as requested)
    for name in names:
        dfm = per_metrics.get(name, pd.DataFrame())
        mae_vals = [dfm.loc[h, "MAE"] if h in dfm.index else np.nan for h in horizons]
        rmse_vals = [dfm.loc[h, "RMSE"] if h in dfm.index else np.nan for h in horizons]
        comp_df[f"{name}_mae"] = mae_vals
        comp_df[f"{name}_rmse"] = rmse_vals

    comp_df.index.name = "horizon"

    # output folder
    if out_path is None:
        base_dir = os.path.dirname(file_paths[0])
        out_path = os.path.join(base_dir, "comparison_metrics")
    os.makedirs(out_path, exist_ok=True)

    comp_csv = os.path.join(out_path, "comparison.csv")
    comp_df.to_csv(comp_csv, index=True)
    print(f"Saved comparison CSV: {comp_csv}")

    # Plotting: RMSE and MAE vs horizon for each model
    try:
        # RMSE plot
        rmse_cols = [c for c in comp_df.columns if c.endswith("_rmse")]
        plt.figure(figsize=(9, 5))
        for c in rmse_cols:
            label = c.replace("_rmse", "")
            plt.plot(comp_df.index, comp_df[c], marker="o", label=label)
        plt.xlabel("horizon")
        plt.ylabel("RMSE")
        plt.title("RMSE per Horizon")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, "comparison_rmse.png"))
        plt.close()

        # MAE plot
        mae_cols = [c for c in comp_df.columns if c.endswith("_mae")]
        plt.figure(figsize=(9, 5))
        for c in mae_cols:
            label = c.replace("_mae", "")
            plt.plot(comp_df.index, comp_df[c], marker="o", label=label)
        plt.xlabel("horizon")
        plt.ylabel("MAE")
        plt.title("MAE per Horizon")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, "comparison_mae.png"))
        plt.close()
        print(f"Saved comparison plots in: {out_path}")
    except Exception as e:
        print("Plotting failed:", e)

    # also save per-file metrics for debugging
    for name, dfm in per_metrics.items():
        dfm.to_csv(os.path.join(out_path, f"{name}_metrics_per_horizon.csv"))

    return {"per_metrics": per_metrics, "comparison": comp_df}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resulttest.py eval_baseline.csv eval_finetuned.csv eval_single.csv")
        sys.exit(1)
    file_args = sys.argv[1:]
    out = results(file_args)
    print(out["comparison"].round(4))