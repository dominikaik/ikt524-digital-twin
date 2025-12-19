from pathlib import Path
import json
import sys
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_JSON = Path("./simulation_data/json/block_00000.json")
TWIN_JSON_FILE = Path(__file__).resolve().parent / "twin_json.py"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_twin_module(path: Path):
    spec = importlib.util.spec_from_file_location("twin_json", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Kann twin_json nicht laden: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(json_path: str | None = None):
    path = Path(json_path) if json_path else DEFAULT_JSON
    if not path.exists():
        print(f"JSON nicht gefunden: {path}", file=sys.stderr)
        return 2

    data = load_json(path)

    try:
        twin_mod = load_twin_module(TWIN_JSON_FILE)
    except Exception as e:
        print(f"Fehler beim Laden von twin_json.py: {e}", file=sys.stderr)
        return 3

    if not hasattr(twin_mod, "forecast"):
        print("twin_json.py enthält keine Funktion `forecast(data)`", file=sys.stderr)
        return 4

    try:
        result = twin_mod.forecast(data)
        # Plot: last historical points on -N..0 and predictions on 1..H
        try:
            # build historical glucose list from input JSON
            hist = []
            for row in data:
                if isinstance(row, dict) and ("glucose_level" in row):
                    try:
                        hist.append(float(row["glucose_level"]))
                    except Exception:
                        hist.append(None)
                else:
                    hist.append(None)
            # filter None but keep relative ordering; use last contiguous valid values
            hist_valid = [v for v in hist if v is not None]
            if len(hist_valid) == 0:
                raise RuntimeError("Keine gültigen glucose_level Werte in Input JSON.")

            # choose last N points for -10..0 (use 11 points -> -10..0). If not enough, use as many as available.
            N = 11
            hist_tail = hist_valid[-N:]
            n_hist = len(hist_tail)
            x_hist = np.arange(-n_hist + 1, 1)  # e.g. -10..0

            # predictions
            preds_no = result.get("csv_no_future") or []
            preds_with = result.get("csv_with_future") or []
            # ensure numeric arrays
            preds_no = [float(x) for x in preds_no]
            preds_with = [float(x) for x in preds_with]
            H = max(len(preds_no), len(preds_with), 0)
            x_pred = np.arange(1, H + 1)

            # create plot
            plt.figure(figsize=(8,4))
            plt.plot(x_hist, hist_tail, marker='o', label='history')
            if len(preds_no) > 0:
                plt.plot(x_pred[:len(preds_no)], preds_no, marker='o', label='csv_no_future')
            if len(preds_with) > 0:
                plt.plot(x_pred[:len(preds_with)], preds_with, marker='x', label='csv_with_future')
            plt.axvline(0, color='k', linestyle='--', linewidth=0.6)
            plt.xlabel('time steps (history: negative -> 0 current, predictions: 1..H)')
            plt.ylabel('glucose_level')
            plt.legend()
            plt.grid(alpha=0.3)

            out_dir = Path(__file__).resolve().parent / "simulation_data" / "json"
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_path = out_dir / "forecast_plot.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            # print result + plot path
            print(json.dumps({"result": result, "plot": str(plot_path)}, ensure_ascii=False, indent=2))
        except Exception as e_plot:
            # if plotting fails, still print result and the error to stderr
            print(json.dumps({"result": result}, ensure_ascii=False, indent=2))
            print(f"FEHLER beim Erstellen des Plots: {e_plot}", file=sys.stderr)
    except Exception as e:
        print(f"Fehler beim Aufruf von forecast(): {e}", file=sys.stderr)
        return 5

    return 0


if __name__ == "__main__":
    sys.exit(main())