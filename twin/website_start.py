# app.py - English version
from pathlib import Path
import streamlit as st
import json
import plotly.graph_objects as go
import importlib.util
import numpy as np
import uuid
import pandas as pd
import re
from pathlib import Path as _Path
import numpy as _np

# Paths
DEFAULT_JSON = Path("./simulation_data/json/block_00006.json")
TWIN_JSON_FILE = Path(__file__).resolve().parent / "twin_json.py"
# FUTURE_JSON_FILE will be computed based on the selected data file (see upload block)
FUTURE_JSON_FILE = None

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_twin_module(path: Path):
    spec = importlib.util.spec_from_file_location("twin_json", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load twin_json from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def render_plot(result, data, future_actuals=None):
    hist = []
    for row in data:
        try:
            hist.append(float(row.get("glucose_level", None)))
        except Exception:
            hist.append(None)
    hist_valid = [v for v in hist if v is not None]
    if len(hist_valid) == 0:
        return None

    N = 11
    hist_tail = hist_valid[-N:]
    n_hist = len(hist_tail)
    x_hist = np.arange(-n_hist + 1, 1)

    preds_dict = {}
    H = 0
    for k, v in result.items():
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            try:
                vals = [float(x) for x in v]
            except Exception:
                continue
            preds_dict[k] = vals
            H = max(H, len(vals))
    model_keys = sorted(preds_dict.keys())
    x_pred = np.arange(1, H + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_hist.tolist(),
        y=hist_tail,
        mode="lines+markers",
        name="History",
        line=dict(color="#FFA500", width=3)
    ))

    last_hist_val = hist_tail[-1] if len(hist_tail) > 0 else None
    def add_extended_trace(preds, name, color, dash):
        if len(preds) == 0:
            return
        if last_hist_val is not None:
            x_ext = np.concatenate(([0], x_pred[:len(preds)]))
            y_ext = [last_hist_val] + preds
        else:
            x_ext = x_pred[:len(preds)]
            y_ext = preds
        fig.add_trace(go.Scatter(
            x=x_ext.tolist(),
            y=y_ext,
            mode="lines+markers",
            name=name,
            line=dict(color=color, dash=dash)
        ))

    colors = ["red", "blue", "green", "magenta", "cyan", "#FF7F0E"]
    dashes = ["dash", "dot", "dashdot", "longdash", "solid"]
    for i, k in enumerate(model_keys):
        label = str(k).replace("_", " ").strip()
        add_extended_trace(preds_dict.get(k, []), label, colors[i % len(colors)], dashes[i % len(dashes)])

    if future_actuals:
        try:
            future_vals = [float(v) for v in future_actuals]
            x_future = np.arange(1, len(future_vals) + 1)
            fig.add_trace(go.Scatter(
                x=x_future.tolist(),
                y=future_vals,
                mode="lines+markers",
                name="Future actuals",
                line=dict(color="lime", dash="dashdot"),
                marker=dict(symbol="diamond")
            ))
        except Exception:
            pass

    fig.add_vline(x=0, line=dict(color="gray", dash="dash"), opacity=0.6)

    fig.update_layout(
        title="Glucose: History (-N..0) and Forecast (1..H)",
        xaxis_title="Time steps (history: negative -> 0 current, predictions: 1..H)",
        yaxis_title="Glucose (mg/dL)",
        legend=dict(x=0, y=1, font=dict(color="white")),
        font=dict(color="white"),
        paper_bgcolor="black",
        plot_bgcolor="black",
        xaxis=dict(title=dict(font=dict(color="white"))),
        yaxis=dict(title=dict(font=dict(color="white")))
    )
    fig.update_xaxes(tickfont=dict(color="white"))
    fig.update_yaxes(tickfont=dict(color="white"))

    return fig

def build_future_preview(future_data, apply_custom, custom_bolus, custom_meal, custom_sleep, custom_exercise):
    if not future_data:
        return None
    rows = []
    for i, item in enumerate(future_data):
        r = {}
        r["timestamp"] = item.get("timestamp")
        r["glucose_level"] = item.get("glucose_level")
        r["bolus_dose"] = float(custom_bolus) if (apply_custom and i == 0) else 0.0
        r["meal_carbs"] = float(custom_meal) if (apply_custom and i == 0) else 0.0
        # Sleep/Exercise must be zero unless custom override is enabled.
        if apply_custom:
            r["basis_sleep_binary"] = int(custom_sleep)
            r["exercise_intensity"] = int(custom_exercise)
        else:
            r["basis_sleep_binary"] = 0
            r["exercise_intensity"] = 0
        rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["timestamp", "glucose_level", "bolus_dose", "meal_carbs", "basis_sleep_binary", "exercise_intensity"]
    df = df[[c for c in cols if c in df.columns]]
    return df

def _save_return_json(result):
    """Save forecast result as JSON into the twin folder (return.json)."""
    out_path = _Path(__file__).resolve().parent / "return.json"
    def _make_serializable(x):
        if isinstance(x, dict):
            return {k: _make_serializable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_make_serializable(v) for v in x]
        if isinstance(x, tuple):
            return [_make_serializable(v) for v in x]
        if isinstance(x, _np.ndarray):
            return x.tolist()
        try:
            if isinstance(x, (float, int, str, bool, type(None))):
                return x
        except Exception:
            pass
        # numpy scalars
        try:
            if hasattr(x, "item"):
                return x.item()
        except Exception:
            pass
        return str(x)
    try:
        with out_path.open("w", encoding="utf-8") as _f:
            json.dump(_make_serializable(result), _f, ensure_ascii=False, indent=2)
    except Exception:
        # don't break the app if saving fails
        pass

# --- Streamlit App ---
st.title("Glucose Forecast Dashboard")

# --- Upload/select JSON (single selection; future filename is derived automatically) ---
uploaded_file = st.file_uploader("Upload JSON file (leave empty to use default)", type="json")
selected_data_path = None
if uploaded_file:
    # save uploaded file to /tmp so we can derive filename/number and reuse path
    tmp_dir = Path("/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"uploaded_{uuid.uuid4().hex}_{uploaded_file.name}"
    with tmp_path.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    data = load_json(tmp_path)
    selected_data_path = tmp_path
else:
    st.info(f"No file uploaded — using default: {DEFAULT_JSON.name}")
    data = load_json(DEFAULT_JSON)
    selected_data_path = DEFAULT_JSON

# --- Select JSON file from simulation_data/json (dropdown) ---
json_dir = Path(__file__).resolve().parent / "simulation_data" / "json"
json_dir.mkdir(parents=True, exist_ok=True)
files = sorted([p.name for p in json_dir.glob("block_*.json")])

# ensure default exists in the list (fallback to first file if not)
default_name = "block_00001.json"
if default_name not in files and files:
    # try to pick a file that contains '00001' or just use first
    fallback = next((n for n in files if "00001" in n), files[0])
    default_name = fallback

selected_name = st.selectbox("Select data block", options=files or [DEFAULT_JSON.name], index=(files.index(default_name) if files and default_name in files else 0))

selected_data_path = json_dir / selected_name
data = load_json(selected_data_path)

# derive future file path from selected data filename (match numeric suffix)
match = re.search(r"(\d+)", selected_data_path.name)
if match:
    num = match.group(1)
    FUTURE_JSON_FILE = selected_data_path.parent / f"future_block_{num}.json"
else:
    FUTURE_JSON_FILE = Path(__file__).resolve().parent / "simulation_data" / "json" / "future_block_00042.json"

st.write(f"Using data: {selected_name} — future block: {FUTURE_JSON_FILE.name}")

# --- Load twin module ---
try:
    twin_mod = load_twin_module(TWIN_JSON_FILE)
except Exception as e:
    st.error(f"Error loading twin_json.py: {e}")
    st.stop()

if not hasattr(twin_mod, "forecast"):
    st.error("twin_json.py does not contain a function `forecast(data)`")
    st.stop()

# (Forecast call removed here — it is executed below so UI options are considered)
# --- Forecast computation ---
try:
    result = twin_mod.forecast(data)
    _save_return_json(result)
except Exception as e:
    st.error(f"Error calling forecast(): {e}")
    st.stop()

st.subheader("Forecast Results (recent steps)")

# --- Load future data ---
future_raw = None
if FUTURE_JSON_FILE.exists():
    try:
        future_raw = load_json(FUTURE_JSON_FILE)
    except Exception:
        future_raw = None

# --- Control panel above the plot ---
with st.expander("Controls / Settings", expanded=True):
    # stacked checkboxes (exactly one above the other) with visible note "#tobe fixed"
    show_future_actuals = st.checkbox("Overlay future actual glucose #tobe fixed", value=True)
    use_future_actuals = st.checkbox("Use future block actuals", value=False)
    # custom_sleep_bool = st.checkbox("basis_sleep_binary (on/off)", value=False)
    # convert checkbox bool to integer flag used elsewhere
    

    # Custom covariates inputs (bolus/meal/exercise) — layout below the stacked checkboxes
    col1, col2 = st.columns(2)
    with col1:
        custom_bolus = st.number_input("Bolus dose", value=0.0, step=0.5, format="%.2f")
        custom_meal = st.number_input("Meal carbs", value=0.0, step=10.0, format="%.1f")
    with col2:
        custom_exercise = st.selectbox("exercise_intensity", options=[0,1,2,3], index=0)
        custom_sleep_bool = st.selectbox("basis_sleep_binary", options=[True,False], index=1)
    custom_sleep = 1 if custom_sleep_bool else 0
    # apply_custom is true when the user entered non-default values
    apply_custom = (
        (custom_bolus != 0.0)
        or (custom_meal != 0.0)
        or (custom_sleep != 0)
        or (custom_exercise != 0)
    )

    # Re-run Button
    rerun_clicked = st.button("Re-run forecast (recompute model predictions)")

# --- Compute initial forecast respecting use_future_actuals / custom inputs ---
# this ensures the checkbox state / custom inputs are taken into account on each Streamlit run
result = None
tmp_path_for_initial = None
last_sent_future = None
try:
    # if user wants to use the future block actuals and the file exists -> pass it
    if use_future_actuals and future_raw:
        override_list = []
        for item in future_raw:
            new_item = {
                "timestamp": item.get("timestamp"),
                "glucose_level": item.get("glucose_level"),
                "bolus_dose": float(item.get("bolus_dose", 0.0)),
                "meal_carbs": float(item.get("meal_carbs", 0.0)),
                "basis_sleep_binary": int(item.get("basis_sleep_binary", 0)),
                "exercise_intensity": float(item.get("exercise_intensity", 0.0)),
            }
            override_list.append(new_item)
        tmp_path_for_initial = Path("/tmp") / f"future_override_{uuid.uuid4().hex}.json"
        tmp_path_for_initial.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path_for_initial.open("w", encoding="utf-8") as _f:
            json.dump(override_list, _f, ensure_ascii=False, indent=2)
        # save a copy in the twin folder for inspection
        send_path = Path(__file__).resolve().parent / "send_future.json"
        with send_path.open("w", encoding="utf-8") as _sf:
            json.dump(override_list, _sf, ensure_ascii=False, indent=2)
        last_sent_future = override_list
        result = twin_mod.forecast(data, future_glob=str(tmp_path_for_initial))
    else:
        # if the user provided custom overrides (apply_custom) and we have a future_raw template, build and pass it
        if apply_custom and (future_raw is not None):
            override_list = []
            for i, item in enumerate(future_raw):
                new_item = {
                    "timestamp": item.get("timestamp"),
                    "glucose_level": item.get("glucose_level"),
                    "bolus_dose": float(custom_bolus) if i == 0 else 0.0,
                    "meal_carbs": float(custom_meal) if i == 0 else 0.0,
                    "basis_sleep_binary": int(custom_sleep) if apply_custom else 0,
                    "exercise_intensity": int(custom_exercise) if apply_custom else 0,
                }
                override_list.append(new_item)
            tmp_path_for_initial = Path("/tmp") / f"future_override_{uuid.uuid4().hex}.json"
            tmp_path_for_initial.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path_for_initial.open("w", encoding="utf-8") as _f:
                json.dump(override_list, _f, ensure_ascii=False, indent=2)
            # save a copy in the twin folder for inspection
            send_path = Path(__file__).resolve().parent / "send_future.json"
            with send_path.open("w", encoding="utf-8") as _sf:
                json.dump(override_list, _sf, ensure_ascii=False, indent=2)
            last_sent_future = override_list
            result = twin_mod.forecast(data, future_glob=str(tmp_path_for_initial))
        else:
            # default: call forecast without future_glob
            result = twin_mod.forecast(data)
    _save_return_json(result)
except Exception as e:
    st.error(f"Error calling forecast(): {e}")
    st.stop()
finally:
    if tmp_path_for_initial and tmp_path_for_initial.exists():
        try:
            tmp_path_for_initial.unlink()
        except Exception:
            pass

# --- Placeholders for plot and table ---
chart_container = st.empty()
table_container = st.empty()

# --- Re-run Forecast when button clicked ---
if rerun_clicked:
    try:
        override_list = []
        if use_future_actuals and future_raw:
            # take values directly from future block (use 0 defaults when missing)
            for item in future_raw:
                new_item = {
                    "timestamp": item.get("timestamp"),
                    "glucose_level": item.get("glucose_level"),
                    "bolus_dose": float(item.get("bolus_dose", 0.0)),
                    "meal_carbs": float(item.get("meal_carbs", 0.0)),
                    "basis_sleep_binary": int(item.get("basis_sleep_binary", 0)),
                    "exercise_intensity": float(item.get("exercise_intensity", 0.0)),
                }
                override_list.append(new_item)
        else:
            # build override from UI custom inputs (or zeros)
            for i, item in enumerate(future_raw or []):
                new_item = {}
                new_item["timestamp"] = item.get("timestamp")
                new_item["glucose_level"] = item.get("glucose_level")
                new_item["bolus_dose"] = float(custom_bolus) if (apply_custom and i == 0) else 0.0
                new_item["meal_carbs"] = float(custom_meal) if (apply_custom and i == 0) else 0.0
                if apply_custom:
                    new_item["basis_sleep_binary"] = int(custom_sleep)
                    new_item["exercise_intensity"] = int(custom_exercise)
                else:
                    new_item["basis_sleep_binary"] = 0
                    new_item["exercise_intensity"] = 0
                override_list.append(new_item)

        tmp_path = Path("/tmp") / f"future_override_{uuid.uuid4().hex}.json"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(override_list, f, ensure_ascii=False, indent=2)

        # save a copy in the twin folder for inspection before calling forecast
        send_path = Path(__file__).resolve().parent / "send_future.json"
        with send_path.open("w", encoding="utf-8") as _sf:
            json.dump(override_list, _sf, ensure_ascii=False, indent=2)
        last_sent_future = override_list

        result = twin_mod.forecast(data, future_glob=str(tmp_path))
        _save_return_json(result)

        st.success("Forecast re-run complete.")

        future_vals = [item.get("glucose_level") for item in future_raw] if (future_raw and show_future_actuals) else None
        fig = render_plot(result, data, future_actuals=future_vals)
        if fig is None:
            chart_container.write("Cannot build plot after re-run: no valid historical glucose found.")
        else:
            chart_container.plotly_chart(fig, width="stretch", key=f"forecast_plot_{uuid.uuid4().hex}")

        if override_list:
            table_container.table(pd.DataFrame(override_list))

    finally:
        if tmp_path.exists():
            tmp_path.unlink()

# --- Initial plot render ---
future_vals = [item.get("glucose_level") for item in future_raw] if (future_raw and show_future_actuals) else None
fig = render_plot(result, data, future_actuals=future_vals)
if fig is None:
    chart_container.write("Cannot build plot: no valid historical glucose found.")
else:
    chart_container.plotly_chart(fig, width="stretch", key=f"forecast_plot_{uuid.uuid4().hex}")

# --- Show future preview ---
# prepare preview_df from UI inputs (used when not showing future-block actuals)
preview_df = build_future_preview(future_raw, apply_custom, custom_bolus, custom_meal, custom_sleep, custom_exercise)

# If there is a last_sent_future prefer showing that (it reflects what was actually passed to forecast)
if last_sent_future is not None:
    try:
        table_container.table(pd.DataFrame(last_sent_future)[["timestamp","glucose_level","bolus_dose","meal_carbs","basis_sleep_binary","exercise_intensity"]])
    except Exception:
        table_container.write("No sent future data to display.")
else:
    # If user chose to use future-block actuals show those values in preview (including sleep/exercise)
    if use_future_actuals and future_raw:
        try:
            # normalize future_raw into DataFrame with expected columns
            preview_df2 = pd.DataFrame(future_raw)
            for c in ("timestamp", "glucose_level", "bolus_dose", "meal_carbs", "basis_sleep_binary", "exercise_intensity"):
                if c not in preview_df2.columns:
                    preview_df2[c] = 0
            preview_df2["bolus_dose"] = preview_df2["bolus_dose"].astype(float)
            preview_df2["meal_carbs"] = preview_df2["meal_carbs"].astype(float)
            preview_df2["basis_sleep_binary"] = preview_df2["basis_sleep_binary"].astype(int)
            preview_df2["exercise_intensity"] = preview_df2["exercise_intensity"].astype(float)
            table_container.table(preview_df2[["timestamp","glucose_level","bolus_dose","meal_carbs","basis_sleep_binary","exercise_intensity"]])
        except Exception:
            table_container.write("No future block available to preview.")
    else:
        if preview_df is None or preview_df.empty:
            table_container.write("No future block available to preview.")
        else:
            # show only the columns that actually exist to avoid KeyError
            cols = ["timestamp","glucose_level","bolus_dose","meal_carbs","basis_sleep_binary","exercise_intensity"]
            available = [c for c in cols if c in preview_df.columns]
            table_container.table(preview_df[available])