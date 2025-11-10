# app.py - English version
import streamlit as st
import json
from pathlib import Path
import plotly.graph_objects as go
import importlib.util
import sys
import numpy as np
import uuid

# Paths
DEFAULT_JSON = Path("/home/coder/digital_twin/twin/simulation_data/json/block_00042.json")
TWIN_JSON_FILE = Path(__file__).resolve().parent / "twin_json.py"
FUTURE_JSON_FILE = Path(__file__).resolve().parent / "simulation_data" / "json" / "future_block_00042.json"

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
    # Build historical glucose list (only valid numeric glucose_level)
    hist = []
    for row in data:
        try:
            hist.append(float(row.get("glucose_level", None)))
        except Exception:
            hist.append(None)
    hist_valid = [v for v in hist if v is not None]
    if len(hist_valid) == 0:
        st.error("No valid glucose_level values found in input JSON.")
        return

    # choose history length N (use 11 to show -10..0 like CLI)
    N = 11
    hist_tail = hist_valid[-N:]
    n_hist = len(hist_tail)
    x_hist = np.arange(-n_hist + 1, 1)  # e.g. -10..0

    # predictions: discover any sequence-like entries in the result and plot them
    # collect any list/tuple/ndarray in result whose elements are numeric (convertible to float)
    preds_dict = {}
    H = 0
    for k, v in result.items():
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            try:
                vals = [float(x) for x in v]
            except Exception:
                # skip non-numeric sequences
                continue
            preds_dict[k] = vals
            H = max(H, len(vals))
    model_keys = sorted(preds_dict.keys())
    x_pred = np.arange(1, H + 1)

    # Plot with Plotly: history on negative indices, predictions on positive indices
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_hist.tolist(),
        y=hist_tail,
        mode="lines+markers",
        name="History",
        line=dict(color="#FFA500", width=3)  # orange for history
    ))

    # extend prediction traces to x=0 to avoid a gap: prepend the last historical value at x=0
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

    # color / dash cycles for multiple model traces
    colors = ["red", "blue", "green", "magenta", "cyan", "#FF7F0E"]
    dashes = ["dash", "dot", "dashdot", "longdash", "solid"]
    for i, k in enumerate(model_keys):
        # use the key name as the legend label (clean underscores)
        label = str(k).replace("_", " ").strip()
        add_extended_trace(preds_dict.get(k, []), label, colors[i % len(colors)], dashes[i % len(dashes)])

    # overlay future actual glucose values (if provided) starting at x=1...
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

    # vertical line at current time 0
    fig.add_vline(x=0, line=dict(color="gray", dash="dash"), opacity=0.6)

    # Layout: white text on dark background, orange history, white ticks/titles
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

    # Set tick color to white
    fig.update_xaxes(tickfont=dict(color="white"))
    fig.update_yaxes(tickfont=dict(color="white"))

    # render plot once with a unique key to avoid Streamlit duplicate-ID errors
    st.plotly_chart(fig, use_container_width=True, key=f"forecast_plot_{uuid.uuid4()}")

# --- Streamlit App ---
st.title("Glucose Forecast Dashboard")

# JSON upload or default
uploaded_file = st.file_uploader("Upload JSON file", type="json")
if uploaded_file:
    data = json.load(uploaded_file)
else:
    st.info(f"No file uploaded — using default: {DEFAULT_JSON}")
    data = load_json(DEFAULT_JSON)

# load twin_json
try:
    twin_mod = load_twin_module(TWIN_JSON_FILE)
except Exception as e:
    st.error(f"Error loading twin_json.py: {e}")
    st.stop()

if not hasattr(twin_mod, "forecast"):
    st.error("twin_json.py does not contain a function `forecast(data)`")
    st.stop()

# initial forecast (default: no future covariates)
try:
    result = twin_mod.forecast(data)
except Exception as e:
    st.error(f"Error calling forecast(): {e}")
    st.stop()

st.subheader("Forecast Results (recent steps)")

# Controls: option to overlay real future glucose values (disabled by default)
st.markdown("---")
st.write("Optional: overlay real future glucose values from the future block (disabled by default).")
show_future_actuals = st.checkbox("Overlay future actual glucose (from future_block_00042.json)", value=False)

future_actuals = None
if show_future_actuals:
    if not FUTURE_JSON_FILE.exists():
        st.error(f"Future file not found: {FUTURE_JSON_FILE}")
    else:
        try:
            future_data = load_json(FUTURE_JSON_FILE)
            # extract glucose_level sequence from the future block
            future_actuals = [float(item.get("glucose_level")) for item in future_data if "glucose_level" in item]
            st.info(f"Loaded {len(future_actuals)} future glucose values.")
        except Exception as e:
            st.error(f"Failed to load future values: {e}")
            future_actuals = None

# render plot (initial and after toggling checkbox will re-render once with or without future_actuals)
render_plot(result, data, future_actuals=future_actuals)

# (optional) keep the existing Re-run forecast button if you still want to re-run model predictions
if st.button("Re-run forecast (recompute model predictions)"):
    try:
        new_result = twin_mod.forecast(data)
        st.success("Forecast re-run complete.")
        # when re-running, respect the checkbox for overlaying future actuals
        render_plot(new_result, data, future_actuals=future_actuals)
    except Exception as e:
        st.error(f"Error re-running forecast: {e}")
