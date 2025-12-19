# flask_app_matching_streamlit.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import json
import numpy as np
import pandas as pd
import importlib.util
import traceback
import re
import uuid

app = Flask(__name__)
CORS(app)  # Allow requests from your React frontend

# Paths (adjust if needed)
BASE_DIR = Path(__file__).resolve().parent
TWIN_JSON_FILE = BASE_DIR / "twin_json.py"
JSON_DIR = BASE_DIR / "simulation_data" / "json"
DEFAULT_JSON = JSON_DIR / "block_00000.json"

print(f"[Flask] Loading twin_json module from: {TWIN_JSON_FILE}")

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

def _make_serializable(x):
    # replicate Streamlit's _make_serializable behavior
    import numpy as _np
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
    try:
        # numpy scalar
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    return str(x)

def save_return_json(result):
    try:
        out_path = BASE_DIR / "return.json"
        with out_path.open("w", encoding="utf-8") as _f:
            json.dump(_make_serializable(result), _f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def generate_interventions(baseline, scenarios, labels):
    """Same logic as your Flask earlier — kept intact."""
    results = []
    base_final = baseline[-1] if baseline else 0
    base_max = max(baseline) if baseline else 0

    for scenario, label in zip(scenarios, labels):
        if not scenario:
            continue

        final = scenario[-1]
        maxv = max(scenario) if scenario else 0
        delta_peak = maxv - base_max
        delta_final = final - base_final

        if "meal" in label.lower() and "bolus" not in label.lower():
            if delta_peak > 30:
                results.append({
                    "type": "warning",
                    "icon": "🔸",
                    "scenario": label,
                    "message": f"Peak glucose rises by {delta_peak:.1f} mg/dL vs baseline — this meal may spike glucose more than expected. You might consider strategies like reducing portion size, choosing lower-carb alternatives, or adjusting insulin timing/dose based on your clinician’s guidance."
                })
        elif "meal" in label.lower() and "bolus" in label.lower():
            if delta_peak < 0:
                results.append({
                    "type": "success",
                    "icon": "✅",
                    "scenario": label,
                    "message": f"Adequate control — insulin reduces the meal spike by {abs(delta_peak):.1f} mg/dL. This suggests that your bolus strategy is working well. You can continue using similar timing or carb-ratio approaches for this type of meal."
                })
            else:
                results.append({
                    "type": "caution",
                    "icon": "⚠️",
                    "scenario": label,
                    "message": f"Partial correction — glucose still increases by {delta_peak:.1f} mg/dL. Options include reviewing carb counting accuracy, adjusting bolus timing, or discussing ratio adjustments with your care team."
                })
        elif "exercise" in label.lower():
            if delta_final < -20:
                results.append({
                    "type": "info",
                    "icon": "🏃",
                    "scenario": label,
                    "message": f"Glucose drops by {abs(delta_final):.1f} mg/dL. There may be a mild hypoglycemia risk. You could consider a small pre-workout carb snack or adjusting exercise intensity/duration, especially for prolonged sessions."
                })
            else:
                results.append({
                    "type": "success",
                    "icon": "🏃",
                    "scenario": label,
                    "message": f"Exercise improves glucose stability by {abs(delta_final):.1f} mg/dL. This level of activity seems well-balanced — continuing similar workouts may help maintain steady glucose trends."
                })
        elif "sleep" in label.lower():
            if delta_final < -10:
                results.append({
                    "type": "info",
                    "icon": "😴",
                    "scenario": label,
                    "message": f"Nighttime glucose decreases by {abs(delta_final):.1f} mg/dL. Consider checking if evening meals, medications, or activity levels influence nighttime patterns."
                })
            else:
                results.append({
                    "type": "success",
                    "icon": "😴",
                    "scenario": label,
                    "message": "Glucose remains stable during sleep. Your evening routine and basal pattern appear well-matched for nighttime needs."
                })

    return results

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Glucose Forecasting API is running"})

@app.route('/api/files', methods=['GET'])
def list_files():
    try:
        JSON_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted([p.name for p in JSON_DIR.glob("block_*.json")])
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def forecast_endpoint():
    """
    Payload expected:
    {
      "filename": "block_00000.json",
      "customInputs": {"bolus": 0.0, "meal": 0.0, "exercise": 0, "sleep": False},
      "useFuture": true/false
    }
    """
    try:
        req_data = request.get_json() or {}
        filename = req_data.get('filename', DEFAULT_JSON.name)
        custom_inputs = req_data.get('customInputs', {}) or {}
        use_future = bool(req_data.get('useFuture', False))

        # Load historical data
        data_path = JSON_DIR / filename
        if not data_path.exists():
            return jsonify({"error": f"File {filename} not found"}), 404
        data = load_json(data_path)

        # Build history like Streamlit: allow None on parse errors, keep list of floats/None
        hist = []
        for row in data:
            try:
                hist.append(float(row.get("glucose_level", None)))
            except Exception:
                hist.append(None)
        # hist_valid used for tail
        hist_valid = [v for v in hist if v is not None]
        N = 11
        hist_tail = hist_valid[-N:] if hist_valid else []
        x_hist = list(range(-len(hist_tail) + 1, 1))

        # Load twin module
        twin_mod = load_twin_module(TWIN_JSON_FILE)

        # Determine whether to create a future override file (temp)
        future_glob = None
        temp_future_path = None
        last_sent_future = None

        # Compute future filename based on filename numeric suffix (replicates Streamlit)
        match = re.search(r"(\d+)", filename)
        num = match.group(1) if match else "00000"
        future_filename = f"future_block_{num}.json"
        future_path = JSON_DIR / future_filename

        # Determine apply_custom (Streamlit logic: true when any custom input is non-default)
        bolus = float(custom_inputs.get("bolus", 0) or 0)
        meal = float(custom_inputs.get("meal", 0) or 0)
        exercise = int(custom_inputs.get("exercise", 0) or 0)
        sleep = bool(custom_inputs.get("sleep", False))
        apply_custom = (bolus != 0.0) or (meal != 0.0) or (1 if sleep else 0) != 0 or (exercise != 0)

        if (use_future or apply_custom) and future_path.exists():
            # load future template
            future_raw = load_json(future_path)

            # If user chose to "use future actuals" then start from future_raw values, otherwise build from template
            override_list = []
            # If use_future True and no custom inputs -> pass future_raw as-is
            # If apply_custom -> modify entries:
            #   - bolus_dose and meal_carbs only applied to i==0
            #   - basis_sleep_binary and exercise_intensity set to provided values for all rows if apply_custom true, else retain original
            for i, item in enumerate(future_raw):
                new_item = {
                    "timestamp": item.get("timestamp"),
                    "glucose_level": item.get("glucose_level")
                }
                # Bolus/meal: only applied to first future timestep when custom provided
                if apply_custom:
                    new_item["bolus_dose"] = float(bolus) if i == 0 else 0.0
                    new_item["meal_carbs"] = float(meal) if i == 0 else 0.0
                    # Sleep/exercise applied for all steps when apply_custom True (matches Streamlit)
                    new_item["basis_sleep_binary"] = int(1 if sleep else 0)
                    new_item["exercise_intensity"] = int(exercise)
                else:
                    # If not applying custom, preserve values from future_raw if they exist (Streamlit uses values when use_future_actuals is true)
                    new_item["bolus_dose"] = float(item.get("bolus_dose", 0.0))
                    new_item["meal_carbs"] = float(item.get("meal_carbs", 0.0))
                    new_item["basis_sleep_binary"] = int(item.get("basis_sleep_binary", 0))
                    new_item["exercise_intensity"] = float(item.get("exercise_intensity", 0.0))
                override_list.append(new_item)

            # If use_future is True but apply_custom is also True, Streamlit still sends override_list (it prefers last_sent_future)
            # Save override_list as a temp file and also write a copy as send_future.json (for inspection)
            temp_future_name = f"temp_future_{num}_{uuid.uuid4().hex}.json"
            temp_future_path = JSON_DIR / temp_future_name
            with temp_future_path.open("w", encoding="utf-8") as tf:
                json.dump(override_list, tf, ensure_ascii=False, indent=2)

            # Save copy in twin folder (send_future.json) like Streamlit does
            try:
                send_path = BASE_DIR / "send_future.json"
                with send_path.open("w", encoding="utf-8") as sf:
                    json.dump(override_list, sf, ensure_ascii=False, indent=2)
            except Exception:
                pass

            future_glob = str(temp_future_path)
            last_sent_future = override_list

        # Call forecast with or without future_glob (matching Streamlit)
        if future_glob:
            result = twin_mod.forecast(data, future_glob=future_glob)
        else:
            # default call
            # If twin_mod.forecast takes named arg future_glob or different signature, twin_json should support it (Streamlit expects forecast(data, future_glob=str(...)))
            result = twin_mod.forecast(data)

        # Save return.json as Streamlit did (non-blocking)
        try:
            save_return_json(result)
        except Exception:
            pass

        # Build predictions: preserve list/ndarray outputs and convert to plain lists of floats where possible
        predictions = {}
        for key, value in (result.items() if isinstance(result, dict) else []):
            if isinstance(value, (list, np.ndarray)):
                try:
                    predictions[key] = [float(v) for v in value]
                except Exception:
                    # keep as-is stringified if conversion fails
                    try:
                        predictions[key] = [float(x) if x is not None else None for x in value]
                    except Exception:
                        pass
            else:
                # skip other keys (Streamlit serializes everything but chart only uses arrays)
                continue

        # As a fallback, if result is a dict but keys may be nested, we attempt to coerce anything list-like
        if not predictions and isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, (list, np.ndarray)):
                    try:
                        predictions[key] = [float(v) for v in value]
                    except:
                        try:
                            predictions[key] = [float(x) if x is not None else None for x in value]
                        except:
                            pass

        # Determine x_pred based on longest series
        max_len = max(len(series) for series in predictions.values())
        x_pred = list(range(0, max_len))


        # Generate interventions using the same keys as Streamlit expects
        baseline = predictions.get("csv_no_future", []) or []
        scenarios = []
        labels = []
        if "d_t_bolus_carbs" in predictions:
            scenarios.append(predictions["d_t_bolus_carbs"])
            labels.append("Meal + Bolus")
        if "d_t_bolus_carbs_sleep_exercise" in predictions:
            scenarios.append(predictions["d_t_bolus_carbs_sleep_exercise"])
            labels.append("With Sleep/Exercise")
        interventions = generate_interventions(baseline, scenarios, labels)

        # Response metadata: include device & model if present in result (mirrors Streamlit)
        metadata = {
            "filename": filename,
            "device": result.get("device", "unknown") if isinstance(result, dict) else "unknown",
            "model": result.get("model_checkpoint", "unknown") if isinstance(result, dict) else "unknown"
        }

        response = {
            "historical": {
                "x": x_hist,
                "y": hist_tail
            },
            "predictions": {
                "x": x_pred,
                "series": predictions
            },
            "interventions": interventions,
            "metadata": metadata,
            # optionally return last_sent_future so frontend can preview (Streamlit shows table)
            "last_sent_future": last_sent_future
        }

        # cleanup temp future file if any
        if temp_future_path and temp_future_path.exists():
            try:
                temp_future_path.unlink()
            except Exception:
                pass

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        filename = file.filename
        filepath = JSON_DIR / filename
        JSON_DIR.mkdir(parents=True, exist_ok=True)
        file.save(str(filepath))
        return jsonify({"message": "File uploaded successfully", "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
