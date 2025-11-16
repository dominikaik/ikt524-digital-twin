from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import json
import numpy as np
import pandas as pd
import importlib.util
import glob
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Paths
TWIN_JSON_FILE = Path(__file__).resolve().parent / "twin_json.py"
JSON_DIR = Path(__file__).resolve().parent / "simulation_data" / "json"
DEFAULT_JSON = JSON_DIR / "block_00000.json"
print(f"Loading twin_json module from: {TWIN_JSON_FILE}")

def load_json(path: Path):
    """Load JSON file"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_twin_module(path: Path):
    """Load twin_json module dynamically"""
    spec = importlib.util.spec_from_file_location("twin_json", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load twin_json from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def generate_interventions(baseline, scenarios, labels):
    """Generate intervention recommendations"""
    results = []
    base_final = baseline[-1] if baseline else 0
    base_max = max(baseline) if baseline else 0

    for scenario, label in zip(scenarios, labels):
        if not scenario:
            continue
            
        final = scenario[-1]
        maxv = max(scenario)
        delta_peak = maxv - base_max
        delta_final = final - base_final

        if "meal" in label.lower() and "bolus" not in label.lower():
            if delta_peak > 30:
                results.append({
                    "type": "warning",
                    "icon": "🔸",
                    "scenario": label,
                    "message": f"Peak glucose rises by {delta_peak:.1f} mg/dL vs baseline — consider a bolus or lighter meal."
                })
        elif "meal" in label.lower() and "bolus" in label.lower():
            if delta_peak < 0:
                results.append({
                    "type": "success",
                    "icon": "✅",
                    "scenario": label,
                    "message": f"Adequate control — insulin reduces meal spike by {abs(delta_peak):.1f} mg/dL."
                })
            else:
                results.append({
                    "type": "caution",
                    "icon": "⚠️",
                    "scenario": label,
                    "message": f"Partial correction — meal still elevates glucose by {delta_peak:.1f} mg/dL."
                })
        elif "exercise" in label.lower():
            if delta_final < -20:
                results.append({
                    "type": "info",
                    "icon": "🏃",
                    "scenario": label,
                    "message": f"Glucose drops by {abs(delta_final):.1f} mg/dL. Risk of mild hypoglycemia, consider small carb intake before exercise."
                })
            else:
                results.append({
                    "type": "success",
                    "icon": "🏃",
                    "scenario": label,
                    "message": f"Exercise improves glucose stability by {abs(delta_final):.1f} mg/dL."
                })
        elif "sleep" in label.lower():
            if delta_final < -10:
                results.append({
                    "type": "info",
                    "icon": "😴",
                    "scenario": label,
                    "message": f"Nighttime glucose decreases slightly ({abs(delta_final):.1f} mg/dL)."
                })
            else:
                results.append({
                    "type": "success",
                    "icon": "😴",
                    "scenario": label,
                    "message": "Glucose stable during sleep."
                })
    
    return results

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Glucose Forecasting API is running"})

@app.route('/api/files', methods=['GET'])
def list_files():
    """List available data files"""
    try:
        files = sorted([p.name for p in JSON_DIR.glob("block_*.json")])
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """Main forecast endpoint"""
    try:
        # Get request data
        req_data = request.get_json()
        filename = req_data.get('filename', 'block_00000.json')
        custom_inputs = req_data.get('customInputs', {})
        use_future = req_data.get('useFuture', False)
        
        # Load data file
        data_path = JSON_DIR / filename
        if not data_path.exists():
            return jsonify({"error": f"File {filename} not found"}), 404
        
        data = load_json(data_path)
        
        # Extract historical glucose
        historical_glucose = []
        for row in data:
            try:
                val = float(row.get("glucose_level"))
                if val is not None and val > 0:
                    historical_glucose.append(val)
            except:
                pass
        
        # Get last 11 points for chart
        N = 11
        hist_tail = historical_glucose[-N:] if len(historical_glucose) >= N else historical_glucose
        x_hist = list(range(-len(hist_tail) + 1, 1))
        
        # Load twin module and run forecast
        twin_mod = load_twin_module(TWIN_JSON_FILE)
        
        # Handle future data if provided
        future_glob = None
        if use_future or any(custom_inputs.values()):
            # Create temporary future file
            import re
            match = re.search(r"(\d+)", filename)
            num = match.group(1) if match else "00000"
            future_filename = f"future_block_{num}.json"
            future_path = JSON_DIR / future_filename
            
            if future_path.exists():
                future_data = load_json(future_path)
                
                # Apply custom inputs if provided
                if custom_inputs:
                    for i, item in enumerate(future_data):
                        if i == 0:  # Apply to first timestep
                            item["bolus_dose"] = float(custom_inputs.get("bolus", 0))
                            item["meal_carbs"] = float(custom_inputs.get("meal", 0))
                        item["exercise_intensity"] = int(custom_inputs.get("exercise", 0))
                        item["basis_sleep_binary"] = 1 if custom_inputs.get("sleep", False) else 0
                
                # Save modified future data
                temp_future = JSON_DIR / f"temp_future_{num}.json"
                with temp_future.open("w") as f:
                    json.dump(future_data, f)
                future_glob = str(temp_future)
        
        # Run forecast
        if future_glob:
            result = twin_mod.forecast(data, future_glob=future_glob)
        else:
            result = twin_mod.forecast(data)
        
        # Extract predictions
        predictions = {}
        for key, value in result.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                try:
                    predictions[key] = [float(v) for v in value]
                except:
                    pass
        
        # Generate x-axis for predictions
        max_len = max([len(v) for v in predictions.values()]) if predictions else 0
        x_pred = list(range(1, max_len + 1))
        
        # Generate interventions
        baseline = predictions.get("csv_no_future", [])
        scenarios = []
        labels = []
        
        if "d_t_bolus_carbs" in predictions:
            scenarios.append(predictions["d_t_bolus_carbs"])
            labels.append("Meal + Bolus")
        
        if "d_t_bolus_carbs_sleep_exercise" in predictions:
            scenarios.append(predictions["d_t_bolus_carbs_sleep_exercise"])
            labels.append("With Sleep/Exercise")
        
        interventions = generate_interventions(baseline, scenarios, labels)
        
        # Prepare response
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
            "metadata": {
                "filename": filename,
                "device": result.get("device", "unknown"),
                "model": result.get("model_checkpoint", "unknown")
            }
        }
        
        # Cleanup temp file
        if future_glob:
            try:
                Path(future_glob).unlink()
            except:
                pass
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in forecast: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a new data file"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save file
        filename = file.filename
        filepath = JSON_DIR / filename
        file.save(str(filepath))
        
        return jsonify({"message": "File uploaded successfully", "filename": filename})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure directories exist
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)