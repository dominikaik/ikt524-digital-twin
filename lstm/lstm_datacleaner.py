import os
import glob
import pandas as pd
import numpy as np

def clean_csv_file(input_path, output_path):
    """Load a CSV, clean it, and save to a new file."""
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Warning: Failed to read {input_path}: {e}")
        return

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Process timestamp
    if "timestamp" not in df.columns:
        print(f"Warning: File {input_path} skipped (no 'timestamp')")
        return

    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(str).str.strip(),
        errors="coerce",
        infer_datetime_format=True
    )
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Expected columns (adjustable)
    keep = ["timestamp", "bolus_dose", "exercise_intensity",
            "basis_sleep_binary", "glucose_level", "meal_carbs"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        print(f"Warning: File {input_path} skipped - missing columns: {missing}")
        return

    data = df[keep].copy()
    data.set_index("timestamp", inplace=True)

    # Convert to numeric
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # Clean glucose values
    g = data["glucose_level"].copy()
    g[g <= 0] = np.nan
    g = g.interpolate(method="time", limit_direction="both")
    g = g.ffill().bfill()
    g = g.clip(lower=40, upper=400)
    data["glucose_level"] = g

    # Fill other columns
    for c in data.columns:
        if c != "glucose_level":
            data[c] = data[c].fillna(0.0)

    # Summary
    print(f"{os.path.basename(input_path)} cleaned:")
    print(f"   -> NaNs in glucose: {data['glucose_level'].isna().sum()}")
    print(f"   -> glucose min/max: {data['glucose_level'].min():.1f}/{data['glucose_level'].max():.1f}")
    print(f"   -> total rows: {len(data)}")

    # Save cleaned file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.reset_index().to_csv(output_path, index=False)
    print(f"Saved to: {output_path}\n")


def clean_dataset_folder(input_dir, output_dir):
    """Clean all CSV files in the given directory."""
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        print(f"Warning: No CSV files found in {input_dir}")
        return

    print(f"Starting cleaning in: {input_dir}")
    print(f"  Cleaned files will be saved to: {output_dir}\n")

    for f in csv_files:
        filename = os.path.basename(f)
        output_path = os.path.join(output_dir, filename)
        clean_csv_file(f, output_path)


if __name__ == "__main__":
    base_path = "data/ohio/2020/"

    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")

    train_cleaned = os.path.join(base_path, "train_cleaned")
    test_cleaned = os.path.join(base_path, "test_cleaned")

    # Clean training and test folders separately
    if os.path.exists(train_dir):
        clean_dataset_folder(train_dir, train_cleaned)
    else:
        print("Warning: Training folder not found:", train_dir)

    if os.path.exists(test_dir):
        clean_dataset_folder(test_dir, test_cleaned)
    else:
        print("Warning: Test folder not found:", test_dir)

    print("\nCleaning completed!")
