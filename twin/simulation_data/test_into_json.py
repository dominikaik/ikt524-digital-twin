#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import json
import sys

try:
    import pandas as pd
except Exception as e:
    print("Fehler: pandas wird benötigt. Installiere mit: pip install pandas", file=sys.stderr)
    raise

# Erstellt Blöcke der Größe 50 aus der Test-CSV und speichert gültige Blöcke (ohne 0 in glucose) als JSON.
# Zusätzlich wird für jeden gültigen Block eine Datei future_block_XXXXX.json mit den nächsten 12 Werten angelegt.
# Speicherort: digital_twin/twin/simulation_data/json (relativ zum Skript)

INPUT_CSV = Path("/home/coder/digital_twin/data/ohio/2018/test_cleaned/559-ws-testing.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "json"
BLOCK_SIZE = 50
FUTURE_HORIZON = 12

def find_glucose_column(df):
    # Suche Spalte die "glucose" im Namen enthält (case-insensitive)
    for col in df.columns:
        if "glucose" in col.lower():
            return col
    return None

def main():
    if not INPUT_CSV.exists():
        print(f"Eingabedatei nicht gefunden: {INPUT_CSV}", file=sys.stderr)
        return 1

    df = pd.read_csv(INPUT_CSV)
    glucose_col = find_glucose_column(df)
    if glucose_col is None:
        print("Keine Spalte mit 'glucose' im Namen gefunden.", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(df)
    written = 0
    block_index = 0

    for start in range(0, total, BLOCK_SIZE):
        block = df.iloc[start:start + BLOCK_SIZE]
        if len(block) < BLOCK_SIZE:
            # letzten unvollständigen Block weglassen
            break

        # Konvertiere und prüfe: keine NaN und kein Wert == 0
        series = pd.to_numeric(block[glucose_col], errors="coerce")
        if series.isna().any():
            # ungültig wegen NaN
            block_index += 1
            continue
        if (series == 0).any():
            # enthält 0, überspringen
            block_index += 1
            continue

        # Gültiger Block -> als JSON speichern
        out_path = OUTPUT_DIR / f"block_{block_index:05d}.json"
        records = block.to_dict(orient="records")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        written += 1

        # Zukünftigen Block erstellen, wenn genügend Daten vorhanden sind
        if start + BLOCK_SIZE + FUTURE_HORIZON <= total:
            future_block = df.iloc[start + BLOCK_SIZE:start + BLOCK_SIZE + FUTURE_HORIZON]
            future_out_path = OUTPUT_DIR / f"future_block_{block_index:05d}.json"
            future_records = future_block.to_dict(orient="records")
            with future_out_path.open("w", encoding="utf-8") as f:
                json.dump(future_records, f, ensure_ascii=False, indent=2)

        block_index += 1

    print(f"Fertig. Insgesamt Blöcke geschrieben: {written} in {OUTPUT_DIR}")
    return 0

if __name__ == "__main__":
    sys.exit(main())