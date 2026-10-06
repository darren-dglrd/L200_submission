#!/usr/bin/env python3
"""Standalone 50,000-Record Semiconductor Wafer Dataset Simulator (`lot_delta_50k`).

Generates 50,000 wafer records x 100 process parameters = 5,000,000 parameter cells
for stress-testing and limit-testing the Process Yield Hypothesis Workbench:
  1. `data/raw_wafers_lot_delta_50k.csv` (confidential raw physical units + RECIPE_ID)
  2. `data/obfuscated_wafers_lot_delta_50k.csv` (redacted RECIPE_ID, STEP_A..STEP_E, [0, 1] normalized)

Planted Causal Physics in Dataset 4 (`lot_delta_50k` - STEP_D Sputter Deposition Excursion):
  - #1 Single-Parameter Driver: `STEP_D_PRESSURE_01` (`SPUTTER_DEPOSITION_PRESSURE_01`uparrow)
  - #1 Two-Way Interaction: `STEP_D_RF_POWER_01` (High) x `STEP_D_TEMPERATURE_01` (Low)
  - Secondary Drivers: `STEP_E_FLOW_RATE_01` (CMP slurry flow) & `STEP_D_DURATION_01`
"""

import argparse
import csv
from pathlib import Path
import random
import time
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

STEP_OBFUSCATION_MAP: Dict[str, str] = {
    "UNDERFILL_LAMINATION": "STEP_A",
    "PLASMA_ETCH": "STEP_B",
    "THERMAL_CURE": "STEP_C",
    "SPUTTER_DEPOSITION": "STEP_D",
    "CHEMICAL_CMP": "STEP_E",
}

PARAM_CATEGORIES: List[str] = [
    "TEMPERATURE",
    "PRESSURE",
    "RF_POWER",
    "FLOW_RATE",
    "DURATION",
]

RAW_RANGES: Dict[str, tuple] = {
    "TEMPERATURE": (140.0, 260.0),   # Celsius
    "PRESSURE": (1.5, 9.5),          # Bar / Torr
    "RF_POWER": (250.0, 950.0),      # Watts
    "FLOW_RATE": (20.0, 180.0),      # sccm
    "DURATION": (15.0, 120.0),       # seconds
}

CONFIDENTIAL_RECIPES = [
    "RECIPE-8841-UF-PROPRIETARY",
    "RECIPE-9920-PE-CONFIDENTIAL",
    "RECIPE-7712-TC-INTERNAL",
    "RECIPE-6604-SD-HIGHVAC",
]


def get_raw_parameter_names() -> List[str]:
    cols: List[str] = []
    for raw_step in STEP_OBFUSCATION_MAP.keys():
        for cat in PARAM_CATEGORIES:
            for idx in range(1, 5):
                cols.append(f"{raw_step}_{cat}_{idx:02d}")
    return cols


def get_obfuscated_parameter_names() -> List[str]:
    cols: List[str] = []
    for obf_step in STEP_OBFUSCATION_MAP.values():
        for cat in PARAM_CATEGORIES:
            for idx in range(1, 5):
                cols.append(f"{obf_step}_{cat}_{idx:02d}")
    return cols


def simulate_50k_wafer_dataset(
    num_wafers: int = 50000,
    seed: int = 512,
    data_dir: Path = DATA_DIR,
) -> Dict[str, Any]:
    """Generate `num_wafers` (default 50,000) x 100 parameters = 5,000,000 cells on disk."""
    t0 = time.perf_counter()
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    raw_cols = get_raw_parameter_names()
    obf_cols = get_obfuscated_parameter_names()

    raw_csv_path = data_dir / "raw_wafers_lot_delta_50k.csv"
    obf_csv_path = data_dir / "obfuscated_wafers_lot_delta_50k.csv"

    raw_header = ["wafer_id", "RECIPE_ID"] + raw_cols + ["yield_pct"]
    obf_header = ["wafer_id"] + obf_cols + ["yield_pct"]

    # Precompute category scale factors for fast inner-loop generation
    col_scales = []
    for col in raw_cols:
        cat = next(c for c in PARAM_CATEGORIES if f"_{c}_" in col)
        low, high = RAW_RANGES[cat]
        col_scales.append((low, high - low))

    # Index positions of causal drivers in the 100-column vector:
    idx_pd1 = raw_cols.index("SPUTTER_DEPOSITION_PRESSURE_01")      # Primary driver (#1)
    idx_rfd1 = raw_cols.index("SPUTTER_DEPOSITION_RF_POWER_01")     # Interaction term 1
    idx_td1 = raw_cols.index("SPUTTER_DEPOSITION_TEMPERATURE_01")   # Interaction term 2
    idx_fle1 = raw_cols.index("CHEMICAL_CMP_FLOW_RATE_01")          # Secondary driver
    idx_durd1 = raw_cols.index("SPUTTER_DEPOSITION_DURATION_01")    # Auxiliary STEP_D driver

    uniform = rng.uniform
    gauss = rng.gauss
    n_recipes = len(CONFIDENTIAL_RECIPES)

    with raw_csv_path.open("w", newline="", encoding="utf-8", buffering=1024 * 1024) as f_raw, \
         obf_csv_path.open("w", newline="", encoding="utf-8", buffering=1024 * 1024) as f_obf:
        raw_writer = csv.writer(f_raw)
        obf_writer = csv.writer(f_obf)
        raw_writer.writerow(raw_header)
        obf_writer.writerow(obf_header)

        raw_batch: List[List[Any]] = []
        obf_batch: List[List[Any]] = []

        for w_idx in range(1, num_wafers + 1):
            wafer_id = f"WAFER_{w_idx:05d}"
            recipe_id = CONFIDENTIAL_RECIPES[w_idx % n_recipes]

            # Generate 100 normalized latent values in [0.0, 1.0]
            u_vals = [round(uniform(0.0, 1.0), 4) for _ in range(100)]
            raw_vals = [
                round(low + u_vals[i] * span, 2)
                for i, (low, span) in enumerate(col_scales)
            ]

            pd1 = u_vals[idx_pd1]
            rfd1 = u_vals[idx_rfd1]
            td1 = u_vals[idx_td1]
            fle1 = u_vals[idx_fle1]
            durd1 = u_vals[idx_durd1]

            # Planted semiconductor yield model for Lot Delta (50,000 wafers)
            base_yield = 93.8
            effect_main = -10.8 * pd1 - (4.8 if pd1 > 0.54 else 0.0)
            interaction_penalty = -11.4 * (rfd1 * (1.0 - td1)) - (4.2 if (rfd1 > 0.55 and td1 < 0.45) else 0.0)
            effect_sec = +5.1 * (fle1 - 0.5)
            effect_aux = -2.6 * (durd1 - 0.5)
            noise = gauss(0.0, 1.15)

            raw_y = base_yield + effect_main + interaction_penalty + effect_sec + effect_aux + noise
            yield_pct = max(65.0, min(99.4, round(raw_y, 2)))

            raw_batch.append([wafer_id, recipe_id] + raw_vals + [yield_pct])
            obf_batch.append([wafer_id] + u_vals + [yield_pct])

            if len(obf_batch) >= 5000:
                raw_writer.writerows(raw_batch)
                obf_writer.writerows(obf_batch)
                raw_batch.clear()
                obf_batch.clear()

        if obf_batch:
            raw_writer.writerows(raw_batch)
            obf_writer.writerows(obf_batch)

    elapsed_s = round(time.perf_counter() - t0, 3)
    raw_mb = round(raw_csv_path.stat().st_size / (1024 * 1024), 2)
    obf_mb = round(obf_csv_path.stat().st_size / (1024 * 1024), 2)

    return {
        "status": "success",
        "dataset_id": "lot_delta_50k",
        "num_wafers": num_wafers,
        "num_parameters": 100,
        "total_parameter_cells": num_wafers * 100,
        "raw_csv_path": str(raw_csv_path),
        "raw_csv_size_mb": raw_mb,
        "obfuscated_csv_path": str(obf_csv_path),
        "obfuscated_csv_size_mb": obf_mb,
        "elapsed_seconds": elapsed_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate 50,000 wafer records x 100 process parameters.")
    parser.add_argument("--records", type=int, default=50000, help="Number of wafer records to simulate (default: 50000)")
    parser.add_argument("--seed", type=int, default=512, help="Random seed")
    args = parser.parse_args()

    info = simulate_50k_wafer_dataset(num_wafers=args.records, seed=args.seed)
    print(
        f"[simulate_50k_dataset] Generated {info['num_wafers']:,} wafers x {info['num_parameters']} params "
        f"= {info['total_parameter_cells']:,} cells in {info['elapsed_seconds']}s "
        f"(obfuscated CSV: {info['obfuscated_csv_size_mb']} MB at {info['obfuscated_csv_path']})"
    )


if __name__ == "__main__":
    main()
