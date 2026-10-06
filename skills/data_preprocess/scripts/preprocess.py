#!/usr/bin/env python3
"""Level 3 Deterministic Executable for the `data_preprocess` Agent Skill.

Design Principles Invoked Verbatim:
1. Shifting Intelligence Left (the Agent Skills whitepaper, p. 44):
   "Move deterministic validation, parsing, math, and schema checks out of prompt prose
   and into scripts/. When the agent executes a script in L3, only the concise stdout
   result enters the context window, achieving 100% determinism at near-zero token cost."
2. File Message Bus (the Agent Skills whitepaper, p. 40):
   "Never use the LLM's conversational context window as a message bus between workflow
   stages. Pass state between skill nodes via structured file references on disk."
3. Confidential Data Obfuscation & Batch Selection:
   - Generates (or ingests) a 100-wafer x 100-parameter dataset (10,000 parameter cells)
     plus an unpivoted event table (`wafer_id, process_step, parameter_name, parameter_value`).
   - Redacts confidential `RECIPE_ID` values (`RECIPE-8841-UF`, etc.).
   - Replaces proprietary process step names (`UNDERFILL_LAMINATION` -> `STEP_A`, etc.).
   - Min-max normalizes all numerical parameter columns to [0.0, 1.0].
   - Selects parameter columns in batches (e.g., `TEMPERATURE`, `PRESSURE`, `STEP_A`, `ALL`).
"""

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

# Confidential Raw Step -> Obfuscated Step Mapping
STEP_OBFUSCATION_MAP: Dict[str, str] = {
    "UNDERFILL_LAMINATION": "STEP_A",
    "PLASMA_ETCH": "STEP_B",
    "THERMAL_CURE": "STEP_C",
    "SPUTTER_DEPOSITION": "STEP_D",
    "CHEMICAL_CMP": "STEP_E",
}

REVERSE_STEP_MAP: Dict[str, str] = {v: k for k, v in STEP_OBFUSCATION_MAP.items()}

PARAM_CATEGORIES: List[str] = [
    "TEMPERATURE",
    "PRESSURE",
    "RF_POWER",
    "FLOW_RATE",
    "DURATION",
]

# Physical units/ranges for realistic raw dataset generation before normalization
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
]

# Catalog of 4 Selectable Semiconductor Process Datasets (3 x 100-Wafer Lots + 1 x 50,000-Wafer Stress Test Lot)
DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    "lot_alpha": {
        "dataset_id": "lot_alpha",
        "name": "Dataset 1: Lot Alpha — Underfill Thermal & Pressure Drift",
        "short_name": "1. Lot Alpha (STEP_A Thermal • 100 Wafers)",
        "description": "100 wafers × 100 params (10,000 cells). Primary excursion in STEP_A temperature & temp×pressure coupling.",
        "seed": 42,
        "num_wafers": 100,
        "raw_filename": "raw_wafers.csv",
        "obf_filename": "obfuscated_wafers.csv",
        "primary_stage": "STEP_A",
    },
    "lot_beta": {
        "dataset_id": "lot_beta",
        "name": "Dataset 2: Lot Beta — Plasma Etch RF Power & Flow Instability",
        "short_name": "2. Lot Beta (STEP_B Plasma RF • 100 Wafers)",
        "description": "100 wafers × 100 params (10,000 cells). Primary excursion in STEP_B RF power arcing & gas flow×pressure.",
        "seed": 108,
        "num_wafers": 100,
        "raw_filename": "raw_wafers_lot_beta.csv",
        "obf_filename": "obfuscated_wafers_lot_beta.csv",
        "primary_stage": "STEP_B",
    },
    "lot_gamma": {
        "dataset_id": "lot_gamma",
        "name": "Dataset 3: Lot Gamma — Thermal Cure Dwell & Sputter Pressure",
        "short_name": "3. Lot Gamma (STEP_C Dwell Time • 100 Wafers)",
        "description": "100 wafers × 100 params (10,000 cells). Primary excursion in STEP_C dwell duration & thermal cure budget.",
        "seed": 256,
        "num_wafers": 100,
        "raw_filename": "raw_wafers_lot_gamma.csv",
        "obf_filename": "obfuscated_wafers_lot_gamma.csv",
        "primary_stage": "STEP_C",
    },
    "lot_delta_50k": {
        "dataset_id": "lot_delta_50k",
        "name": "Dataset 4: Lot Delta — 50,000 Wafers Stress Test (STEP_D Sputter Excursion)",
        "short_name": "4. Lot Delta — 50,000 Wafers (Limit Test)",
        "description": "50,000 wafers × 100 params (5,000,000 cells). High-volume limit test with STEP_D pressure & RF×temp excursion.",
        "seed": 512,
        "num_wafers": 50000,
        "raw_filename": "raw_wafers_lot_delta_50k.csv",
        "obf_filename": "obfuscated_wafers_lot_delta_50k.csv",
        "primary_stage": "STEP_D",
    },
}


def normalize_dataset_id(dataset_id: Optional[str] = None) -> str:
    """Validate and normalize dataset_id to one of the 4 supported datasets."""
    if not dataset_id:
        return "lot_alpha"
    key = str(dataset_id).strip().lower()
    if key in ("50k", "delta", "lot_delta", "lot_delta_50k"):
        return "lot_delta_50k"
    return key if key in DATASET_CATALOG else "lot_alpha"


def get_raw_parameter_names() -> List[str]:
    """Return 100 raw process parameter column names (5 steps x 5 categories x 4 sensors = 100)."""
    cols: List[str] = []
    for raw_step in STEP_OBFUSCATION_MAP.keys():
        for cat in PARAM_CATEGORIES:
            for idx in range(1, 5):
                cols.append(f"{raw_step}_{cat}_{idx:02d}")
    return cols


def obfuscate_column_name(raw_col: str) -> str:
    """Replace confidential step prefixes like UNDERFILL_LAMINATION with STEP_A."""
    for raw_step, obf_step in STEP_OBFUSCATION_MAP.items():
        if raw_col.startswith(raw_step + "_"):
            return raw_col.replace(raw_step + "_", obf_step + "_", 1)
    return raw_col


def generate_wafer_dataset(
    num_wafers: int = 100,
    seed: Optional[int] = None,
    data_dir: Optional[Path] = None,
    dataset_id: str = "lot_alpha",
) -> Dict[str, Any]:
    """Deterministically generate wafer x 100 parameter datasets (including 50,000-wafer Lot Delta)."""
    ds_key = normalize_dataset_id(dataset_id)
    ds_meta = DATASET_CATALOG[ds_key]
    target_dir = data_dir or DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if ds_key == "lot_delta_50k":
        from scripts.simulate_50k_dataset import simulate_50k_wafer_dataset
        n_target = int(ds_meta.get("num_wafers", 50000)) if num_wafers == 100 else num_wafers
        sim_res = simulate_50k_wafer_dataset(
            num_wafers=n_target,
            seed=seed if seed is not None else int(ds_meta["seed"]),
            data_dir=target_dir,
        )
        return {
            "dataset_id": ds_key,
            "dataset_name": ds_meta["name"],
            "raw_csv_path": sim_res["raw_csv_path"],
            "unpivoted_csv_path": sim_res["obfuscated_csv_path"],
            "num_wafers": sim_res["num_wafers"],
            "num_parameters": sim_res["num_parameters"],
            "total_parameter_cells": sim_res["total_parameter_cells"],
        }

    actual_seed = seed if seed is not None else int(ds_meta["seed"])
    rng = random.Random(actual_seed)

    raw_cols = get_raw_parameter_names()
    assert len(raw_cols) == 100, f"Expected 100 parameters, got {len(raw_cols)}"

    raw_rows: List[Dict[str, Any]] = []
    unpivoted_events: List[Dict[str, Any]] = []

    for w_idx in range(1, num_wafers + 1):
        wafer_id = f"WAFER_{w_idx:03d}"
        recipe_id = CONFIDENTIAL_RECIPES[w_idx % len(CONFIDENTIAL_RECIPES)]
        row: Dict[str, Any] = {"wafer_id": wafer_id, "RECIPE_ID": recipe_id}

        # Generate raw physical values for all 100 parameters
        norm_latent: Dict[str, float] = {}
        for col in raw_cols:
            cat = next(c for c in PARAM_CATEGORIES if f"_{c}_" in col)
            low, high = RAW_RANGES[cat]
            u = rng.uniform(0.02, 0.98)
            norm_latent[col] = u
            row[col] = round(low + u * (high - low), 3)

        base_yield = 93.5
        noise = rng.gauss(0.0, 1.15)

        if ds_key == "lot_beta":
            # Dataset 2 (Lot Beta): STEP_B (PLASMA_ETCH) RF Power & Flow x Pressure Excursion
            rf1 = norm_latent["PLASMA_ETCH_RF_POWER_01"]
            fl2 = norm_latent["PLASMA_ETCH_FLOW_RATE_01"]
            pr1 = norm_latent["PLASMA_ETCH_PRESSURE_01"]
            dur_c = norm_latent["THERMAL_CURE_DURATION_01"]
            step_b_aux = norm_latent["PLASMA_ETCH_TEMPERATURE_01"]

            effect_main = -10.2 * rf1 - (4.8 if rf1 > 0.54 else 0.0)
            interaction_penalty = -11.2 * (fl2 * (1.0 - pr1)) - (4.2 if (fl2 > 0.55 and pr1 < 0.45) else 0.0)
            effect_sec = -5.0 * (dur_c - 0.5)
            effect_aux = -2.4 * (step_b_aux - 0.5)
            raw_y = base_yield + effect_main + interaction_penalty + effect_sec + effect_aux + noise
        elif ds_key == "lot_gamma":
            # Dataset 3 (Lot Gamma): STEP_C (THERMAL_CURE) Dwell Duration & Temp x Pressure Excursion
            dur1 = norm_latent["THERMAL_CURE_DURATION_01"]
            tc2 = norm_latent["THERMAL_CURE_TEMPERATURE_01"]
            pc1 = norm_latent["THERMAL_CURE_PRESSURE_01"]
            pd1 = norm_latent["SPUTTER_DEPOSITION_PRESSURE_01"]
            step_c_aux = norm_latent["THERMAL_CURE_FLOW_RATE_01"]

            effect_main = -10.5 * dur1 - (4.6 if dur1 > 0.54 else 0.0)
            interaction_penalty = -11.0 * (tc2 * (1.0 - pc1)) - (4.0 if (tc2 > 0.55 and pc1 < 0.45) else 0.0)
            effect_sec = +5.2 * (pd1 - 0.5)
            effect_aux = -2.3 * (step_c_aux - 0.5)
            raw_y = base_yield + effect_main + interaction_penalty + effect_sec + effect_aux + noise
        else:
            # Dataset 1 (Lot Alpha - Default): STEP_A (UNDERFILL_LAMINATION) Thermal & Pressure Excursion
            t1 = norm_latent["UNDERFILL_LAMINATION_TEMPERATURE_01"]
            t2 = norm_latent["UNDERFILL_LAMINATION_TEMPERATURE_02"]
            p1 = norm_latent["UNDERFILL_LAMINATION_PRESSURE_01"]
            rf1 = norm_latent["PLASMA_ETCH_RF_POWER_01"]
            step_a_aux = norm_latent["UNDERFILL_LAMINATION_DURATION_01"]

            effect_t1 = -9.5 * t1 - (4.5 if t1 > 0.55 else 0.0)
            interaction_penalty = -11.0 * (t2 * (1.0 - p1)) - (4.0 if (t2 > 0.55 and p1 < 0.45) else 0.0)
            effect_rf1 = +4.8 * (rf1 - 0.5)
            effect_step_a_aux = -2.2 * (step_a_aux - 0.5)
            raw_y = base_yield + effect_t1 + interaction_penalty + effect_rf1 + effect_step_a_aux + noise

        yield_pct = max(65.0, min(99.4, round(raw_y, 2)))
        row["yield_pct"] = yield_pct
        raw_rows.append(row)

        # Populate unpivoted event table rows: (wafer_id, process_step, parameter_name, parameter_value)
        for col in raw_cols:
            raw_step = next(s for s in STEP_OBFUSCATION_MAP if col.startswith(s + "_"))
            param_suffix = col[len(raw_step) + 1:]
            unpivoted_events.append({
                "wafer_id": wafer_id,
                "process_step": raw_step,
                "parameter_name": param_suffix,
                "parameter_value": row[col],
                "yield_pct": yield_pct,
            })

    # Write raw pivoted CSV (100 wafers x 100 parameters + wafer_id, RECIPE_ID, yield_pct)
    raw_csv_path = target_dir / ds_meta["raw_filename"]
    fieldnames = ["wafer_id", "RECIPE_ID"] + raw_cols + ["yield_pct"]
    with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    # Write unpivoted event table CSV (10,000 parameter rows) for default lot_alpha or dataset-specific
    unpiv_name = "unpivoted_wafer_events.csv" if ds_key == "lot_alpha" else f"unpivoted_wafer_events_{ds_key}.csv"
    unpivoted_csv_path = target_dir / unpiv_name
    with unpivoted_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["wafer_id", "process_step", "parameter_name", "parameter_value", "yield_pct"],
        )
        writer.writeheader()
        writer.writerows(unpivoted_events)

    return {
        "dataset_id": ds_key,
        "dataset_name": ds_meta["name"],
        "raw_csv_path": str(raw_csv_path),
        "unpivoted_csv_path": str(unpivoted_csv_path),
        "num_wafers": len(raw_rows),
        "num_parameters": len(raw_cols),
        "total_parameter_cells": len(raw_rows) * len(raw_cols),
    }


def obfuscate_and_select_batches(
    batches: Optional[List[str]] = None,
    custom_columns: Optional[List[str]] = None,
    data_dir: Optional[Path] = None,
    dataset_id: str = "lot_alpha",
    row_offset: int = 0,
    row_limit: int = 250,
) -> Dict[str, Any]:
    """Obfuscate confidential columns, normalize values to [0, 1], and filter by batch or hand-picked columns."""
    import itertools
    ds_key = normalize_dataset_id(dataset_id)
    ds_meta = DATASET_CATALOG[ds_key]
    target_dir = data_dir or DATA_DIR
    raw_csv_path = target_dir / ds_meta["raw_filename"]
    obf_csv_path = target_dir / ds_meta["obf_filename"]

    raw_cols = get_raw_parameter_names()
    obf_col_map = {raw_col: obfuscate_column_name(raw_col) for raw_col in raw_cols}
    all_obf_cols = [obf_col_map[c] for c in raw_cols]

    if ds_key == "lot_delta_50k":
        if not obf_csv_path.exists() or not raw_csv_path.exists():
            generate_wafer_dataset(data_dir=target_dir, dataset_id=ds_key)
        total_wafers = int(ds_meta.get("num_wafers", 50000))
        safe_offset = max(0, min(total_wafers - 1, int(row_offset or 0)))
        safe_limit = max(10, min(500, int(row_limit or 250)))
        obf_rows = []
        with obf_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in itertools.islice(reader, safe_offset, safe_offset + safe_limit):
                obf_row: Dict[str, Any] = {"wafer_id": r["wafer_id"], "yield_pct": float(r["yield_pct"])}
                for obf_col in all_obf_cols:
                    obf_row[obf_col] = float(r[obf_col])
                obf_rows.append(obf_row)
    else:
        if not raw_csv_path.exists():
            generate_wafer_dataset(data_dir=target_dir, dataset_id=ds_key)

        with raw_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_rows = list(reader)

        # Compute min and max per parameter column for [0, 1] normalization
        col_Bounds: Dict[str, tuple] = {}
        for raw_col in raw_cols:
            vals = [float(r[raw_col]) for r in raw_rows]
            c_min, c_max = min(vals), max(vals)
            col_Bounds[raw_col] = (c_min, c_max if c_max > c_min else c_min + 1.0)

        obf_rows = []
        for r in raw_rows:
            # Note: RECIPE_ID is deliberately stripped during obfuscation!
            obf_row = {"wafer_id": r["wafer_id"]}
            for raw_col in raw_cols:
                obf_col = obf_col_map[raw_col]
                c_min, c_max = col_Bounds[raw_col]
                norm_val = (float(r[raw_col]) - c_min) / (c_max - c_min)
                obf_row[obf_col] = round(norm_val, 4)
            obf_row["yield_pct"] = float(r["yield_pct"])
            obf_rows.append(obf_row)

        obf_fieldnames = ["wafer_id"] + all_obf_cols + ["yield_pct"]
        with obf_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=obf_fieldnames)
            writer.writeheader()
            writer.writerows(obf_rows)
        total_wafers = len(obf_rows)
        safe_offset = 0
        safe_limit = total_wafers

    # Column selection logic: either explicit hand-picked custom_columns OR batch tokens
    requested_batches = [b.upper().strip() for b in (batches or ["ALL"]) if b.strip()]
    if custom_columns is not None and len(custom_columns) > 0:
        valid_set = set(all_obf_cols)
        selected_cols = [c for c in custom_columns if c in valid_set]
        if not selected_cols:
            selected_cols = list(all_obf_cols)
    elif not requested_batches or "ALL" in requested_batches:
        selected_cols = list(all_obf_cols)
    else:
        selected_cols = []
        for col in all_obf_cols:
            if any(token in col for token in requested_batches):
                selected_cols.append(col)
        if not selected_cols:
            selected_cols = list(all_obf_cols)

    # Available batch groups summary for UI pills
    batch_catalog = {
        "By Parameter Type": {
            cat: [c for c in all_obf_cols if f"_{cat}_" in c]
            for cat in PARAM_CATEGORIES
        },
        "By Obfuscated Process Step": {
            step: [c for c in all_obf_cols if c.startswith(step + "_")]
            for step in STEP_OBFUSCATION_MAP.values()
        },
    }

    preview_rows = [
        {
            "wafer_id": row["wafer_id"],
            "yield_pct": row["yield_pct"],
            **{c: row[c] for c in selected_cols[:6]},
        }
        for row in obf_rows[:8]
    ]

    full_rows = [
        {
            "wafer_id": row["wafer_id"],
            "yield_pct": row["yield_pct"],
            **{c: row[c] for c in selected_cols},
        }
        for row in obf_rows
    ]

    available_datasets = [
        {
            "dataset_id": k,
            "name": v["name"],
            "short_name": v["short_name"],
            "description": v["description"],
            "primary_stage": v["primary_stage"],
            "num_wafers": v.get("num_wafers", 100),
        }
        for k, v in DATASET_CATALOG.items()
    ]

    result = {
        "status": "success",
        "dataset_id": ds_key,
        "dataset_name": ds_meta["name"],
        "dataset_short_name": ds_meta["short_name"],
        "dataset_description": ds_meta["description"],
        "available_datasets": available_datasets,
        "obfuscated_csv_path": str(obf_csv_path),
        "num_wafers": total_wafers,
        "total_rows": total_wafers,
        "total_parameters": len(all_obf_cols),
        "total_parameter_cells": total_wafers * len(all_obf_cols),
        "is_paginated": total_wafers > len(full_rows),
        "row_offset": safe_offset,
        "row_limit": safe_limit,
        "returned_rows_count": len(full_rows),
        "displayed_rows_count": len(full_rows),
        "all_columns": all_obf_cols,
        "selected_batches": requested_batches,
        "selected_columns_count": len(selected_cols),
        "selected_columns": selected_cols,
        "redacted_fields": ["RECIPE_ID"],
        "step_aliases": {v: "CONFIDENTIAL_PROCESS_STEP" for v in STEP_OBFUSCATION_MAP.values()},
        "batch_catalog": batch_catalog,
        "batch_catalog_counts": {
            **{cat: len(cols) for cat, cols in batch_catalog["By Parameter Type"].items()},
            **{step: len(cols) for step, cols in batch_catalog["By Obfuscated Process Step"].items()},
        },
        "preview_rows": preview_rows,
        "full_rows": full_rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data_preprocess L3 skill script.")
    parser.add_argument(
        "--batches",
        nargs="*",
        default=["ALL"],
        help="Batch tokens to select (e.g. TEMPERATURE PRESSURE STEP_A ALL)",
    )
    args = parser.parse_args()
    res = obfuscate_and_select_batches(batches=args.batches)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
