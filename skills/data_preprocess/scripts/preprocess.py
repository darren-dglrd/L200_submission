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
    seed: int = 42,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Deterministically generate 100 unique wafers x 100 parameters = 10,000 parameter cells.

    Plants three ground-truth process yield drivers for reproducible ML & statistical testing:
      1. Main Effect #1: `UNDERFILL_LAMINATION_TEMPERATURE_01` (`STEP_A_TEMPERATURE_01`)
         strongly reduces yield when high (optimal operating window < 0.55).
      2. 2-Way Interaction #2: High `UNDERFILL_LAMINATION_TEMPERATURE_02` (`STEP_A_TEMPERATURE_02`)
         combined with Low `UNDERFILL_LAMINATION_PRESSURE_01` (`STEP_A_PRESSURE_01`) causes
         delamination / severe yield loss (-10.5% interaction penalty).
      3. Main Effect #3: `PLASMA_ETCH_RF_POWER_01` (`STEP_B_RF_POWER_01`) positively improves
         yield when above 0.45 (+4.2% gain).
      4. Static / Step-Level Effect: Because Drivers #1 and #2 (plus minor `UNDERFILL_LAMINATION`
         co-stress) reside in `UNDERFILL_LAMINATION` (`STEP_A`), `STEP_A` as a whole process stage
         is the dominant bottleneck step across the line.
    """
    target_dir = data_dir or DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

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
            # Uniform/beta-like spread across [0, 1]
            u = rng.uniform(0.02, 0.98)
            norm_latent[col] = u
            row[col] = round(low + u * (high - low), 3)

        # Compute realistic wafer yield (%) from planted causal drivers + minor noise
        t1 = norm_latent["UNDERFILL_LAMINATION_TEMPERATURE_01"]
        t2 = norm_latent["UNDERFILL_LAMINATION_TEMPERATURE_02"]
        p1 = norm_latent["UNDERFILL_LAMINATION_PRESSURE_01"]
        rf1 = norm_latent["PLASMA_ETCH_RF_POWER_01"]
        step_a_aux = norm_latent["UNDERFILL_LAMINATION_DURATION_01"]

        base_yield = 93.5
        # Driver 1: High STEP_A_TEMPERATURE_01 hurts yield sharply above 0.55
        effect_t1 = -9.5 * t1 - (4.5 if t1 > 0.55 else 0.0)
        # Driver 2: 2-Way Interaction (High STEP_A_TEMPERATURE_02 AND Low STEP_A_PRESSURE_01)
        interaction_penalty = -11.0 * (t2 * (1.0 - p1)) - (4.0 if (t2 > 0.55 and p1 < 0.45) else 0.0)
        # Driver 3: STEP_B_RF_POWER_01 positive effect
        effect_rf1 = +4.8 * (rf1 - 0.5)
        # Minor additional Step A contribution so STEP_A is clearly the #1 stage
        effect_step_a_aux = -2.2 * (step_a_aux - 0.5)
        noise = rng.gauss(0.0, 1.15)

        yield_pct = max(65.0, min(99.4, round(
            base_yield + effect_t1 + interaction_penalty + effect_rf1 + effect_step_a_aux + noise,
            2,
        )))
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
    raw_csv_path = target_dir / "raw_wafers.csv"
    fieldnames = ["wafer_id", "RECIPE_ID"] + raw_cols + ["yield_pct"]
    with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    # Write unpivoted event table CSV (10,000 parameter rows)
    unpivoted_csv_path = target_dir / "unpivoted_wafer_events.csv"
    with unpivoted_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["wafer_id", "process_step", "parameter_name", "parameter_value", "yield_pct"],
        )
        writer.writeheader()
        writer.writerows(unpivoted_events)

    return {
        "raw_csv_path": str(raw_csv_path),
        "unpivoted_csv_path": str(unpivoted_csv_path),
        "num_wafers": len(raw_rows),
        "num_parameters": len(raw_cols),
        "total_parameter_cells": len(raw_rows) * len(raw_cols),
    }


def obfuscate_and_select_batches(
    batches: Optional[List[str]] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Obfuscate confidential columns, normalize values to [0, 1], and filter by batch selection."""
    target_dir = data_dir or DATA_DIR
    raw_csv_path = target_dir / "raw_wafers.csv"
    if not raw_csv_path.exists():
        generate_wafer_dataset(data_dir=target_dir)

    with raw_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    raw_cols = get_raw_parameter_names()
    obf_col_map = {raw_col: obfuscate_column_name(raw_col) for raw_col in raw_cols}
    all_obf_cols = [obf_col_map[c] for c in raw_cols]

    # Compute min and max per parameter column for [0, 1] normalization
    col_Bounds: Dict[str, tuple] = {}
    for raw_col in raw_cols:
        vals = [float(r[raw_col]) for r in raw_rows]
        c_min, c_max = min(vals), max(vals)
        col_Bounds[raw_col] = (c_min, c_max if c_max > c_min else c_min + 1.0)

    obf_rows: List[Dict[str, Any]] = []
    for r in raw_rows:
        # Note: RECIPE_ID is deliberately stripped during obfuscation!
        obf_row: Dict[str, Any] = {"wafer_id": r["wafer_id"]}
        for raw_col in raw_cols:
            obf_col = obf_col_map[raw_col]
            c_min, c_max = col_Bounds[raw_col]
            norm_val = (float(r[raw_col]) - c_min) / (c_max - c_min)
            obf_row[obf_col] = round(norm_val, 4)
        obf_row["yield_pct"] = float(r["yield_pct"])
        obf_rows.append(obf_row)

    obf_csv_path = target_dir / "obfuscated_wafers.csv"
    obf_fieldnames = ["wafer_id"] + all_obf_cols + ["yield_pct"]
    with obf_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=obf_fieldnames)
        writer.writeheader()
        writer.writerows(obf_rows)

    # Batch selection logic
    requested_batches = [b.upper().strip() for b in (batches or ["ALL"]) if b.strip()]
    if not requested_batches or "ALL" in requested_batches:
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

    result = {
        "status": "success",
        "obfuscated_csv_path": str(obf_csv_path),
        "num_wafers": len(obf_rows),
        "total_parameters": len(all_obf_cols),
        "total_parameter_cells": len(obf_rows) * len(all_obf_cols),
        "selected_batches": requested_batches,
        "selected_columns_count": len(selected_cols),
        "selected_columns": selected_cols,
        "redacted_fields": ["RECIPE_ID"],
        "step_aliases": {v: "CONFIDENTIAL_PROCESS_STEP" for v in STEP_OBFUSCATION_MAP.values()},
        "batch_catalog_counts": {
            **{cat: len(cols) for cat, cols in batch_catalog["By Parameter Type"].items()},
            **{step: len(cols) for step, cols in batch_catalog["By Obfuscated Process Step"].items()},
        },
        "preview_rows": preview_rows,
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
