---
name: data-preprocess
description: "Obfuscates confidential process datasets (redacting Recipe IDs, mapping process steps like UNDERFILL_LAMINATION to STEP_A..STEP_E, normalizing values to [0, 1]) and selects parameter columns in batches (e.g. all temperatures, all pressures, or specific process steps) via a deterministic L3 script."
---

# Data Preprocess Agent Skill (`data_preprocess`)

> **Design Principle — 3-Level Progressive Disclosure & Shifting Intelligence Left (the Agent Skills whitepaper, pp. 44, 52):**
> - **Level 1 (Metadata Index):** YAML `name` + `description` (~45 tokens) loaded at startup so the agent carries specialized capability without **Context Rot**.
> - **Level 2 (Core Runbook Body):** This markdown SOP loaded on demand when the user selects or preprocesses a dataset.
> - **Level 3 (Bundled Executable Resource — Shifting Intelligence Left):** `scripts/preprocess.py` executes deterministic normalization, confidential step/recipe obfuscation, and batch regex column selection outside the LLM context window (0 code tokens in prompt, eliminating **Context Debt**).

## Standard Operating Procedure (SOP)

1. **Never Load Raw Confidential Values Into Prompt Context (`File Message Bus`, the Agent Skills whitepaper, p. 40):**
   - Raw datasets contain confidential `RECIPE_ID` strings (e.g., `RECIPE-8841-UF`) and proprietary process step names (`UNDERFILL_LAMINATION`, `PLASMA_ETCH`, `THERMAL_CURE`, `SPUTTER_DEPOSITION`, `CHEMICAL_CMP`).
   - Always invoke `scripts/preprocess.py` to generate `data/obfuscated_wafers.csv` and pass only the obfuscated file reference and selected column list in `session.state`.

2. **Batch Column Selection:**
   - Users can select parameters in batches by physical quantity (`TEMPERATURE`, `PRESSURE`, `RF_POWER`, `FLOW_RATE`, `DURATION`), by obfuscated process step (`STEP_A`, `STEP_B`, `STEP_C`, `STEP_D`, `STEP_E`), or `ALL`.
   - If the user selects more than 50 parameters at once in strict mode, request Human-in-the-Loop confirmation via `tool_context.request_confirmation()`.

## CLI Usage (`L3` Execution)

```bash
python3 skills/data_preprocess/scripts/preprocess.py --batches TEMPERATURE PRESSURE STEP_A
```
