# Agent Harness Constitution (`AGENTS.md`)

> **Design Principle — Agentic Engineering Harness (the Agentic SDLC whitepaper, pp. 16, 26–29):**
> *"Agent = Model + Harness. Static context is always loaded: system instructions, rule files (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`), global memory and personas... Start an `AGENTS.md` with ten lines: stack, conventions, hard rules, workflow."*

## 1. Stack & Architecture
- **Framework:** Google ADK (`google-adk`) multi-agent orchestration (`LlmAgent`, `SequentialAgent`, `AgentTool`, `App`, `Runner`) with deterministic pure-Python statistical/ML execution (`app/stats_engine.py`) and zero-dependency local fallback.
- **UI Protocol:** **A2UI v0.9 (`Agent-to-User Interface`)** streaming `createSurface`, `updateComponents` (flat adjacency list), and `updateDataModel` (RFC 6901 JSON Pointers) using the 18-component Basic Catalog (`Agent Tools & Interoperability.pdf`, pp. 33–42).
- **Procedural Memory Skill:** `skills/data_preprocess/SKILL.md` + `scripts/preprocess.py` implementing **3-Level Progressive Disclosure** and **"Shifting Intelligence Left"** (the Agent Skills whitepaper, pp. 44, 52).

## 2. Non-Negotiable Invariants & Guardrails
1. **Subject-Matter Agnostic Dataset Invariant:** Input is a single dataset keyed by a unique entity ID (`wafer_id`, `WAFER_001`..`WAFER_100`) with `100` parameters across process steps (`100 × 100 = 10,000` parameter cells), plus an unpivoted event table (`wafer_id, process_step, parameter_name, parameter_value`).
2. **Sensitive-Data Classifier (`after_model_callback` Pre-Response Hook):** *"Since Recipe IDs and raw process step names (`RECIPE-xxxx`, `UNDERFILL_LAMINATION`, `PLASMA_ETCH`) are highly confidential, we use regex patterns and ML/heuristic classifiers in a pre-response hook (`SafetyPlugin.after_model_callback`) to ensure no confidential output reaches the user or persistent logs."* (`Agent Quality.pdf`, p. 29; `Introduction to Agents.pdf`, pp. 38–39).
3. **File Message Bus Over Context Stuffing:** Never pass the 10,000-cell matrix inside prompt prose; pass file references (`data/obfuscated_wafers.csv`) via `session.state` (the Agent Skills whitepaper, p. 40).
4. **Zero-ML-Jargon Output:** Translate all $p$-values, Benjamini-Hochberg FDR $q$-values, group ANOVA $F$-tests, and tree-split importances into plain-English verdicts and **Good (Green) vs. Bad (Red) Wafer Distribution Charts**.
5. **Verification Defines the Mode:** Every change must pass both deterministic unit tests (`python3 -m unittest discover -s tests -v`) and Glass-Box Trajectory / $\text{pass}^k$ evaluations (`evals/hypothesis_workflow.test.json`).
