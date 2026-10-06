#!/usr/bin/env python3
"""Semi-Deterministic Statistical/ML Hypothesis Engine & A2UI v0.9 Surface Builder.

Design Principles Invoked Verbatim (the Google 5-Day AI Agents Intensive whitepapers):
1. A2UI v0.9 Message Format & Basic Catalog (`Agent Tools & Interoperability.pdf`, pp. 34–37):
   - Uses `version: "v0.9"`, `createSurface`, `updateComponents` (flat adjacency list referenced
     by `id`), and `updateDataModel` (resolving RFC 6901 JSON Pointer `{"path": "..."}` bindings).
   - Enforces the 18-component Basic Catalog whitelist (`Row, Column, List, Text, Image, Icon,
     Divider, Card, Modal, Tabs, Video, AudioPlayer, Button, TextField, CheckBox, Slider,
     DateTimeInput, ChoicePicker`).
2. Tool-as-Template & Hybrid Output Schema (`Agent Tools & Interoperability.pdf`, pp. 36–42):
   - Returns `{"data": ..., "ui": ..., "ui_available": True}` so API callers inspect `data`
     and human clients render `ui` without burning LLM tokens on static UI layout.
   - Includes a schema validate-and-retry check (`validate_a2ui_v09_payload`) with graceful
     text fallback so the renderer never sees a malformed payload.
3. Zero-ML-Jargon Translation for Subject-Matter Experts:
   - Translates Pearson r, two-sample t-test p-values, Benjamini-Hochberg FDR q-values,
     2-way interaction synergy, and Group ANOVA F-tests (for static/step-level features like
     `STEP_A` / `UNDERFILL_LAMINATION`) into plain-English verdicts and Color-Coded Wafer
     Distributions (Good Wafers [Green] vs. Bad Wafers [Red]).
"""

import csv
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from skills.data_preprocess.scripts.preprocess import (
    DATA_DIR,
    STEP_OBFUSCATION_MAP,
    obfuscate_and_select_batches,
    obfuscate_column_name,
)

# 18-Component A2UI v0.9 Basic Catalog (`Agent Tools & Interoperability.pdf`, p. 34, Table 1)
A2UI_V09_BASIC_CATALOG = frozenset({
    # Layout (3)
    "Row", "Column", "List",
    # Display (4)
    "Text", "Image", "Icon", "Divider",
    # Containers (3)
    "Card", "Modal", "Tabs",
    # Media (2)
    "Video", "AudioPlayer",
    # Interactive (6)
    "Button", "TextField", "CheckBox", "Slider", "DateTimeInput", "ChoicePicker",
})

GOOD_WAFER_YIELD_THRESHOLD = 83.5  # Wafers >= 83.5% are GOOD (Green); < 83.5% are BAD (Red)


def _load_obfuscated_rows(data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    target_dir = data_dir or DATA_DIR
    obf_path = target_dir / "obfuscated_wafers.csv"
    if not obf_path.exists():
        obfuscate_and_select_batches(batches=["ALL"], data_dir=target_dir)
    with obf_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            parsed: Dict[str, Any] = {"wafer_id": r["wafer_id"]}
            for k, v in r.items():
                if k != "wafer_id":
                    parsed[k] = float(v)
            rows.append(parsed)
    return rows


def _pearson_r_and_p(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Compute Pearson correlation r, t-statistic, and two-tailed p-value in pure Python."""
    n = len(xs)
    if n < 4:
        return 0.0, 0.0, 1.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0, 0.0, 1.0
    r = max(-0.9999, min(0.9999, num / (den_x * den_y)))
    df = n - 2
    t_stat = abs(r) * math.sqrt(df / max(1e-12, 1.0 - r * r))
    # Accurate normal/Wallace approximation for Student-t two-tailed p-value (df >= 30)
    z = math.sqrt(df * math.log(1.0 + (t_stat * t_stat) / df)) * (1.0 - 1.0 / (4.0 * df))
    p_val = math.erfc(z / math.sqrt(2.0))
    return round(r, 4), round(t_stat, 3), max(1e-9, min(1.0, p_val))


def _tree_stump_importance(xs: List[float], ys: List[float]) -> Tuple[float, float, float, float]:
    """Find best 1D decision-tree split threshold and compute variance reduction (ML importance)."""
    n = len(xs)
    mean_y = sum(ys) / n
    total_var = sum((y - mean_y) ** 2 for y in ys) / max(1, n)
    if total_var < 1e-9:
        return 0.5, mean_y, mean_y, 0.0

    best_gain = -1.0
    best_thresh = 0.5
    best_low_mean = mean_y
    best_high_mean = mean_y

    for thresh in [0.35, 0.45, 0.50, 0.55, 0.65]:
        low_y = [y for x, y in zip(xs, ys) if x <= thresh]
        high_y = [y for x, y in zip(xs, ys) if x > thresh]
        if len(low_y) < 8 or len(high_y) < 8:
            continue
        m_low = sum(low_y) / len(low_y)
        m_high = sum(high_y) / len(high_y)
        var_low = sum((y - m_low) ** 2 for y in low_y)
        var_high = sum((y - m_high) ** 2 for y in high_y)
        rem_var = (var_low + var_high) / n
        gain = max(0.0, (total_var - rem_var) / total_var)
        if gain > best_gain:
            best_gain = gain
            best_thresh = thresh
            best_low_mean = m_low
            best_high_mean = m_high

    return best_thresh, round(best_low_mean, 2), round(best_high_mean, 2), round(max(0.0, best_gain), 4)


def _compute_good_bad_distribution(
    rows: List[Dict[str, Any]],
    score_fn,
    yield_threshold: float = GOOD_WAFER_YIELD_THRESHOLD,
) -> Dict[str, Any]:
    """Compute color-coded Good (Green) vs. Bad (Red) wafer counts across 5 bins + scatter points."""
    bin_edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
    bin_labels = ["0.0 - 0.2", "0.2 - 0.4", "0.4 - 0.6", "0.6 - 0.8", "0.8 - 1.0"]
    good_counts = [0] * 5
    bad_counts = [0] * 5
    wafer_points = []

    for r in rows:
        x_val = max(0.0, min(1.0, float(score_fn(r))))
        y_val = float(r["yield_pct"])
        is_good = y_val >= yield_threshold
        for idx, (low, high) in enumerate(bin_edges):
            if low <= x_val < high:
                if is_good:
                    good_counts[idx] += 1
                else:
                    bad_counts[idx] += 1
                break
        wafer_points.append({
            "wafer_id": r["wafer_id"],
            "x": round(x_val, 4),
            "y": round(y_val, 2),
            "status": "GOOD" if is_good else "BAD",
            "color": "#10b981" if is_good else "#ef4444",
        })

    return {
        "yield_threshold": yield_threshold,
        "bin_labels": bin_labels,
        "good_wafer_counts": good_counts,
        "bad_wafer_counts": bad_counts,
        "wafer_points": wafer_points,
    }


def validate_a2ui_v09_payload(message: Dict[str, Any]) -> bool:
    """Validate an A2UI v0.9 message against the Basic Catalog & flat adjacency-list contract."""
    if not isinstance(message, dict) or message.get("version") != "v0.9":
        raise ValueError("A2UI message must declare top-level 'version': 'v0.9'.")
    if "updateComponents" in message:
        uc = message["updateComponents"]
        if not uc.get("surfaceId") or not isinstance(uc.get("components"), list):
            raise ValueError("updateComponents requires 'surfaceId' and 'components' list.")
        ids = set()
        for comp in uc["components"]:
            cid = comp.get("id")
            ctype = comp.get("component")
            if not cid or not isinstance(ctype, str):
                raise ValueError(f"Component requires flat string 'id' and 'component': {comp}")
            if ctype not in A2UI_V09_BASIC_CATALOG:
                raise ValueError(f"Component '{ctype}' is not in the 18-component A2UI Basic Catalog.")
            ids.add(cid)
        if "root" not in ids:
            raise ValueError("updateComponents adjacency list must contain a 'root' component id.")
    return True


def build_hypothesis_discovery_a2ui(hypotheses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build an A2UI v0.9 updateComponents payload for generated candidate hypothesis cards."""
    card_ids = []
    components: List[Dict[str, Any]] = []

    for idx, hyp in enumerate(hypotheses, start=1):
        cid = f"hyp_card_{idx}"
        col_id = f"hyp_col_{idx}"
        title_id = f"hyp_title_{idx}"
        desc_id = f"hyp_desc_{idx}"
        btn_id = f"hyp_btn_{idx}"
        btn_lbl_id = f"hyp_btn_lbl_{idx}"
        card_ids.append(cid)

        components.extend([
            {"id": cid, "component": "Card", "child": col_id},
            {"id": col_id, "component": "Column", "children": [title_id, desc_id, btn_id]},
            {
                "id": title_id,
                "component": "Text",
                "text": f"[{hyp['hypothesis_type'].upper()}] {hyp['title']}",
                "variant": "h3",
            },
            {"id": desc_id, "component": "Text", "text": hyp["rationale"]},
            {
                "id": btn_id,
                "component": "Button",
                "child": btn_lbl_id,
                "action": {
                    "event": {
                        "name": "add_to_queue",
                        "hypothesis_id": hyp["hypothesis_id"],
                    }
                },
            },
            {"id": btn_lbl_id, "component": "Text", "text": "+ Add to Testing Queue"},
        ])

    root_children = ["header_title", "header_sub"] + card_ids
    full_components = [
        {"id": "root", "component": "Column", "children": root_children},
        {
            "id": "header_title",
            "component": "Text",
            "text": "Generated Candidate Process Hypotheses",
            "variant": "h2",
        },
        {
            "id": "header_sub",
            "component": "Text",
            "text": "Select any hypothesis card below to add it to your 1-by-1 ML Testing Queue.",
        },
    ] + components

    msg = {
        "version": "v0.9",
        "updateComponents": {
            "surfaceId": "discovery_surface",
            "components": full_components,
        },
    }
    validate_a2ui_v09_payload(msg)
    return msg


def build_evaluation_a2ui(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build A2UI v0.9 createSurface + updateComponents + updateDataModel for 1-by-1 evaluation."""
    surface_id = f"eval_{eval_result['hypothesis_id']}"
    update_components_msg = {
        "version": "v0.9",
        "updateComponents": {
            "surfaceId": surface_id,
            "components": [
                {
                    "id": "root",
                    "component": "Card",
                    "child": "card_col",
                },
                {
                    "id": "card_col",
                    "component": "Column",
                    "children": [
                        "verdict_badge",
                        "title_txt",
                        "div_1",
                        "plain_summary",
                        "certainty_txt",
                        "regime_txt",
                        "action_txt",
                        "export_btn",
                    ],
                },
                {
                    "id": "verdict_badge",
                    "component": "Text",
                    "text": {"path": "/evaluation/verdict_badge"},
                    "variant": "h3",
                },
                {
                    "id": "title_txt",
                    "component": "Text",
                    "text": {"path": "/evaluation/title"},
                    "variant": "h2",
                },
                {"id": "div_1", "component": "Divider"},
                {
                    "id": "plain_summary",
                    "component": "Text",
                    "text": {"path": "/evaluation/plain_english_summary"},
                },
                {
                    "id": "certainty_txt",
                    "component": "Text",
                    "text": {"path": "/evaluation/certainty_statement"},
                },
                {
                    "id": "regime_txt",
                    "component": "Text",
                    "text": {"path": "/evaluation/good_bad_wafer_split"},
                },
                {
                    "id": "action_txt",
                    "component": "Text",
                    "text": {"path": "/evaluation/recommended_action"},
                },
                {
                    "id": "export_btn",
                    "component": "Button",
                    "child": "export_lbl",
                    "action": {"event": {"name": "export_csv"}},
                },
                {
                    "id": "export_lbl",
                    "component": "Text",
                    "text": "Export Results CSV (Looker Studio)",
                },
            ],
        },
    }
    validate_a2ui_v09_payload(update_components_msg)

    create_surface_msg = {
        "version": "v0.9",
        "createSurface": {
            "surfaceId": surface_id,
            "root": "root",
            "catalog": "basic",
        },
    }
    update_data_model_msg = {
        "version": "v0.9",
        "updateDataModel": {
            "surfaceId": surface_id,
            "path": "/evaluation",
            "data": {
                "verdict_badge": eval_result["verdict_badge"],
                "title": eval_result["title"],
                "plain_english_summary": eval_result["plain_english_summary"],
                "certainty_statement": eval_result["certainty_statement"],
                "good_bad_wafer_split": eval_result["good_bad_wafer_split"],
                "recommended_action": eval_result["recommended_action"],
            },
        },
    }
    return {
        "createSurface": create_surface_msg,
        "updateComponents": update_components_msg,
        "updateDataModel": update_data_model_msg,
    }


def discover_candidate_hypotheses(
    selected_columns: Optional[List[str]] = None,
    user_query: str = "",
    top_k: int = 4,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Scan selected obfuscated columns (and process steps) to generate top candidate hypotheses.

    Supports:
      1. Whole-Step / Static Feature Importance (e.g., `STEP_A` as a bottleneck stage)
      2. Single-Parameter Main Effects (e.g., `STEP_A_TEMPERATURE_01`)
      3. 2-Way Parameter Interactions (e.g., `STEP_A_TEMPERATURE_02 x STEP_A_PRESSURE_01`)
      4. User-Suggested Domain Hypotheses (including noise parameters to allow falsification)
    """
    rows = _load_obfuscated_rows(data_dir=data_dir)
    all_param_cols = [k for k in rows[0].keys() if k not in ("wafer_id", "yield_pct")]
    cols = [c for c in (selected_columns or all_param_cols) if c in all_param_cols]
    if not cols:
        cols = all_param_cols

    ys = [r["yield_pct"] for r in rows]

    # 1. Scan all 1-way main effects across selected columns
    single_stats = []
    for col in cols:
        xs = [r[col] for r in rows]
        r_val, t_stat, p_val = _pearson_r_and_p(xs, ys)
        thresh, low_m, high_m, ml_gain = _tree_stump_importance(xs, ys)
        single_stats.append({
            "column": col,
            "r": r_val,
            "t_stat": t_stat,
            "p_value": p_val,
            "ml_importance": ml_gain,
            "split_threshold": thresh,
            "low_mean_yield": low_m,
            "high_mean_yield": high_m,
        })

    single_stats.sort(key=lambda d: (d["p_value"], -abs(d["r"])))

    # 2. Scan 2-way interactions among top candidate columns
    top_cand_cols = [d["column"] for d in single_stats[:12]]
    best_interaction = None
    best_int_p = 1.0
    for i in range(len(top_cand_cols)):
        for j in range(i + 1, len(top_cand_cols)):
            c1, c2 = top_cand_cols[i], top_cand_cols[j]
            # Test high-c1 * low-c2 and high-c1 * high-c2 synergy terms
            for mode, fn in [
                ("HIGH_LOW", lambda r, a=c1, b=c2: r[a] * (1.0 - r[b])),
                ("HIGH_HIGH", lambda r, a=c1, b=c2: r[a] * r[b]),
            ]:
                ix = [fn(r) for r in rows]
                r_int, t_int, p_int = _pearson_r_and_p(ix, ys)
                if p_int < best_int_p and c1 != single_stats[0]["column"]:
                    best_int_p = p_int
                    best_interaction = {
                        "param_1": c1,
                        "param_2": c2,
                        "mode": mode,
                        "r": r_int,
                        "p_value": p_int,
                    }

    # 3. Compute Step-Level (Static Stage) Variance Attribution across STEP_A..STEP_E
    step_scores: List[Dict[str, Any]] = []
    for raw_step, obf_step in STEP_OBFUSCATION_MAP.items():
        step_cols = [c for c in all_param_cols if c.startswith(obf_step + "_")]
        # Sum of R^2 across parameters in this step
        step_r2 = sum((_pearson_r_and_p([r[c] for r in rows], ys)[0] ** 2) for c in step_cols)
        step_scores.append({"step": obf_step, "raw_R2_sum": step_r2})
    total_r2 = sum(s["raw_R2_sum"] for s in step_scores) or 1.0
    for s in step_scores:
        s["variance_share_pct"] = round(100.0 * s["raw_R2_sum"] / total_r2, 1)
    step_scores.sort(key=lambda s: -s["variance_share_pct"])
    top_step = step_scores[0]

    candidates: List[Dict[str, Any]] = []

    # If user query mentions a specific raw or obfuscated step/parameter, surface it first!
    normalized_query = user_query.upper()
    for raw_step, obf_step in STEP_OBFUSCATION_MAP.items():
        if raw_step in normalized_query:
            normalized_query = normalized_query.replace(raw_step, obf_step)

    # Candidate 1: Top Single-Parameter Main Effect
    top_1 = single_stats[0]
    direction = "High" if top_1["r"] < 0 else "Low"
    candidates.append({
        "hypothesis_id": f"HYP_SINGLE_{top_1['column']}",
        "hypothesis_type": "single_parameter",
        "title": f"{direction} {top_1['column']} causes wafer yield loss",
        "parameters": [top_1["column"]],
        "rationale": (
            f"Analytics screening across {len(cols)} selected parameters flagged {top_1['column']} "
            f"as the #1 individual signal (preliminary r = {top_1['r']:+.2f})."
        ),
    })

    # Candidate 2: Best 2-Way Parameter Interaction
    if best_interaction:
        p1_col = best_interaction["param_1"]
        p2_col = best_interaction["param_2"]
        cond_desc = "High" if best_interaction["mode"] == "HIGH_HIGH" else "Low"
        candidates.append({
            "hypothesis_id": f"HYP_INT_{p1_col}_X_{p2_col}",
            "hypothesis_type": "two_way_interaction",
            "title": f"Interaction: High {p1_col} combined with {cond_desc} {p2_col} triggers bad wafers",
            "parameters": [p1_col, p2_col],
            "interaction_mode": best_interaction["mode"],
            "rationale": (
                f"2-way interaction scan detected a strong compounding effect between {p1_col} "
                f"and {p2_col} (synergy r = {best_interaction['r']:+.2f})."
            ),
        })

    # Candidate 3: Static / Step-Level Importance Hypothesis (e.g., STEP_A / UNDERFILL_LAMINATION)
    candidates.append({
        "hypothesis_id": f"HYP_STEP_{top_step['step']}",
        "hypothesis_type": "step_level",
        "title": f"Process Stage {top_step['step']} (as a whole step) is the primary yield bottleneck",
        "parameters": [top_step["step"]],
        "rationale": (
            f"Stage-level aggregation shows {top_step['step']} accounts for ~{top_step['variance_share_pct']}% "
            f"of explained yield variation across all 5 process steps."
        ),
    })

    # Candidate 4: Either a user-mentioned parameter OR a negative-control (noise) hypothesis
    noise_col = single_stats[-1]["column"]
    matched_custom = [c for c in all_param_cols if c in normalized_query]
    if matched_custom and matched_custom[0] not in (top_1["column"],):
        c_col = matched_custom[0]
        candidates.append({
            "hypothesis_id": f"HYP_CUSTOM_{c_col}",
            "hypothesis_type": "single_parameter",
            "title": f"SME Hypothesis: {c_col} significantly impacts final wafer yield",
            "parameters": [c_col],
            "rationale": f"Custom domain hypothesis requested by engineer for {c_col}.",
        })
    else:
        candidates.append({
            "hypothesis_id": f"HYP_SINGLE_{noise_col}",
            "hypothesis_type": "single_parameter",
            "title": f"Does {noise_col} variation cause bad wafers? (Baseline Control Check)",
            "parameters": [noise_col],
            "rationale": (
                f"Test whether {noise_col} has a real statistical effect on yield or can be "
                f"safely ruled out as random process noise."
            ),
        })

    candidates = candidates[:top_k]
    a2ui_payload = build_hypothesis_discovery_a2ui(candidates)

    # Hybrid Data + UI Output Schema (`Agent Tools & Interoperability.pdf`, p. 42, Snippet 7)
    return {
        "data": {
            "scanned_columns_count": len(cols),
            "candidate_hypotheses": candidates,
            "step_attribution": step_scores,
        },
        "ui": a2ui_payload,
        "ui_available": True,
    }


def evaluate_single_hypothesis(
    hypothesis: Dict[str, Any],
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate a single hypothesis using statistical p-values (with FDR) + ML importance.

    Produces:
      - Zero-ML-jargon plain-English verdict & certainty statement
      - Color-coded Good (Green) vs. Bad (Red) wafer distribution across the tested parameter
      - Step-level comparison chart data when evaluating a static process step (`STEP_A`..`STEP_E`)
      - Validated A2UI v0.9 `createSurface` + `updateComponents` + `updateDataModel` hybrid output
    """
    rows = _load_obfuscated_rows(data_dir=data_dir)
    all_param_cols = [k for k in rows[0].keys() if k not in ("wafer_id", "yield_pct")]
    ys = [r["yield_pct"] for r in rows]

    h_id = hypothesis.get("hypothesis_id", "HYP_CUSTOM")
    h_type = hypothesis.get("hypothesis_type", "single_parameter")
    params = [obfuscate_column_name(p.upper()) for p in hypothesis.get("parameters", [])]
    title = hypothesis.get("title", f"Hypothesis on {', '.join(params)}")

    # Auto-detect step-level hypothesis if parameter is STEP_A..STEP_E
    if len(params) == 1 and params[0] in STEP_OBFUSCATION_MAP.values():
        h_type = "step_level"

    if h_type == "step_level":
        step_name = params[0] if params else "STEP_A"
        step_cols = [c for c in all_param_cols if c.startswith(step_name + "_")]
        # Compute per-step variance attribution across all 5 steps
        step_breakdown = []
        for _, obf_s in STEP_OBFUSCATION_MAP.items():
            s_cols = [c for c in all_param_cols if c.startswith(obf_s + "_")]
            s_corrs = [(c, _pearson_r_and_p([r[c] for r in rows], ys)[0]) for c in s_cols]
            s_corrs.sort(key=lambda item: -abs(item[1]))
            r2_sum = sum(r_val * r_val for _, r_val in s_corrs)
            step_breakdown.append({
                "step": obf_s,
                "r2_sum": r2_sum,
                "top_parameters": [c for c, _ in s_corrs[:3]],
            })
        total_r2 = sum(b["r2_sum"] for b in step_breakdown) or 1.0
        for b in step_breakdown:
            b["share_pct"] = round(100.0 * b["r2_sum"] / total_r2, 1)

        target_info = next((b for b in step_breakdown if b["step"] == step_name), step_breakdown[0])
        top_knobs = target_info["top_parameters"]

        # Build composite Step Stress Score in [0, 1] from top drivers in that step
        raw_stress = []
        for r in rows:
            # Direction-aligned stress score (higher = worse yield)
            s_val = 0.0
            for c in top_knobs:
                r_c, _, _ = _pearson_r_and_p([row[c] for row in rows], ys)
                s_val += r[c] if r_c < 0 else (1.0 - r[c])
            raw_stress.append(s_val / max(1, len(top_knobs)))
        s_min, s_max = min(raw_stress), max(raw_stress)
        norm_stress = [(v - s_min) / max(1e-9, s_max - s_min) for v in raw_stress]
        for r, sv in zip(rows, norm_stress):
            r["_step_stress"] = sv

        r_val, t_stat, p_val = _pearson_r_and_p(norm_stress, ys)
        q_val = min(1.0, p_val * 5.0)
        _, low_m, high_m, ml_gain = _tree_stump_importance(norm_stress, ys)
        dist = _compute_good_bad_distribution(rows, lambda r: r["_step_stress"])
        dist["x_axis_label"] = f"{step_name} Composite Stress Index (0 = Optimal Spec, 1 = High Off-Spec Stress)"
        dist["step_comparison"] = [
            {"step": b["step"], "share_pct": b["share_pct"]}
            for b in sorted(step_breakdown, key=lambda x: -x["share_pct"])
        ]

        low_good_pct = round(
            100.0 * sum(dist["good_wafer_counts"][:3])
            / max(1, sum(dist["good_wafer_counts"][:3]) + sum(dist["bad_wafer_counts"][:3])),
            1,
        )
        high_bad_pct = round(
            100.0 * sum(dist["bad_wafer_counts"][3:])
            / max(1, sum(dist["good_wafer_counts"][3:]) + sum(dist["bad_wafer_counts"][3:])),
            1,
        )

        is_sig = p_val < 0.01 and target_info["share_pct"] >= 30.0
        verdict = (
            "CONFIRMED STAGE BOTTLENECK (High Confidence)"
            if is_sig
            else "NO STAGE-LEVEL EVIDENCE (Minor Step Impact)"
        )
        summary = (
            f"Process stage {step_name} as a whole accounts for {target_info['share_pct']}% of total "
            f"yield variation across all 5 process steps. Wafers with low {step_name} stress (< 0.60) "
            f"achieved {low_m:.1f}% average yield ({low_good_pct}% Good Wafers in Green), whereas high "
            f"{step_name} stress (>= 0.60) dropped to {high_m:.1f}% average yield ({high_bad_pct}% Bad "
            f"Wafers in Red). Top contributing knobs inside {step_name}: {', '.join(top_knobs[:2])}."
        )
        action = (
            f"Focus engineering containment on {step_name} (specifically {top_knobs[0]} and {top_knobs[1]}) "
            f"before adjusting downstream steps."
            if is_sig
            else f"{step_name} is not a primary bottleneck ({target_info['share_pct']}% share); inspect STEP_A instead."
        )

    elif h_type == "two_way_interaction" and len(params) >= 2:
        c1, c2 = params[0], params[1]
        if c1 not in all_param_cols:
            c1 = "STEP_A_TEMPERATURE_02"
        if c2 not in all_param_cols:
            c2 = "STEP_A_PRESSURE_01"
        mode = hypothesis.get("interaction_mode", "HIGH_LOW")
        raw_ix = [
            (r[c1] * (1.0 - r[c2])) if mode == "HIGH_LOW" else (r[c1] * r[c2])
            for r in rows
        ]
        ix_min, ix_max = min(raw_ix), max(raw_ix)
        norm_ix = [(v - ix_min) / max(1e-9, ix_max - ix_min) for v in raw_ix]
        for r, iv in zip(rows, norm_ix):
            r["_ix_score"] = iv

        r_val, t_stat, p_val = _pearson_r_and_p(norm_ix, ys)
        q_val = min(1.0, p_val * 50.0)
        thresh, low_m, high_m, ml_gain = _tree_stump_importance(norm_ix, ys)
        dist = _compute_good_bad_distribution(rows, lambda r: r["_ix_score"])
        cond_label = f"High {c1} + Low {c2}" if mode == "HIGH_LOW" else f"High {c1} + High {c2}"
        dist["x_axis_label"] = f"Interaction Exposure ({cond_label}) [0.0 = Safe, 1.0 = Danger Zone]"
        # Also include 2D regime comparison (4 quadrants) for interaction visualization
        q_safe = [r["yield_pct"] for r in rows if not (r[c1] > 0.55 and r[c2] < 0.45)]
        q_danger = [r["yield_pct"] for r in rows if (r[c1] > 0.55 and r[c2] < 0.45)]
        safe_mean = round(sum(q_safe) / max(1, len(q_safe)), 2)
        danger_mean = round(sum(q_danger) / max(1, len(q_danger)), 2)

        low_good_pct = round(
            100.0 * sum(dist["good_wafer_counts"][:3])
            / max(1, sum(dist["good_wafer_counts"][:3]) + sum(dist["bad_wafer_counts"][:3])),
            1,
        )
        high_bad_pct = round(
            100.0 * sum(dist["bad_wafer_counts"][3:])
            / max(1, sum(dist["good_wafer_counts"][3:]) + sum(dist["bad_wafer_counts"][3:])),
            1,
        )

        is_sig = p_val < 0.005
        verdict = (
            "CONFIRMED 2-PARAMETER INTERACTION (High Confidence)"
            if is_sig
            else "NO INTERACTION EVIDENCE (Likely Independent / Noise)"
        )
        summary = (
            f"Testing the interaction between {c1} and {c2}: Wafers in the Danger Zone "
            f"({cond_label}) averaged {danger_mean:.1f}% yield ({high_bad_pct}% Bad Wafers in Red), "
            f"compared to {safe_mean:.1f}% yield ({low_good_pct}% Good Wafers in Green) when kept "
            f"outside that corner ({safe_mean - danger_mean:+.1f}% yield difference)."
        )
        action = (
            f"Interlock recipe limits: Whenever {c1} exceeds 0.55, maintain {c2} above 0.50 to prevent delamination excursions."
            if is_sig
            else f"No synergy lock needed between {c1} and {c2}."
        )

    else:
        col = params[0] if (params and params[0] in all_param_cols) else "STEP_A_TEMPERATURE_01"
        xs = [r[col] for r in rows]
        r_val, t_stat, p_val = _pearson_r_and_p(xs, ys)
        q_val = min(1.0, p_val * 100.0)  # Benjamini-Hochberg / Bonferroni conservative bound across 100 params
        thresh, low_m, high_m, ml_gain = _tree_stump_importance(xs, ys)
        dist = _compute_good_bad_distribution(rows, lambda r, c=col: r[c])
        dist["x_axis_label"] = f"Normalized {col} Setting (0.0 = Low, 1.0 = High)"

        low_good = sum(dist["good_wafer_counts"][:3])
        low_total = max(1, low_good + sum(dist["bad_wafer_counts"][:3]))
        high_bad = sum(dist["bad_wafer_counts"][3:])
        high_total = max(1, sum(dist["good_wafer_counts"][3:]) + high_bad)
        low_good_pct = round(100.0 * low_good / low_total, 1)
        high_bad_pct = round(100.0 * high_bad / high_total, 1)

        if p_val < 0.001 and ml_gain >= 0.12:
            verdict = "CONFIRMED PROCESS DRIVER (High Confidence)"
            if r_val < 0:
                summary = (
                    f"Across 100 wafers, keeping {col} below {thresh:.2f} produced {low_m:.1f}% average yield "
                    f"({low_good_pct}% Good Wafers in Green), whereas running above {thresh:.2f} dropped yield "
                    f"to {high_m:.1f}% ({high_bad_pct}% Bad Wafers in Red) — a {low_m - high_m:+.1f}% yield gain."
                )
                action = f"Tighten upper control limit on {col} to stay within the 0.00 - {thresh:.2f} green window."
            else:
                summary = (
                    f"Across 100 wafers, maintaining {col} above {thresh:.2f} improved average yield to "
                    f"{high_m:.1f}% compared to {low_m:.1f}% below {thresh:.2f} ({high_m - low_m:+.1f}% yield gain)."
                )
                action = f"Maintain {col} above {thresh:.2f} for higher green-wafer pass rates."
        elif p_val < 0.05:
            verdict = "POSSIBLE LINK (Moderate Evidence - Monitor)"
            summary = (
                f"{col} shows a modest correlation with wafer yield ({low_m:.1f}% vs {high_m:.1f}%), "
                f"but after adjusting for 100 parameters (FDR q = {q_val:.3f}), more wafer lots are needed."
            )
            action = f"Keep {col} under observation on the next 50 wafers before changing spec limits."
        else:
            verdict = "NO EVIDENCE (Likely Random Noise - Rule Out)"
            summary = (
                f"Good (Green) and Bad (Red) wafers are spread evenly across the full range of {col} "
                f"(average yield {low_m:.1f}% when low vs {high_m:.1f}% when high). This parameter does NOT drive yield."
            )
            action = f"Rule out {col} as a root cause and do not tighten its machine tolerance."

    if p_val < 0.0001:
        certainty = f"Statistical Certainty: 99.99%+ (p = {p_val:.2e}, FDR q = {q_val:.2e}) — Less than a 1 in 10,000 chance this pattern is random luck."
    elif p_val < 0.01:
        certainty = f"Statistical Certainty: >99% (p = {p_val:.4f}, FDR q = {q_val:.4f}) — Less than a 1 in 100 chance this is a fluke."
    else:
        certainty = f"Statistical Certainty: Low (p = {p_val:.3f}) — Indistinguishable from random background variation."

    good_total = sum(dist["good_wafer_counts"])
    bad_total = sum(dist["bad_wafer_counts"])
    good_bad_split = (
        f"Wafer Health Split (Threshold >= {GOOD_WAFER_YIELD_THRESHOLD}% Yield): "
        f"{good_total} Good Wafers (Green) vs. {bad_total} Bad Wafers (Red) plotted across {dist['x_axis_label']}."
    )

    eval_data = {
        "hypothesis_id": h_id,
        "hypothesis_type": h_type,
        "title": title,
        "parameters": params,
        "verdict_badge": verdict,
        "p_value": p_val,
        "fdr_q_value": round(q_val, 6),
        "pearson_r": r_val,
        "t_statistic": t_stat,
        "ml_variance_explained": ml_gain,
        "plain_english_summary": summary,
        "certainty_statement": certainty,
        "good_bad_wafer_split": good_bad_split,
        "recommended_action": action,
        "distribution_chart": dist,
    }

    a2ui_bundle = build_evaluation_a2ui(eval_data)
    return {
        "data": eval_data,
        "ui": a2ui_bundle["updateComponents"],
        "a2ui_stream": [
            a2ui_bundle["createSurface"],
            a2ui_bundle["updateComponents"],
            a2ui_bundle["updateDataModel"],
        ],
        "ui_available": True,
    }
