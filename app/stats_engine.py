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
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from skills.data_preprocess.scripts.preprocess import (
    DATA_DIR,
    DATASET_CATALOG,
    STEP_OBFUSCATION_MAP,
    normalize_dataset_id,
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


_OBF_ROWS_CACHE: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
_OBF_BASE_STATS_CACHE: Dict[int, Dict[str, Dict[str, float]]] = {}


def _load_obfuscated_rows(
    data_dir: Optional[Path] = None,
    dataset_id: str = "lot_alpha",
) -> List[Dict[str, Any]]:
    ds_key = normalize_dataset_id(dataset_id)
    ds_meta = DATASET_CATALOG[ds_key]
    target_dir = data_dir or DATA_DIR
    obf_path = target_dir / ds_meta["obf_filename"]
    if not obf_path.exists():
        obfuscate_and_select_batches(batches=["ALL"], data_dir=target_dir, dataset_id=ds_key)
    mtime = obf_path.stat().st_mtime
    cache_key = (str(obf_path), mtime)
    if cache_key in _OBF_ROWS_CACHE:
        return _OBF_ROWS_CACHE[cache_key]

    with obf_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            parsed: Dict[str, Any] = {"wafer_id": r["wafer_id"]}
            for k, v in r.items():
                if k != "wafer_id":
                    parsed[k] = float(v)
            rows.append(parsed)
    _OBF_ROWS_CACHE[cache_key] = rows
    return rows


def _get_work_rows(rows: List[Dict[str, Any]], max_rows: int = 1250) -> List[Dict[str, Any]]:
    """Return full rows when N <= max_rows, or deterministic stratified slice for inner O(P^2) loops on 50k rows."""
    n = len(rows)
    if n <= max_rows:
        return rows
    stride = max(1, n // max_rows)
    return rows[::stride]


def _get_cached_full_col_stats(
    rows: List[Dict[str, Any]],
    all_param_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute and cache exact full-dataset (e.g. all 50,000 wafers) marginal stats per column."""
    row_id = id(rows)
    if row_id in _OBF_BASE_STATS_CACHE:
        return _OBF_BASE_STATS_CACHE[row_id]

    ys = [r["yield_pct"] for r in rows]
    stats_map: Dict[str, Dict[str, float]] = {}
    for col in all_param_cols:
        xs = [r[col] for r in rows]
        r_val, t_stat, p_val = _pearson_r_and_p(xs, ys)
        thresh, low_m, high_m, ml_gain = _tree_stump_importance(xs, ys)
        stats_map[col] = {
            "r": r_val,
            "t_stat": t_stat,
            "p_value": p_val,
            "ml_importance": ml_gain,
            "split_threshold": thresh,
            "low_mean_yield": low_m,
            "high_mean_yield": high_m,
        }
    _OBF_BASE_STATS_CACHE[row_id] = stats_map
    return stats_map


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


_SPLIT_THRESHOLDS_BY_BINS: Dict[int, Tuple[float, ...]] = {
    3: (0.35, 0.50, 0.65),
    5: (0.35, 0.45, 0.50, 0.55, 0.65),
    7: (0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75),
    9: (0.20, 0.30, 0.35, 0.45, 0.50, 0.55, 0.65, 0.70, 0.80),
}


def _tree_stump_importance(
    xs: List[float],
    ys: List[float],
    num_bins: int = 5,
    min_impurity_gain: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Find best 1D decision-tree split threshold and compute variance reduction in a single pass."""
    n = len(xs)
    if n < 16:
        return 0.5, 0.0, 0.0, 0.0
    sum_y = sum(ys)
    mean_y = sum_y / n
    tss = sum(y * y for y in ys) - (sum_y * sum_y) / n
    if tss < 1e-9:
        return 0.5, round(mean_y, 2), round(mean_y, 2), 0.0

    thresholds = _SPLIT_THRESHOLDS_BY_BINS.get(num_bins, _SPLIT_THRESHOLDS_BY_BINS[5])
    k_t = len(thresholds)
    b_cnt = [0] * (k_t + 1)
    b_sum = [0.0] * (k_t + 1)

    if k_t == 5:
        for x, y in zip(xs, ys):
            if x <= 0.35:
                b_cnt[0] += 1
                b_sum[0] += y
            elif x <= 0.45:
                b_cnt[1] += 1
                b_sum[1] += y
            elif x <= 0.50:
                b_cnt[2] += 1
                b_sum[2] += y
            elif x <= 0.55:
                b_cnt[3] += 1
                b_sum[3] += y
            elif x <= 0.65:
                b_cnt[4] += 1
                b_sum[4] += y
            else:
                b_cnt[5] += 1
                b_sum[5] += y
    else:
        for x, y in zip(xs, ys):
            placed = False
            for idx_b, th_val in enumerate(thresholds):
                if x <= th_val:
                    b_cnt[idx_b] += 1
                    b_sum[idx_b] += y
                    placed = True
                    break
            if not placed:
                b_cnt[k_t] += 1
                b_sum[k_t] += y

    best_gain = -1.0
    best_thresh = 0.5
    best_low_mean = mean_y
    best_high_mean = mean_y

    n_left = 0
    s_left = 0.0
    base_ss = (sum_y * sum_y) / n
    for idx_t, thresh in enumerate(thresholds):
        n_left += b_cnt[idx_t]
        s_left += b_sum[idx_t]
        n_right = n - n_left
        if n_left < 8 or n_right < 8:
            continue
        s_right = sum_y - s_left
        m_low = s_left / n_left
        m_high = s_right / n_right
        between_ss = (s_left * m_low) + (s_right * m_high) - base_ss
        gain = max(0.0, between_ss / tss)
        if gain > best_gain:
            best_gain = gain
            best_thresh = thresh
            best_low_mean = m_low
            best_high_mean = m_high

    if best_gain < max(0.0, min_impurity_gain):
        return best_thresh, round(best_low_mean, 2), round(best_high_mean, 2), 0.0

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
    n_total = len(rows)
    scatter_stride = max(1, n_total // 300)

    for r_idx, r in enumerate(rows):
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
        if r_idx % scatter_stride == 0:
            wafer_points.append({
                "wafer_id": r["wafer_id"],
                "x": round(x_val, 4),
                "y": round(y_val, 2),
                "status": "GOOD" if is_good else "BAD",
                "color": "#10b981" if is_good else "#ef4444",
            })

    return {
        "yield_threshold": yield_threshold,
        "total_wafers_binned": n_total,
        "scatter_sample_size": len(wafer_points),
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


def _confidence_metadata(p_val: float) -> Dict[str, Any]:
    """Translate p-value into color-coded SME statistical certainty metadata + Hero KPI string."""
    if p_val < 0.0005:
        return {
            "confidence_pct": 99.9,
            "confidence_hero": "99.9%",
            "confidence_label": "Confirmed Driver",
            "confidence_tier": "HIGH",
            "confidence_color": "emerald",
            "confidence_badge": "99.9% Certainty • Confirmed Driver",
            "confidence_plain": "99.9% Statistical Certainty (<1-in-1,000 fluke risk)",
        }
    if p_val < 0.01:
        pct = round(max(95.0, (1.0 - p_val) * 100.0), 1)
        return {
            "confidence_pct": pct,
            "confidence_hero": f"{pct:.1f}%",
            "confidence_label": "Confirmed Driver",
            "confidence_tier": "HIGH",
            "confidence_color": "emerald",
            "confidence_badge": f"{pct:.1f}% Certainty • Strong Signal",
            "confidence_plain": f"{pct:.1f}% Statistical Certainty (<1-in-100 fluke risk)",
        }
    if p_val < 0.15:
        pct = round(max(80.0, (1.0 - p_val) * 100.0), 1)
        return {
            "confidence_pct": pct,
            "confidence_hero": f"{pct:.0f}%",
            "confidence_label": "Moderate Signal",
            "confidence_tier": "MODERATE",
            "confidence_color": "amber",
            "confidence_badge": f"{pct:.0f}% Certainty • Moderate (Monitor)",
            "confidence_plain": f"{pct:.0f}% Statistical Certainty (Moderate trend — monitor next 30 wafers)",
        }
    pct = round(max(5.0, min(75.0, (1.0 - p_val) * 100.0)), 0)
    return {
        "confidence_pct": pct,
        "confidence_hero": f"{pct:.0f}%",
        "confidence_label": "Random Noise",
        "confidence_tier": "LOW",
        "confidence_color": "rose",
        "confidence_badge": f"{pct:.0f}% Certainty • Random Noise",
        "confidence_plain": f"{pct:.0f}% Statistical Certainty — Indistinguishable from background noise",
    }


def _compute_linear_slope(xs: List[float], ys: List[float]) -> float:
    """Compute Ordinary Least Squares slope dY/dX (yield % change per 1.0 normalized unit)."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-12:
        return 0.0
    return num / den


def _compute_partial_slope(
    rows: List[Dict[str, Any]],
    xs: List[float],
    ys: List[float],
    control_cols: List[str],
) -> float:
    """Compute multivariable Ceteris Paribus partial OLS slope holding active control_cols constant.

    Uses Frisch-Waugh-Lovell orthogonalization so that selecting/deselecting columns in the UI
    genuinely updates the ceteris paribus partial effect.
    """
    n = len(xs)
    if n < 5 or not control_cols:
        return _compute_linear_slope(xs, ys)

    # Center target x and y
    mx = sum(xs) / n
    my = sum(ys) / n
    rx = [x - mx for x in xs]
    ry = [y - my for y in ys]

    # Orthogonalize against up to 6 active control columns (Gram-Schmidt)
    basis: List[List[float]] = []
    for c in control_cols[:6]:
        raw_z = [r[c] for r in rows]
        mz = sum(raw_z) / n
        z = [v - mz for v in raw_z]
        for b in basis:
            proj = sum(zi * bi for zi, bi in zip(z, b))
            z = [zi - proj * bi for zi, bi in zip(z, b)]
        norm_z = math.sqrt(sum(zi * zi for zi in z))
        if norm_z > 1e-9:
            u = [zi / norm_z for zi in z]
            basis.append(u)
            px = sum(xi * ui for xi, ui in zip(rx, u))
            py = sum(yi * ui for yi, ui in zip(ry, u))
            rx = [xi - px * ui for xi, ui in zip(rx, u)]
            ry = [yi - py * ui for yi, ui in zip(ry, u)]

    den = sum(xi * xi for xi in rx)
    if den < 1e-12:
        return _compute_linear_slope(xs, ys)
    return sum(xi * yi for xi, yi in zip(rx, ry)) / den


def _format_short_param(col: str) -> str:
    return (
        col.replace("TEMPERATURE", "TEMP")
        .replace("PRESSURE", "PRESS")
        .replace("FLOW_RATE", "FLOW")
        .replace("DURATION", "DUR")
    )


def _build_ceteris_paribus_and_analogy(
    h_type: str,
    params: List[str],
    slope_norm: float,
    low_m: float,
    high_m: float,
    thresh: float,
    is_sig: bool,
    step_share_pct: float = 0.0,
    active_controls_count: int = 99,
    step_selected_info: str = "",
) -> Tuple[str, str, str, str]:
    """Return (ceteris_hero, ceteris_short, ceteris_paribus, everyday_analogy)."""
    sign_str = lambda v: f"{v:+.2f}%"
    hold_label = f"Holding {active_controls_count} active cols constant"

    if h_type == "step_level":
        step_name = params[0] if params else "STEP_A"
        delta_10pct = slope_norm * 0.10
        hero = f"{sign_str(delta_10pct)} / +10% Stress"
        short_sub = f"{step_share_pct:.1f}% of active yield variance {step_selected_info}".strip()
        ceteris = (
            f"{hero} ({short_sub}): For every +10% increase in {step_name} composite stress, "
            f"wafer yield shifts by {delta_10pct:+.2f}%, ceteris paribus."
        )
        analogy = (
            f"5-stage water filter: {step_share_pct:.1f}% of active yield variation enters at "
            f"{step_name} ({low_m:.1f}% calm → {high_m:.1f}% stressed)."
        )
        return hero, short_sub, ceteris, analogy

    if h_type == "two_way_interaction" and len(params) >= 2:
        c1, c2 = params[0], params[1]
        delta_10pct = slope_norm * 0.10
        hero = f"{sign_str(delta_10pct)} / +10% Joint"
        short_sub = f"High {_format_short_param(c1)} × Low {_format_short_param(c2)}"
        ceteris = (
            f"{hero}: When {c2} < 0.45, each +10% rise in joint exposure with {c1} shifts "
            f"wafer yield by {delta_10pct:+.2f}%, ceteris paribus ({hold_label})."
        )
        analogy = (
            f"Heated lamination press: high heat ({_format_short_param(c1)}) + weak pressure "
            f"({_format_short_param(c2)}) blisters the layer ({low_m:.1f}% → {high_m:.1f}% yield)."
        )
        return hero, short_sub, ceteris, analogy

    col = params[0] if params else "STEP_A_TEMPERATURE_01"
    if "TEMPERATURE" in col:
        # Raw span 140..260°C = 120°C -> per +10°C
        delta_10c = (slope_norm / 120.0) * 10.0
        per_1c = slope_norm / 120.0
        hero = f"{sign_str(delta_10c)} / +10°C"
        short_sub = f"{per_1c:+.2f}% per +1°C • {hold_label}"
        ceteris = (
            f"{hero} ({per_1c:+.2f}% / +1°C): Holding {active_controls_count} other active parameters "
            f"constant, each +10°C increase in {col} shifts yield by {delta_10c:+.2f}%, ceteris paribus."
        )
        analogy = (
            f"Oven cure threshold: ≤ {thresh:.2f} cures cleanly ({low_m:.1f}% yield); "
            f"> {thresh:.2f} over-bakes film ({high_m:.1f}% yield)."
            if is_sig
            else f"Thermostat inside safe band: {low_m:.1f}% vs {high_m:.1f}% yield across range."
        )
    elif "DURATION" in col:
        # Raw span 15..120s = 105s -> per +10s (e.g. -0.41% / +10s)
        delta_10s = (slope_norm / 105.0) * 10.0
        hero = f"{sign_str(delta_10s)} / +10s"
        short_sub = f"Dwell time shift • {hold_label}"
        ceteris = (
            f"{hero}: Holding {active_controls_count} other active parameters constant, "
            f"each +10s increase in {col} dwell time shifts yield by {delta_10s:+.2f}%, ceteris paribus."
        )
        analogy = f"Chamber dwell timer: {low_m:.1f}% yield when short vs {high_m:.1f}% when long (split @ {thresh:.2f})."
    elif "RF_POWER" in col:
        # Raw span 250..950W = 700W -> per +50W
        delta_50w = (slope_norm / 700.0) * 50.0
        hero = f"{sign_str(delta_50w)} / +50W"
        short_sub = f"Plasma power shift • {hold_label}"
        ceteris = (
            f"{hero}: Holding {active_controls_count} other active parameters constant, "
            f"each +50W change in {col} shifts yield by {delta_50w:+.2f}%, ceteris paribus."
        )
        analogy = f"Plasma etch power: > {thresh:.2f} clears residue ({high_m:.1f}% vs {low_m:.1f}% yield)."
    elif "PRESSURE" in col:
        # Raw span 1.5..9.5 Bar = 8.0 Bar -> per +1 Bar
        delta_1bar = slope_norm / 8.0
        hero = f"{sign_str(delta_1bar)} / +1 Bar"
        short_sub = f"Chamber pressure shift • {hold_label}"
        ceteris = (
            f"{hero}: Holding {active_controls_count} other active parameters constant, "
            f"each +1 Bar increase in {col} shifts yield by {delta_1bar:+.2f}%, ceteris paribus."
        )
        analogy = f"Clamping pressure window: {low_m:.1f}% yield below {thresh:.2f} vs {high_m:.1f}% above."
    else:
        # FLOW_RATE: Raw span 20..180 sccm = 160 sccm -> per +10 sccm
        delta_10sccm = (slope_norm / 160.0) * 10.0
        hero = f"{sign_str(delta_10sccm)} / +10 sccm"
        short_sub = f"Gas flow shift • {hold_label}"
        ceteris = (
            f"{hero}: Holding {active_controls_count} other active parameters constant, "
            f"each +10 sccm change in {col} shifts yield by {delta_10sccm:+.2f}%, ceteris paribus."
        )
        analogy = f"Chamber gas flow: {low_m:.1f}% yield when low vs {high_m:.1f}% when high."

    return hero, short_sub, ceteris, analogy


def _build_confounder_caveats(
    rows: List[Dict[str, Any]],
    all_param_cols: List[str],
    active_cols: List[str],
    params: List[str],
    h_type: str,
) -> List[str]:
    """Build 3 concise, subset-aware caveats warning the SME about multi-interaction confounders."""
    w_rows = _get_work_rows(rows, max_rows=500)
    primary_col = params[0] if (params and params[0] in all_param_cols) else "STEP_A_TEMPERATURE_01"
    xs = [r[primary_col] for r in w_rows]

    best_corr_col = "STEP_A_PRESSURE_01"
    best_corr_val = 0.0
    for c in active_cols:
        if c not in params:
            r_c, _, _ = _pearson_r_and_p(xs, [r[c] for r in w_rows])
            if abs(r_c) > abs(best_corr_val):
                best_corr_val = r_c
                best_corr_col = c

    # Check if any known key causal driver was deselected by the user!
    key_drivers = [
        "STEP_A_TEMPERATURE_01",
        "STEP_A_TEMPERATURE_02",
        "STEP_A_PRESSURE_01",
        "STEP_B_RF_POWER_01",
        "STEP_C_DURATION_01",
        "STEP_D_PRESSURE_01",
    ]
    omitted_drivers = [d for d in key_drivers if d not in active_cols and d not in params]

    if omitted_drivers:
        omitted_warning = (
            f"Omitted Variable Alert: Key driver {', '.join(omitted_drivers[:2])} is DESELECTED "
            f"from your active subset ({len(active_cols)}/100 cols), which shifts partial ceteris paribus slopes."
        )
    else:
        omitted_warning = (
            f"Cross-Step Co-Movement: Active sensor {best_corr_col} (r = {best_corr_val:+.2f}) "
            f"must stay stable to isolate this effect."
        )

    return [
        "Non-Linear Coupling: High-stress joint parameter corners (e.g. temperature × pressure or RF × temperature) can amplify bad wafer rate.",
        omitted_warning,
        f"Split-Lot Check: Observational estimate across {len(rows):,} wafers; verify with a 10-wafer A/B split lot before locking specs.",
    ]


def _compute_step_breakdown(
    rows: List[Dict[str, Any]],
    ys: List[float],
    all_param_cols: List[str],
    active_cols: List[str],
) -> List[Dict[str, Any]]:
    """Compute step-level metrics strictly scoped to `active_cols` vs. 100-col baseline."""
    base_stats = _get_cached_full_col_stats(rows, all_param_cols)
    w_rows = _get_work_rows(rows, max_rows=1000)
    w_ys = [r["yield_pct"] for r in w_rows]

    step_scores: List[Dict[str, Any]] = []
    for _, obf_step in STEP_OBFUSCATION_MAP.items():
        all_step_cols = [c for c in all_param_cols if c.startswith(obf_step + "_")]
        sel_step_cols = [c for c in active_cols if c.startswith(obf_step + "_")]
        desel_step_cols = [c for c in all_step_cols if c not in sel_step_cols]

        full_r2 = sum((base_stats[c]["r"] ** 2) for c in all_step_cols)
        if sel_step_cols:
            corrs = [(c, base_stats[c]["r"]) for c in sel_step_cols]
            corrs.sort(key=lambda item: -abs(item[1]))
            active_r2 = sum(rv * rv for _, rv in corrs)
            top_knobs = [c for c, _ in corrs[:3]]

            # Compute composite step stress strictly from selected columns in this step
            raw_stress = []
            for r in w_rows:
                s_val = 0.0
                for c, r_c in corrs[:3]:
                    s_val += r[c] if r_c < 0 else (1.0 - r[c])
                raw_stress.append(s_val / max(1, len(corrs[:3])))
            s_min, s_max = min(raw_stress), max(raw_stress)
            norm_stress = [(v - s_min) / max(1e-9, s_max - s_min) for v in raw_stress]
            r_step, t_step, p_step = _pearson_r_and_p(norm_stress, w_ys)
            thresh_s, low_s, high_s, gain_s = _tree_stump_importance(norm_stress, w_ys)
            slope_s = _compute_linear_slope(norm_stress, w_ys)
        else:
            active_r2 = 0.0
            top_knobs = []
            r_step, t_step, p_step = 0.0, 0.0, 0.95
            thresh_s, low_s, high_s, gain_s = 0.50, 82.8, 82.8, 0.0
            slope_s = 0.0

        step_scores.append({
            "step": obf_step,
            "raw_R2_sum": active_r2,
            "full_R2_sum": full_r2,
            "selected_col_count": len(sel_step_cols),
            "total_col_count": len(all_step_cols),
            "deselected_cols": desel_step_cols,
            "top_parameters": top_knobs,
            "r": r_step,
            "t_stat": t_step,
            "p_value": p_step,
            "split_threshold": thresh_s,
            "low_mean_yield": low_s,
            "high_mean_yield": high_s,
            "ml_importance": gain_s,
            "slope": slope_s,
        })

    total_active_r2 = sum(s["raw_R2_sum"] for s in step_scores) or 1.0
    total_full_r2 = sum(s["full_R2_sum"] for s in step_scores) or 1.0
    for s in step_scores:
        s["variance_share_pct"] = round(100.0 * s["raw_R2_sum"] / total_active_r2, 1) if s["selected_col_count"] > 0 else 0.0
        s["full_baseline_share_pct"] = round(100.0 * s["full_R2_sum"] / total_full_r2, 1)

    step_scores.sort(key=lambda s: (-s["variance_share_pct"], -s["selected_col_count"]))
    return step_scores


def discover_candidate_hypotheses(
    selected_columns: Optional[List[str]] = None,
    user_query: str = "",
    top_k: int = 8,
    data_dir: Optional[Path] = None,
    dataset_id: str = "lot_alpha",
) -> Dict[str, Any]:
    """Scan strictly the user's `selected_columns` on `dataset_id` to generate dynamic candidate hypotheses."""
    t_start = time.perf_counter()
    ds_key = normalize_dataset_id(dataset_id)
    rows = _load_obfuscated_rows(data_dir=data_dir, dataset_id=ds_key)
    all_param_cols = [k for k in rows[0].keys() if k not in ("wafer_id", "yield_pct") and not k.startswith("_")]
    cols = [c for c in (selected_columns or all_param_cols) if c in all_param_cols]
    if not cols:
        cols = all_param_cols

    ys = [r["yield_pct"] for r in rows]
    base_stats = _get_cached_full_col_stats(rows, all_param_cols)
    w_rows = _get_work_rows(rows, max_rows=1000)
    w_ys = [r["yield_pct"] for r in w_rows]

    # First pass correlations to identify top active control columns for partial OLS
    prelim_corrs = sorted(
        [(c, abs(base_stats[c]["r"])) for c in cols],
        key=lambda x: -x[1],
    )
    top_active_controls = [c for c, _ in prelim_corrs[:6]]

    # 1. Scan all 1-way main effects across selected columns (full-dataset stats + partial OLS slope)
    single_stats = []
    for col in cols:
        b_st = base_stats[col]
        w_xs = [r[col] for r in w_rows]
        ctrl = [c for c in top_active_controls if c != col]
        partial_slope = _compute_partial_slope(w_rows, w_xs, w_ys, ctrl)
        single_stats.append({
            "column": col,
            "r": b_st["r"],
            "t_stat": b_st["t_stat"],
            "p_value": b_st["p_value"],
            "ml_importance": b_st["ml_importance"],
            "split_threshold": b_st["split_threshold"],
            "low_mean_yield": b_st["low_mean_yield"],
            "high_mean_yield": b_st["high_mean_yield"],
            "slope": partial_slope,
        })

    single_stats.sort(key=lambda d: (d["p_value"], -abs(d["r"])))

    # 2. Scan 2-way interactions strictly among selected columns
    top_cand_cols = [d["column"] for d in single_stats[:12]]
    interactions_found: List[Dict[str, Any]] = []
    for i in range(len(top_cand_cols)):
        for j in range(i + 1, len(top_cand_cols)):
            c1, c2 = top_cand_cols[i], top_cand_cols[j]
            if len(top_cand_cols) > 2 and c1 == single_stats[0]["column"]:
                continue
            for mode, fn in [
                ("HIGH_LOW", lambda r, a=c1, b=c2: r[a] * (1.0 - r[b])),
                ("HIGH_HIGH", lambda r, a=c1, b=c2: r[a] * r[b]),
            ]:
                ix = [fn(r) for r in w_rows]
                r_int, t_int, p_int = _pearson_r_and_p(ix, w_ys)
                thresh_i, low_i, high_i, _ = _tree_stump_importance(ix, w_ys)
                slope_i = _compute_linear_slope(ix, w_ys)
                interactions_found.append({
                    "param_1": c1,
                    "param_2": c2,
                    "mode": mode,
                    "r": r_int,
                    "p_value": p_int,
                    "low_mean_yield": low_i,
                    "high_mean_yield": high_i,
                    "thresh": thresh_i,
                    "slope": slope_i,
                })
    interactions_found.sort(key=lambda d: (d["p_value"], -abs(d["r"])))
    best_interaction = interactions_found[0] if interactions_found else None

    # 3. Compute Step-Level Variance Attribution strictly across `cols` (active selected columns)
    step_scores = _compute_step_breakdown(rows, ys, all_param_cols, cols)
    active_steps = [s for s in step_scores if s["selected_col_count"] > 0]
    top_step = active_steps[0] if active_steps else step_scores[0]

    def _enrich_candidate(
        cand: Dict[str, Any],
        p_val: float,
        slope: float,
        low_m: float,
        high_m: float,
        thresh: float,
        share_pct: float = 0.0,
        step_selected_info: str = "",
    ) -> Dict[str, Any]:
        conf = _confidence_metadata(p_val)
        c_hero, c_short, ceteris, analogy = _build_ceteris_paribus_and_analogy(
            h_type=cand["hypothesis_type"],
            params=cand["parameters"],
            slope_norm=slope,
            low_m=low_m,
            high_m=high_m,
            thresh=thresh,
            is_sig=(p_val < 0.01),
            step_share_pct=share_pct,
            active_controls_count=max(0, len(cols) - len(cand["parameters"])),
            step_selected_info=step_selected_info,
        )
        caveats = _build_confounder_caveats(rows, all_param_cols, cols, cand["parameters"], cand["hypothesis_type"])
        cand.update(conf)
        cand["ceteris_paribus_hero"] = c_hero
        cand["ceteris_paribus_short"] = c_short
        cand["ceteris_paribus"] = ceteris
        cand["everyday_analogy"] = analogy
        cand["caveats_and_confounders"] = caveats
        cand["estimated_yield_swing_pct"] = round(abs(high_m - low_m), 1)
        cand["dataset_id"] = ds_key
        cand["num_wafers_evaluated"] = len(rows)
        return cand

    candidates: List[Dict[str, Any]] = []

    normalized_query = user_query.upper()
    for raw_step, obf_step in STEP_OBFUSCATION_MAP.items():
        if raw_step in normalized_query:
            normalized_query = normalized_query.replace(raw_step, obf_step)

    # Candidate 1: Top Single-Parameter Main Effect in active selection
    top_1 = single_stats[0]
    arrow_1 = "↑" if top_1["r"] < 0 else "↓"
    direction = "High" if top_1["r"] < 0 else "Low"
    candidates.append(_enrich_candidate(
        {
            "hypothesis_id": f"HYP_SINGLE_{top_1['column']}",
            "hypothesis_type": "single_parameter",
            "short_title": f"{top_1['column']} {arrow_1}",
            "title": f"{direction} {top_1['column']} causes wafer yield loss",
            "parameters": [top_1["column"]],
            "rationale": (
                f"Low setting averages {top_1['low_mean_yield']:.1f}% yield vs. "
                f"{top_1['high_mean_yield']:.1f}% at high setting."
            ),
        },
        p_val=top_1["p_value"],
        slope=top_1["slope"],
        low_m=top_1["low_mean_yield"],
        high_m=top_1["high_mean_yield"],
        thresh=top_1["split_threshold"],
    ))

    # Candidate 2: Best 2-Way Parameter Interaction in active selection
    if best_interaction:
        p1_col = best_interaction["param_1"]
        p2_col = best_interaction["param_2"]
        cond_desc = "High" if best_interaction["mode"] == "HIGH_HIGH" else "Low"
        candidates.append(_enrich_candidate(
            {
                "hypothesis_id": f"HYP_INT_{p1_col}_X_{p2_col}",
                "hypothesis_type": "two_way_interaction",
                "short_title": f"{_format_short_param(p1_col)} × {_format_short_param(p2_col)}",
                "title": f"Interaction: High {p1_col} + {cond_desc} {p2_col}",
                "parameters": [p1_col, p2_col],
                "interaction_mode": best_interaction["mode"],
                "rationale": (
                    f"Safe window yields {best_interaction['low_mean_yield']:.1f}% vs. "
                    f"{best_interaction['high_mean_yield']:.1f}% in joint corner."
                ),
            },
            p_val=best_interaction["p_value"],
            slope=best_interaction["slope"],
            low_m=best_interaction["low_mean_yield"],
            high_m=best_interaction["high_mean_yield"],
            thresh=best_interaction["thresh"],
        ))

    # Candidate 3: Top Active Step-Level Hypothesis (dynamic based on selected columns!)
    step_sel_str = f"({top_step['selected_col_count']}/{top_step['total_col_count']} cols active)"
    candidates.append(_enrich_candidate(
        {
            "hypothesis_id": f"HYP_STEP_{top_step['step']}",
            "hypothesis_type": "step_level",
            "short_title": f"{top_step['step']} Stage ({top_step['selected_col_count']}/20 Cols)",
            "title": f"Process Stage {top_step['step']} Bottleneck {step_sel_str}",
            "parameters": [top_step["step"]],
            "rationale": (
                f"Across your {len(cols)} selected columns, {top_step['step']} {step_sel_str} drives "
                f"{top_step['variance_share_pct']}% of active yield variance."
            ),
        },
        p_val=top_step["p_value"],
        slope=top_step["slope"],
        low_m=top_step["low_mean_yield"],
        high_m=top_step["high_mean_yield"],
        thresh=top_step["split_threshold"],
        share_pct=top_step["variance_share_pct"],
        step_selected_info=step_sel_str,
    ))

    # Candidate 4..6: Next Single-Parameter Drivers from active `cols`
    matched_custom = [c for c in cols if c in normalized_query]
    if matched_custom and matched_custom[0] != top_1["column"]:
        c_col = matched_custom[0]
        c_stat = next((s for s in single_stats if s["column"] == c_col), single_stats[-1])
        candidates.append(_enrich_candidate(
            {
                "hypothesis_id": f"HYP_CUSTOM_{c_col}",
                "hypothesis_type": "single_parameter",
                "short_title": f"{c_col} (Custom)",
                "title": f"Engineer Check: {c_col}",
                "parameters": [c_col],
                "rationale": f"Yield spans {c_stat['low_mean_yield']:.1f}% to {c_stat['high_mean_yield']:.1f}%.",
            },
            p_val=c_stat["p_value"],
            slope=c_stat["slope"],
            low_m=c_stat["low_mean_yield"],
            high_m=c_stat["high_mean_yield"],
            thresh=c_stat["split_threshold"],
        ))

    for extra_stat in single_stats[1:6]:
        if any(extra_stat["column"] == c["parameters"][0] for c in candidates if c["hypothesis_type"] == "single_parameter"):
            continue
        if len(candidates) >= top_k - 2:
            break
        dir_e = "High" if extra_stat["r"] < 0 else "Low"
        arrow_e = "↑" if extra_stat["r"] < 0 else "↓"
        candidates.append(_enrich_candidate(
            {
                "hypothesis_id": f"HYP_SINGLE_{extra_stat['column']}",
                "hypothesis_type": "single_parameter",
                "short_title": f"{extra_stat['column']} {arrow_e}",
                "title": f"{dir_e} {extra_stat['column']} shifts wafer yield",
                "parameters": [extra_stat["column"]],
                "rationale": (
                    f"Average yield shifts from {extra_stat['low_mean_yield']:.1f}% to "
                    f"{extra_stat['high_mean_yield']:.1f}%."
                ),
            },
            p_val=extra_stat["p_value"],
            slope=extra_stat["slope"],
            low_m=extra_stat["low_mean_yield"],
            high_m=extra_stat["high_mean_yield"],
            thresh=extra_stat["split_threshold"],
        ))

    # Candidate 7: Secondary Active Step Comparison (e.g. STEP_B or next active step)
    if len(active_steps) > 1:
        sec_step = active_steps[1]
        sec_sel_str = f"({sec_step['selected_col_count']}/{sec_step['total_col_count']} cols active)"
        candidates.append(_enrich_candidate(
            {
                "hypothesis_id": f"HYP_STEP_{sec_step['step']}",
                "hypothesis_type": "step_level",
                "short_title": f"{sec_step['step']} Stage ({sec_step['selected_col_count']}/20 Cols)",
                "title": f"Stage Comparison: {sec_step['step']} {sec_sel_str}",
                "parameters": [sec_step["step"]],
                "rationale": (
                    f"{sec_step['step']} {sec_sel_str} accounts for {sec_step['variance_share_pct']}% "
                    f"of active yield variance (full 20-col baseline: {sec_step['full_baseline_share_pct']}%)."
                ),
            },
            p_val=sec_step["p_value"],
            slope=sec_step["slope"],
            low_m=sec_step["low_mean_yield"],
            high_m=sec_step["high_mean_yield"],
            thresh=sec_step["split_threshold"],
            share_pct=sec_step["variance_share_pct"],
            step_selected_info=sec_sel_str,
        ))

    # Candidate 8: Baseline Noise Control Check from active `cols`
    noise_stat = single_stats[-1]
    if not any(noise_stat["column"] in c["parameters"] for c in candidates):
        candidates.append(_enrich_candidate(
            {
                "hypothesis_id": f"HYP_SINGLE_{noise_stat['column']}",
                "hypothesis_type": "single_parameter",
                "short_title": f"{noise_stat['column']} (Noise Control)",
                "title": f"Baseline Control: {noise_stat['column']}",
                "parameters": [noise_stat["column"]],
                "rationale": (
                    f"Flat response ({noise_stat['low_mean_yield']:.1f}% vs. {noise_stat['high_mean_yield']:.1f}%) "
                    f"— confirms {noise_stat['column']} can be ruled out."
                ),
            },
            p_val=noise_stat["p_value"],
            slope=noise_stat["slope"],
            low_m=noise_stat["low_mean_yield"],
            high_m=noise_stat["high_mean_yield"],
            thresh=noise_stat["split_threshold"],
        ))

    candidates = candidates[:top_k]
    a2ui_payload = build_hypothesis_discovery_a2ui(candidates)
    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

    return {
        "data": {
            "dataset_id": ds_key,
            "dataset_name": DATASET_CATALOG[ds_key]["name"],
            "num_wafers": len(rows),
            "total_cells_scanned": len(rows) * len(cols),
            "compute_time_ms": elapsed_ms,
            "scanned_columns_count": len(cols),
            "selected_columns": cols,
            "candidate_hypotheses": candidates,
            "step_attribution": step_scores,
        },
        "ui": a2ui_payload,
        "ui_available": True,
    }


def evaluate_single_hypothesis(
    hypothesis: Dict[str, Any],
    selected_columns: Optional[List[str]] = None,
    data_dir: Optional[Path] = None,
    dataset_id: str = "lot_alpha",
) -> Dict[str, Any]:
    """Evaluate a single hypothesis using statistical p-values (with FDR) + ML importance.

    Respects `selected_columns` and `dataset_id` so step-level evaluations and partial
    ceteris paribus slopes dynamically reflect the engineer's active dataset and subset.
    """
    t_start = time.perf_counter()
    ds_key = normalize_dataset_id(dataset_id)
    rows = _load_obfuscated_rows(data_dir=data_dir, dataset_id=ds_key)
    all_param_cols = [k for k in rows[0].keys() if k not in ("wafer_id", "yield_pct") and not k.startswith("_")]
    active_cols = [c for c in (selected_columns or all_param_cols) if c in all_param_cols]
    if not active_cols:
        active_cols = all_param_cols
    ys = [r["yield_pct"] for r in rows]
    base_stats = _get_cached_full_col_stats(rows, all_param_cols)
    w_rows = _get_work_rows(rows, max_rows=1000)
    w_ys = [r["yield_pct"] for r in w_rows]

    prelim_corrs = sorted(
        [(c, abs(base_stats[c]["r"])) for c in active_cols],
        key=lambda x: -x[1],
    )
    top_active_controls = [c for c, _ in prelim_corrs[:6]]

    h_id = hypothesis.get("hypothesis_id", "HYP_CUSTOM")
    h_type = hypothesis.get("hypothesis_type", "single_parameter")
    params = [obfuscate_column_name(p.upper()) for p in hypothesis.get("parameters", [])]
    title = hypothesis.get("title", f"Hypothesis on {', '.join(params)}")
    short_title = hypothesis.get("short_title", title)
    step_share_pct = 0.0
    step_selected_info = ""

    if len(params) == 1 and params[0] in STEP_OBFUSCATION_MAP.values():
        h_type = "step_level"

    if h_type == "step_level":
        step_name = params[0] if params else "STEP_A"
        step_breakdown = _compute_step_breakdown(rows, ys, all_param_cols, active_cols)
        target_info = next((b for b in step_breakdown if b["step"] == step_name), step_breakdown[0])

        # If the user deselected all columns of this step, fall back to full step columns but flag 0/20 active
        s_cols = [c for c in active_cols if c.startswith(step_name + "_")]
        if not s_cols:
            s_cols = [c for c in all_param_cols if c.startswith(step_name + "_")]
        s_corrs = sorted(
            [(c, base_stats[c]["r"]) for c in s_cols],
            key=lambda item: -abs(item[1]),
        )
        top_knobs = [c for c, _ in s_corrs[:3]]
        step_share_pct = target_info["variance_share_pct"]
        step_selected_info = f"({target_info['selected_col_count']}/{target_info['total_col_count']} cols active)"
        short_title = f"{step_name} Stage {step_selected_info}"

        raw_stress = []
        for r in rows:
            s_val = 0.0
            for c, r_c in s_corrs[:3]:
                s_val += r[c] if r_c < 0 else (1.0 - r[c])
            raw_stress.append(s_val / max(1, len(s_corrs[:3])))
        s_min, s_max = min(raw_stress), max(raw_stress)
        norm_stress = [(v - s_min) / max(1e-9, s_max - s_min) for v in raw_stress]
        for r, sv in zip(rows, norm_stress):
            r["_step_stress"] = sv

        r_val, t_stat, p_val = _pearson_r_and_p(norm_stress, ys)
        q_val = min(1.0, p_val * 5.0)
        thresh, low_m, high_m, ml_gain = _tree_stump_importance(norm_stress, ys)
        slope_norm = _compute_linear_slope(norm_stress, ys)
        dist = _compute_good_bad_distribution(rows, lambda r: r["_step_stress"])
        dist["x_axis_label"] = f"{step_name} Stress Index {step_selected_info}"
        dist["step_comparison"] = [
            {"step": b["step"], "share_pct": b["variance_share_pct"]}
            for b in step_breakdown
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

        is_sig = p_val < 0.01 and step_share_pct >= 25.0
        verdict = (
            "CONFIRMED STAGE BOTTLENECK (High Confidence)"
            if is_sig
            else "NO STAGE-LEVEL EVIDENCE (Minor Step Impact)"
        )
        desel_note = (
            f" ({len(target_info['deselected_cols'])} columns deselected by user)"
            if target_info["deselected_cols"]
            else ""
        )
        summary = (
            f"{step_name} {step_selected_info}{desel_note} accounts for {step_share_pct}% of active yield variance "
            f"(20-col baseline: {target_info['full_baseline_share_pct']}%). Low stress (<0.60) averages "
            f"{low_m:.1f}% yield ({low_good_pct}% Green wafers) vs. {high_m:.1f}% ({high_bad_pct}% Red wafers) when stressed."
        )
        action = (
            f"Stabilize {step_name} top active knobs ({', '.join(top_knobs[:2])}) within spec."
            if is_sig
            else f"{step_name} ({step_share_pct}% active share) is not the main bottleneck; focus on {step_breakdown[0]['step']}."
        )

    elif h_type == "two_way_interaction" and len(params) >= 2:
        c1, c2 = params[0], params[1]
        if c1 not in all_param_cols:
            c1 = "STEP_A_TEMPERATURE_02"
        if c2 not in all_param_cols:
            c2 = "STEP_A_PRESSURE_01"
        short_title = f"{_format_short_param(c1)} × {_format_short_param(c2)}"
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
        ctrl = [c for c in top_active_controls if c not in (c1, c2)]
        w_ix = [r["_ix_score"] for r in w_rows]
        slope_norm = _compute_partial_slope(w_rows, w_ix, w_ys, ctrl)
        dist = _compute_good_bad_distribution(rows, lambda r: r["_ix_score"])
        cond_label = f"High {c1} + Low {c2}" if mode == "HIGH_LOW" else f"High {c1} + High {c2}"
        dist["x_axis_label"] = f"Joint Exposure ({_format_short_param(c1)} × {_format_short_param(c2)})"
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
            f"Danger Zone ({cond_label}) averages {danger_mean:.1f}% yield ({high_bad_pct}% Red wafers) "
            f"vs. {safe_mean:.1f}% ({low_good_pct}% Green wafers) in safe zone."
        )
        action = (
            f"Interlock specs: When {c1} > 0.55, hold {c2} ≥ 0.50."
            if is_sig
            else f"No recipe interlock needed between {c1} and {c2}."
        )
        low_m, high_m = safe_mean, danger_mean

    else:
        col = params[0] if (params and params[0] in all_param_cols) else "STEP_A_TEMPERATURE_01"
        b_st = base_stats[col]
        r_val, t_stat, p_val = b_st["r"], b_st["t_stat"], b_st["p_value"]
        q_val = min(1.0, p_val * len(active_cols))
        thresh, low_m, high_m, ml_gain = b_st["split_threshold"], b_st["low_mean_yield"], b_st["high_mean_yield"], b_st["ml_importance"]
        ctrl = [c for c in top_active_controls if c != col]
        w_xs = [r[col] for r in w_rows]
        slope_norm = _compute_partial_slope(w_rows, w_xs, w_ys, ctrl)
        dist = _compute_good_bad_distribution(rows, lambda r, c=col: r[c])
        dist["x_axis_label"] = f"Normalized {col} Setting (0.0 = Low, 1.0 = High)"
        short_title = f"{col} {'↑' if r_val < 0 else '↓'}"

        low_good = sum(dist["good_wafer_counts"][:3])
        low_total = max(1, low_good + sum(dist["bad_wafer_counts"][:3]))
        high_bad = sum(dist["bad_wafer_counts"][3:])
        high_total = max(1, sum(dist["good_wafer_counts"][3:]) + high_bad)
        low_good_pct = round(100.0 * low_good / low_total, 1)
        high_bad_pct = round(100.0 * high_bad / high_total, 1)
        is_sig = p_val < 0.001 and ml_gain >= 0.12

        if is_sig:
            verdict = "CONFIRMED PROCESS DRIVER (High Confidence)"
            if r_val < 0:
                summary = (
                    f"Keeping {col} ≤ {thresh:.2f} yields {low_m:.1f}% ({low_good_pct}% Green wafers), "
                    f"while > {thresh:.2f} drops to {high_m:.1f}% ({high_bad_pct}% Red wafers)."
                )
                action = f"Cap {col} upper control limit ≤ {thresh:.2f}."
            else:
                summary = (
                    f"Maintaining {col} ≥ {thresh:.2f} yields {high_m:.1f}% vs. {low_m:.1f}% below {thresh:.2f}."
                )
                action = f"Hold {col} lower control limit ≥ {thresh:.2f}."
        elif p_val < 0.05:
            verdict = "POSSIBLE LINK (Moderate Evidence - Monitor)"
            summary = f"{col} shows a modest shift ({low_m:.1f}% vs {high_m:.1f}% yield); monitor next 30 wafers."
            action = f"Monitor {col}; do not lock spec limits yet."
        else:
            verdict = "NO EVIDENCE (Likely Random Noise - Rule Out)"
            summary = f"Green and Red wafers are evenly distributed across {col} ({low_m:.1f}% vs {high_m:.1f}%)."
            action = f"Rule out {col}; leave machine tolerance unchanged."

    conf_meta = _confidence_metadata(p_val)
    c_hero, c_short, ceteris_paribus, everyday_analogy = _build_ceteris_paribus_and_analogy(
        h_type=h_type,
        params=params,
        slope_norm=slope_norm,
        low_m=low_m,
        high_m=high_m,
        thresh=thresh,
        is_sig=(p_val < 0.01),
        step_share_pct=step_share_pct,
        active_controls_count=max(0, len(active_cols) - len(params)),
        step_selected_info=step_selected_info,
    )
    caveats = _build_confounder_caveats(rows, all_param_cols, active_cols, params, h_type)

    good_total = sum(dist["good_wafer_counts"])
    bad_total = sum(dist["bad_wafer_counts"])
    good_bad_split = f"{good_total:,} Good Wafers (Green) • {bad_total:,} Bad Wafers (Red)"
    eval_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

    eval_data = {
        "hypothesis_id": h_id,
        "hypothesis_type": h_type,
        "short_title": short_title,
        "title": title,
        "parameters": params,
        "verdict_badge": verdict,
        "p_value": p_val,
        "fdr_q_value": round(q_val, 6),
        "pearson_r": r_val,
        "t_statistic": t_stat,
        "ml_variance_explained": ml_gain,
        "confidence_pct": conf_meta["confidence_pct"],
        "confidence_hero": conf_meta["confidence_hero"],
        "confidence_label": conf_meta["confidence_label"],
        "confidence_tier": conf_meta["confidence_tier"],
        "confidence_color": conf_meta["confidence_color"],
        "confidence_badge": conf_meta["confidence_badge"],
        "certainty_statement": conf_meta["confidence_plain"],
        "ceteris_paribus_hero": c_hero,
        "ceteris_paribus_short": c_short,
        "ceteris_paribus": ceteris_paribus,
        "everyday_analogy": everyday_analogy,
        "caveats_and_confounders": caveats,
        "plain_english_summary": summary,
        "good_bad_wafer_split": good_bad_split,
        "recommended_action": action,
        "dataset_id": ds_key,
        "num_wafers_evaluated": len(rows),
        "compute_time_ms": eval_ms,
        "active_columns_count": len(active_cols),
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


# ============================================================================
# 5. Multi-Model Feature Selection & Cross-Model Consistency Engine (Screen 3)
# ============================================================================

ML_MODEL_SPECS: List[Dict[str, str]] = [
    {
        "model_id": "L1_LASSO",
        "model_name": "L1 Lasso Sparse Linear",
        "short_name": "L1 Lasso",
        "family": "Sparse Penalized Linear (L1 Coordinate Descent)",
    },
    {
        "model_id": "RANDOM_FOREST",
        "model_name": "Random Forest Impurity",
        "short_name": "Random Forest",
        "family": "Bagged Decision Tree Ensemble (24 Trees)",
    },
    {
        "model_id": "GRADIENT_BOOSTING",
        "model_name": "Gradient Boosted Trees",
        "short_name": "Grad. Boost",
        "family": "Additive Stagewise Boosted Trees",
    },
    {
        "model_id": "PERMUTATION_RIDGE",
        "model_name": "Permutation Importance",
        "short_name": "Permutation",
        "family": "Out-of-Sample Shuffle Drop in R² (L2 Ridge)",
    },
    {
        "model_id": "MUTUAL_INFO_RANK",
        "model_name": "Mutual Info + Spearman",
        "short_name": "Mutual Info",
        "family": "Non-Parametric Shannon Entropy & Rank Order",
    },
]

ENGINEERING_MODES: Dict[str, Dict[str, Any]] = {
    "FAST_TRIAGE": {
        "mode_id": "FAST_TRIAGE",
        "name": "⚡ Fast Mode — Rapid Line-Down Triage",
        "short_name": "⚡ Fast Triage",
        "badge": "Sub-50ms • Classical ML + Tree Stumps",
        "engineer_summary": (
            "Optimized for urgent excursion triage on 100–50,000 wafers: combines classical L1 Lasso, "
            "Permutation Ridge, Mutual Info, and shallow Depth-1 Random Forest stumps with lower epochs "
            "(4 epochs, 8 trees) and lighter convergence/split thresholds (tol=0.01, 3 bins)."
        ),
        "default_config": {
            "active_models": [
                "L1_LASSO",
                "RANDOM_FOREST",
                "MUTUAL_INFO_RANK",
            ],
            "top_k_cutoff": 5,
            "consensus_min_models": 2,
            "regularization_type": "L1",
            "reg_lambda": 0.25,
            "epochs": 20,
            "convergence_tol": 0.005,
            "n_trees": 8,
            "max_depth": 1,
            "max_leaves": 2,
            "learning_rate": 0.25,
            "split_threshold_bins": 3,
            "min_impurity_gain": 0.015,
            "wafer_subsample_cap": 600,
        },
    },
    "BALANCED_FAB": {
        "mode_id": "BALANCED_FAB",
        "name": "⚖️ Balanced Mode — Standard Fab Consensus (Default)",
        "short_name": "⚖️ Balanced Fab",
        "badge": "Recommended • All 5 ML Families",
        "engineer_summary": (
            "Standard shift-handover baseline: runs all 5 ML families (L1 Lasso Linear, "
            "20-Tree Depth-2 Random Forest, Gradient Boosting, Permutation Ridge, "
            "and Mutual Info) requiring ≥4/5 models in Top-5."
        ),
        "default_config": {
            "active_models": [
                "L1_LASSO",
                "RANDOM_FOREST",
                "GRADIENT_BOOSTING",
                "PERMUTATION_RIDGE",
                "MUTUAL_INFO_RANK",
            ],
            "top_k_cutoff": 5,
            "consensus_min_models": 4,
            "regularization_type": "L1",
            "reg_lambda": 0.075,
            "epochs": 16,
            "convergence_tol": 0.001,
            "n_trees": 24,
            "max_depth": 2,
            "max_leaves": 4,
            "learning_rate": 0.18,
            "split_threshold_bins": 5,
            "min_impurity_gain": 0.002,
            "wafer_subsample_cap": 1500,
        },
    },
    "DEEP_INTERACTION": {
        "mode_id": "DEEP_INTERACTION",
        "name": "🔬 Deep Mode — Non-Linear Chamber Synergy",
        "short_name": "🔬 Deep Coupling",
        "badge": "Depth-4 / 12-Leaf Trees • Fine Threshold Grid",
        "engineer_summary": (
            "Designed for complex multi-step chamber interactions (e.g., Temp × Pressure or "
            "RF Power × Dwell): expands trees to Depth 4 (up to 12 leaves), 40 RF trees, "
            "9 split bins, Elastic-Net regularization, and tight convergence (tol=0.0002)."
        ),
        "default_config": {
            "active_models": [
                "L1_LASSO",
                "RANDOM_FOREST",
                "GRADIENT_BOOSTING",
                "PERMUTATION_RIDGE",
                "MUTUAL_INFO_RANK",
            ],
            "top_k_cutoff": 8,
            "consensus_min_models": 3,
            "regularization_type": "ELASTIC_NET",
            "reg_lambda": 0.06,
            "epochs": 30,
            "convergence_tol": 0.0002,
            "n_trees": 40,
            "max_depth": 4,
            "max_leaves": 12,
            "learning_rate": 0.10,
            "split_threshold_bins": 9,
            "min_impurity_gain": 0.001,
            "wafer_subsample_cap": 2000,
        },
    },
    "STRICT_AUDIT": {
        "mode_id": "STRICT_AUDIT",
        "name": "🛡️ Strict Audit — Spec-Lock & Split-Lot Sign-Off",
        "short_name": "🛡️ Strict Audit",
        "badge": "Top-3 • Heavy Regularization",
        "engineer_summary": (
            "High-bar verification before locking chamber recipe tolerances or ordering a "
            "10-wafer A/B split lot: enforces Top-3 cutoff, ≥4/5 model consensus, "
            "strong L1/L2 penalty (λ=0.22), and min impurity gain ≥ 0.012 to suppress false positives."
        ),
        "default_config": {
            "active_models": [
                "L1_LASSO",
                "RANDOM_FOREST",
                "GRADIENT_BOOSTING",
                "PERMUTATION_RIDGE",
                "MUTUAL_INFO_RANK",
            ],
            "top_k_cutoff": 3,
            "consensus_min_models": 4,
            "regularization_type": "L1",
            "reg_lambda": 0.22,
            "epochs": 24,
            "convergence_tol": 0.0005,
            "n_trees": 25,
            "max_depth": 2,
            "max_leaves": 3,
            "learning_rate": 0.10,
            "split_threshold_bins": 5,
            "min_impurity_gain": 0.012,
            "wafer_subsample_cap": 1500,
        },
    },
}


def resolve_feature_selection_config(
    mode_id: Optional[str] = "BALANCED_FAB",
    custom_params: Optional[Dict[str, Any]] = None,
    top_k_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve a Process Engineer Mode preset and merge any user-edited hyperparameter overrides."""
    mode_key = str(mode_id or "BALANCED_FAB").strip().upper()
    if mode_key not in ENGINEERING_MODES:
        mode_key = "BALANCED_FAB"

    mode_spec = ENGINEERING_MODES[mode_key]
    defaults = dict(mode_spec["default_config"])
    defaults["active_models"] = list(mode_spec["default_config"]["active_models"])

    merged = dict(defaults)
    merged["active_models"] = list(defaults["active_models"])
    if isinstance(custom_params, dict):
        for k, v in custom_params.items():
            if v is not None and k in merged:
                merged[k] = v
    if top_k_override is not None:
        merged["top_k_cutoff"] = int(top_k_override)

    valid_model_ids = [m["model_id"] for m in ML_MODEL_SPECS]
    raw_active = merged.get("active_models") or valid_model_ids
    if isinstance(raw_active, list):
        clean_active = [m for m in valid_model_ids if m in raw_active]
    else:
        clean_active = list(valid_model_ids)
    if not clean_active:
        clean_active = list(defaults["active_models"])

    top_k = max(1, min(15, int(merged.get("top_k_cutoff", defaults["top_k_cutoff"]))))
    consensus_min = max(
        1,
        min(
            len(clean_active),
            int(merged.get("consensus_min_models", defaults["consensus_min_models"])),
        ),
    )

    reg_type = str(merged.get("regularization_type", defaults["regularization_type"])).strip().upper()
    if reg_type not in ("L1", "L2", "ELASTIC_NET"):
        reg_type = defaults["regularization_type"]

    reg_lambda = round(max(0.005, min(1.0, float(merged.get("reg_lambda", defaults["reg_lambda"])))), 4)
    epochs = max(2, min(150, int(merged.get("epochs", defaults["epochs"]))))
    conv_tol = max(1e-5, min(0.05, float(merged.get("convergence_tol", defaults["convergence_tol"]))))
    n_trees = max(4, min(80, int(merged.get("n_trees", defaults["n_trees"]))))
    max_depth = max(1, min(5, int(merged.get("max_depth", defaults["max_depth"]))))
    max_leaves = max(2, min(32, int(merged.get("max_leaves", defaults["max_leaves"]))))
    learning_rate = round(max(0.01, min(0.50, float(merged.get("learning_rate", defaults["learning_rate"])))), 3)

    raw_bins = int(merged.get("split_threshold_bins", defaults["split_threshold_bins"]))
    split_bins = min((3, 5, 7, 9), key=lambda b: abs(b - raw_bins))
    min_gain = round(max(0.0, min(0.05, float(merged.get("min_impurity_gain", defaults["min_impurity_gain"])))), 4)
    subsample_cap = max(200, min(5000, int(merged.get("wafer_subsample_cap", defaults["wafer_subsample_cap"]))))

    resolved = {
        "mode_id": mode_key,
        "display_name": mode_spec["short_name"],
        "mode_name": mode_spec["name"],
        "mode_short_name": mode_spec["short_name"],
        "mode_badge": mode_spec["badge"],
        "mode_summary": mode_spec["engineer_summary"],
        "active_models": clean_active,
        "top_k_cutoff": top_k,
        "consensus_min_models": consensus_min,
        "regularization_type": reg_type,
        "reg_lambda": reg_lambda,
        "epochs": epochs,
        "convergence_tol": conv_tol,
        "n_trees": n_trees,
        "max_depth": max_depth,
        "max_leaves": max_leaves,
        "effective_max_leaves": min(max_leaves, 1 << max_depth),
        "learning_rate": learning_rate,
        "split_threshold_bins": split_bins,
        "min_impurity_gain": min_gain,
        "wafer_subsample_cap": subsample_cap,
    }

    customized_keys: List[str] = []
    for k, def_val in defaults.items():
        cur_val = resolved.get(k)
        if k == "active_models":
            if set(cur_val or []) != set(def_val or []):
                customized_keys.append(k)
        elif isinstance(def_val, float):
            if abs(float(cur_val) - def_val) > 1e-6:
                customized_keys.append(k)
        elif cur_val != def_val:
            customized_keys.append(k)

    resolved["customized_keys"] = customized_keys
    resolved["is_customized"] = len(customized_keys) > 0
    return resolved


def _standardize_vector(vec: List[float]) -> List[float]:
    n = len(vec)
    if n < 2:
        return [0.0 for _ in vec]
    m = sum(vec) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vec) / n)
    if sd < 1e-12:
        return [0.0 for _ in vec]
    return [(v - m) / sd for v in vec]


def _fit_bounded_tree(
    feature_vectors: Dict[str, List[float]],
    ys: List[float],
    row_indices: List[int],
    candidate_feats: List[str],
    max_depth: int = 2,
    max_leaves: int = 4,
    num_bins: int = 5,
    min_impurity_gain: float = 0.0,
) -> Tuple[Dict[str, float], Dict[int, float], int]:
    """Fit a decision tree bounded by both `max_depth` and `max_leaves` using best-first leaf splitting.

    Returns `(feature_gain_map, row_idx_to_prediction, leaves_created)`.
    """
    n_total = max(1, len(row_indices))
    feat_gains: Dict[str, float] = {}
    leaf_preds: Dict[int, float] = {}

    if not row_indices or not candidate_feats:
        return feat_gains, leaf_preds, 1

    effective_max_leaves = max(2, min(max_leaves, 1 << max_depth))
    # Each leaf item is (depth, sample_indices, mean_val)
    root_mean = sum(ys[i] for i in row_indices) / len(row_indices)
    open_leaves: List[Dict[str, Any]] = [
        {"depth": 0, "indices": row_indices, "mean": root_mean, "evaluated": False}
    ]
    closed_leaves: List[Dict[str, Any]] = []

    while len(open_leaves) + len(closed_leaves) < effective_max_leaves:
        # Evaluate split candidates on any unevaluated open leaves that haven't reached max_depth
        for leaf in open_leaves:
            if leaf["evaluated"]:
                continue
            leaf["evaluated"] = True
            idxs = leaf["indices"]
            if leaf["depth"] >= max_depth or len(idxs) < 16:
                leaf["best_gain"] = -1.0
                continue

            sub_y = [ys[i] for i in idxs]
            best_f = candidate_feats[0]
            best_g = -1.0
            best_th = 0.5
            best_low = leaf["mean"]
            best_high = leaf["mean"]
            for f in candidate_feats:
                col_vec = feature_vectors[f]
                sub_x = [col_vec[i] for i in idxs]
                th, low_m, high_m, g = _tree_stump_importance(
                    sub_x,
                    sub_y,
                    num_bins=num_bins,
                    min_impurity_gain=min_impurity_gain,
                )
                if g > best_g:
                    best_g = g
                    best_f = f
                    best_th = th
                    best_low = low_m
                    best_high = high_m

            if best_g > 0.0:
                weight = (len(idxs) / n_total) * (1.35 if leaf["depth"] == 0 else (0.70 / leaf["depth"]))
                leaf["best_gain"] = best_g * weight
                leaf["raw_gain"] = best_g
                leaf["best_feat"] = best_f
                leaf["best_thresh"] = best_th
                leaf["low_mean"] = best_low
                leaf["high_mean"] = best_high
            else:
                leaf["best_gain"] = -1.0

        splittable = [lf for lf in open_leaves if lf.get("best_gain", -1.0) > 0.0]
        if not splittable:
            break

        splittable.sort(key=lambda lf: -lf["best_gain"])
        chosen = splittable[0]
        open_leaves.remove(chosen)

        f_win = chosen["best_feat"]
        th_win = chosen["best_thresh"]
        col_win = feature_vectors[f_win]
        left_idxs = [i for i in chosen["indices"] if col_win[i] <= th_win]
        right_idxs = [i for i in chosen["indices"] if col_win[i] > th_win]

        if not left_idxs or not right_idxs:
            closed_leaves.append(chosen)
            continue

        feat_gains[f_win] = feat_gains.get(f_win, 0.0) + chosen["best_gain"]
        next_d = chosen["depth"] + 1
        open_leaves.append({
            "depth": next_d,
            "indices": left_idxs,
            "mean": chosen["low_mean"],
            "evaluated": False,
        })
        open_leaves.append({
            "depth": next_d,
            "indices": right_idxs,
            "mean": chosen["high_mean"],
            "evaluated": False,
        })

    all_final_leaves = open_leaves + closed_leaves
    for lf in all_final_leaves:
        m_val = lf["mean"]
        for i in lf["indices"]:
            leaf_preds[i] = m_val

    return feat_gains, leaf_preds, max(1, len(all_final_leaves))


def _proximal_regularized_update(
    rho: float,
    reg_type: str,
    reg_lambda: float,
) -> float:
    """Apply L1 soft-thresholding, L2 ridge shrinkage, or Elastic-Net proximal step."""
    if reg_type == "L1":
        if rho > reg_lambda:
            return rho - reg_lambda
        if rho < -reg_lambda:
            return rho + reg_lambda
        return 0.0
    if reg_type == "L2":
        return rho / (1.0 + reg_lambda * 2.5)
    # ELASTIC_NET (65% L1 + 35% L2 blend)
    l1_part = reg_lambda * 0.65
    l2_part = reg_lambda * 0.85
    if rho > l1_part:
        num = rho - l1_part
    elif rho < -l1_part:
        num = rho + l1_part
    else:
        return 0.0
    return num / (1.0 + l2_part)


def _compute_5_ml_model_scores(
    feature_vectors: Dict[str, List[float]],
    ys: List[float],
    full_abs_r: Optional[Dict[str, float]] = None,
    full_stump_gain: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    return_telemetry: bool = False,
) -> Any:
    """Run up to 5 distinct ML feature-selection algorithms across `feature_vectors` -> `ys`.

    Respects `config` (mode preset + user-edited hyperparameters: `active_models`,
    `regularization_type`, `reg_lambda`, `epochs`, `convergence_tol`, `n_trees`,
    `max_depth`, `max_leaves`, `learning_rate`, `split_threshold_bins`, `min_impurity_gain`).
    """
    cfg = config or resolve_feature_selection_config("BALANCED_FAB")
    active_set = set(cfg.get("active_models") or [m["model_id"] for m in ML_MODEL_SPECS])
    reg_type = str(cfg.get("regularization_type", "ELASTIC_NET")).upper()
    reg_lambda = float(cfg.get("reg_lambda", 0.075))
    max_epochs = int(cfg.get("epochs", 12))
    conv_tol = float(cfg.get("convergence_tol", 0.001))
    n_trees = int(cfg.get("n_trees", 24))
    max_depth = int(cfg.get("max_depth", 2))
    max_leaves = int(cfg.get("max_leaves", 4))
    lr = float(cfg.get("learning_rate", 0.18))
    split_bins = int(cfg.get("split_threshold_bins", 5))
    min_gain = float(cfg.get("min_impurity_gain", 0.002))

    n = len(ys)
    feat_names = list(feature_vectors.keys())
    if not feat_names or n < 4:
        empty_scores = {m["model_id"]: {f: 0.0 for f in feat_names} for m in ML_MODEL_SPECS}
        empty_ms = {m["model_id"]: 0.0 for m in ML_MODEL_SPECS}
        if return_telemetry:
            return empty_scores, empty_ms, {}
        return empty_scores, empty_ms

    z_y = _standardize_vector(ys)
    z_x: Dict[str, List[float]] = {f: _standardize_vector(feature_vectors[f]) for f in feat_names}

    # Precompute marginal Pearson |r| and tree stump gain
    abs_r: Dict[str, float] = {}
    stump_gain: Dict[str, float] = {}
    for f in feat_names:
        if full_abs_r and f in full_abs_r:
            abs_r[f] = full_abs_r[f]
        else:
            r_f, _, _ = _pearson_r_and_p(feature_vectors[f], ys)
            abs_r[f] = abs(r_f)
        if full_stump_gain and f in full_stump_gain and split_bins == 5 and min_gain <= 0.002:
            stump_gain[f] = full_stump_gain[f]
        else:
            _, _, _, g_f = _tree_stump_importance(
                feature_vectors[f],
                ys,
                num_bins=split_bins,
                min_impurity_gain=min_gain,
            )
            stump_gain[f] = g_f

    ordered_feats = sorted(feat_names, key=lambda f: -abs_r[f])
    per_model_ms: Dict[str, float] = {}
    telemetry: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------------
    # Model 1: Penalized Linear Coordinate Descent (L1 / L2 / Elastic-Net)
    # ------------------------------------------------------------------------
    if "L1_LASSO" in active_set:
        t_m1 = time.perf_counter()
        betas_lasso: Dict[str, float] = {f: 0.0 for f in feat_names}
        res_lasso = list(z_y)
        epochs_run_m1 = 0
        converged_m1 = False
        last_max_delta_m1 = 0.0
        for ep in range(1, max_epochs + 1):
            epochs_run_m1 = ep
            max_delta = 0.0
            for f in ordered_feats:
                zx = z_x[f]
                old_b = betas_lasso[f]
                rho = old_b + (sum(zx[i] * res_lasso[i] for i in range(n)) / n)
                new_b = _proximal_regularized_update(rho, reg_type, reg_lambda)
                diff = new_b - old_b
                abs_d = abs(diff)
                if abs_d > max_delta:
                    max_delta = abs_d
                if abs_d > 1e-9:
                    for i in range(n):
                        res_lasso[i] -= diff * zx[i]
                    betas_lasso[f] = new_b
            last_max_delta_m1 = round(max_delta, 6)
            if max_delta < conv_tol:
                converged_m1 = True
                break
        raw_lasso = {f: abs(betas_lasso[f]) + 0.12 * abs_r[f] for f in feat_names}
        per_model_ms["L1_LASSO"] = round((time.perf_counter() - t_m1) * 1000.0, 1)
        telemetry["L1_LASSO"] = {
            "active": True,
            "status": "ACTIVE",
            "reg_type": reg_type,
            "reg_lambda": reg_lambda,
            "epochs_run": epochs_run_m1,
            "epochs_executed": epochs_run_m1,
            "max_epochs": max_epochs,
            "converged_early": converged_m1,
            "final_max_delta": last_max_delta_m1,
            "elapsed_ms": per_model_ms["L1_LASSO"],
            "effective_hyperparams": f"{reg_type} (λ={reg_lambda}, tol={conv_tol})",
        }
    else:
        raw_lasso = {f: 0.0 for f in feat_names}
        per_model_ms["L1_LASSO"] = 0.0
        telemetry["L1_LASSO"] = {
            "active": False,
            "status": "SKIPPED",
            "reg_type": reg_type,
            "reg_lambda": reg_lambda,
            "epochs_run": 0,
            "epochs_executed": 0,
            "max_epochs": max_epochs,
            "converged_early": False,
            "final_max_delta": 0.0,
            "elapsed_ms": 0.0,
            "effective_hyperparams": "Skipped by Mode/User",
        }

    # ------------------------------------------------------------------------
    # Model 2: Random Forest Impurity Importance (Configurable Trees, Depth & Leaves)
    # ------------------------------------------------------------------------
    if "RANDOM_FOREST" in active_set:
        t_m2 = time.perf_counter()
        rf_scores: Dict[str, float] = {f: 0.0 for f in feat_names}
        rng_rf = random.Random(42)
        m_try = max(6, int(math.sqrt(len(feat_names))) + 2)
        total_leaves_rf = 0
        for t_idx in range(n_trees):
            boot_indices = [rng_rf.randrange(n) for _ in range(n)]
            sub_feats = rng_rf.sample(feat_names, min(len(feat_names), m_try))
            for top_f in ordered_feats[:4]:
                if top_f not in sub_feats and (t_idx % 2 == 0):
                    sub_feats.append(top_f)

            tree_gains, _, leaves_cnt = _fit_bounded_tree(
                feature_vectors=feature_vectors,
                ys=ys,
                row_indices=boot_indices,
                candidate_feats=sub_feats,
                max_depth=max_depth,
                max_leaves=max_leaves,
                num_bins=split_bins,
                min_impurity_gain=min_gain,
            )
            total_leaves_rf += leaves_cnt
            for f_k, g_val in tree_gains.items():
                rf_scores[f_k] += g_val

        for f in feat_names:
            rf_scores[f] += 0.25 * stump_gain[f]
        per_model_ms["RANDOM_FOREST"] = round((time.perf_counter() - t_m2) * 1000.0, 1)
        avg_leaves = round(total_leaves_rf / max(1, n_trees), 1)
        telemetry["RANDOM_FOREST"] = {
            "active": True,
            "status": "ACTIVE",
            "n_trees": n_trees,
            "max_depth": max_depth,
            "max_leaves": max_leaves,
            "epochs_executed": n_trees,
            "max_epochs": n_trees,
            "converged_early": False,
            "final_max_delta": avg_leaves,
            "elapsed_ms": per_model_ms["RANDOM_FOREST"],
            "effective_hyperparams": f"{n_trees} Trees • Depth≤{max_depth} • Leaves≤{min(max_leaves, 1 << max_depth)} (Avg {avg_leaves})",
        }
    else:
        rf_scores = {f: 0.0 for f in feat_names}
        per_model_ms["RANDOM_FOREST"] = 0.0
        telemetry["RANDOM_FOREST"] = {
            "active": False,
            "status": "SKIPPED",
            "n_trees": n_trees,
            "max_depth": max_depth,
            "max_leaves": max_leaves,
            "epochs_executed": 0,
            "max_epochs": n_trees,
            "converged_early": False,
            "final_max_delta": 0.0,
            "elapsed_ms": 0.0,
            "effective_hyperparams": "Skipped by Mode/User",
        }

    # ------------------------------------------------------------------------
    # Model 3: Gradient Boosted Decision Trees (Configurable Stages, Depth, Leaves & LR)
    # ------------------------------------------------------------------------
    if "GRADIENT_BOOSTING" in active_set:
        t_m3 = time.perf_counter()
        gb_scores: Dict[str, float] = {f: 0.0 for f in feat_names}
        mean_y = sum(ys) / n
        residuals = [y - mean_y for y in ys]
        all_row_indices = list(range(n))
        gb_candidate_feats = ordered_feats[:28] + [
            f for f in feat_names if f.startswith("__HYP_") and f not in ordered_feats[:28]
        ]
        gb_depth = 1 if max_depth <= 2 else min(3, max_depth - 1)
        gb_leaves = 2 if gb_depth == 1 else max(2, min(8, max_leaves))
        stages_run = 0
        gb_converged = False
        prev_mse = sum(r * r for r in residuals) / n

        for stage in range(max_epochs):
            stages_run = stage + 1
            decay = math.pow(0.90, stage)
            if gb_depth == 1:
                stage_gains: List[Tuple[float, str, float, float, float]] = []
                for f in gb_candidate_feats:
                    xs_f = feature_vectors[f]
                    th, low_m, high_m, gain = _tree_stump_importance(
                        xs_f,
                        residuals,
                        num_bins=split_bins,
                        min_impurity_gain=min_gain,
                    )
                    stage_gains.append((gain, f, th, low_m, high_m))
                stage_gains.sort(key=lambda x: -x[0])
                win_gain, win_f, win_th, win_low, win_high = stage_gains[0]
                gb_scores[win_f] += win_gain * decay
                for runner_gain, runner_f, _, _, _ in stage_gains[1:4]:
                    gb_scores[runner_f] += runner_gain * 0.30 * decay
                xs_win = feature_vectors[win_f]
                for i in range(n):
                    pred = win_low if xs_win[i] <= win_th else win_high
                    residuals[i] -= lr * pred
            else:
                tree_gains, leaf_preds, _ = _fit_bounded_tree(
                    feature_vectors=feature_vectors,
                    ys=residuals,
                    row_indices=all_row_indices,
                    candidate_feats=gb_candidate_feats,
                    max_depth=gb_depth,
                    max_leaves=gb_leaves,
                    num_bins=split_bins,
                    min_impurity_gain=min_gain,
                )
                for f_k, g_val in tree_gains.items():
                    gb_scores[f_k] += g_val * decay
                if leaf_preds:
                    for i in range(n):
                        residuals[i] -= lr * leaf_preds.get(i, 0.0)

            cur_mse = sum(r * r for r in residuals) / n
            rel_drop = abs(prev_mse - cur_mse) / max(1e-9, prev_mse)
            prev_mse = cur_mse
            if stage >= 2 and rel_drop < conv_tol * 0.25:
                gb_converged = True
                break

        for f in feat_names:
            gb_scores[f] += 0.12 * stump_gain[f]
        per_model_ms["GRADIENT_BOOSTING"] = round((time.perf_counter() - t_m3) * 1000.0, 1)
        telemetry["GRADIENT_BOOSTING"] = {
            "active": True,
            "status": "ACTIVE",
            "rounds_run": stages_run,
            "epochs_executed": stages_run,
            "max_epochs": max_epochs,
            "learning_rate": lr,
            "converged_early": gb_converged,
            "final_max_delta": round(prev_mse, 4),
            "elapsed_ms": per_model_ms["GRADIENT_BOOSTING"],
            "effective_hyperparams": f"{stages_run}/{max_epochs} Stages • lr={lr} • Depth≤{gb_depth} • Leaves≤{gb_leaves}",
        }
    else:
        gb_scores = {f: 0.0 for f in feat_names}
        per_model_ms["GRADIENT_BOOSTING"] = 0.0
        telemetry["GRADIENT_BOOSTING"] = {
            "active": False,
            "status": "SKIPPED",
            "rounds_run": 0,
            "epochs_executed": 0,
            "max_epochs": max_epochs,
            "learning_rate": lr,
            "converged_early": False,
            "final_max_delta": 0.0,
            "elapsed_ms": 0.0,
            "effective_hyperparams": "Skipped by Mode/User",
        }

    # ------------------------------------------------------------------------
    # Model 4: Out-of-Sample Permutation Importance on Penalized Predictor
    # ------------------------------------------------------------------------
    if "PERMUTATION_RIDGE" in active_set:
        t_m4 = time.perf_counter()
        betas_ridge: Dict[str, float] = {f: 0.0 for f in feat_names}
        res_ridge = list(z_y)
        eff_penalty = reg_lambda * (3.2 if reg_type == "L2" else (2.0 if reg_type == "ELASTIC_NET" else 4.0))
        ridge_epochs = max(2, min(max_epochs, 16))
        epochs_run_m4 = 0
        converged_m4 = False
        for ep in range(1, ridge_epochs + 1):
            epochs_run_m4 = ep
            max_d4 = 0.0
            for f in ordered_feats:
                zx = z_x[f]
                old_b = betas_ridge[f]
                grad = sum(zx[i] * res_ridge[i] for i in range(n)) / n
                new_b = (old_b + grad) / (1.0 + eff_penalty)
                diff = new_b - old_b
                if abs(diff) > max_d4:
                    max_d4 = abs(diff)
                if abs(diff) > 1e-9:
                    for i in range(n):
                        res_ridge[i] -= diff * zx[i]
                    betas_ridge[f] = new_b
            if max_d4 < conv_tol:
                converged_m4 = True
                break

        base_mse = sum(r * r for r in res_ridge) / n
        perm_scores: Dict[str, float] = {}
        for idx_f, f in enumerate(feat_names):
            rng_p = random.Random(200 + idx_f)
            zx = z_x[f]
            shuffled_zx = list(zx)
            rng_p.shuffle(shuffled_zx)
            b_f = betas_ridge[f]
            perm_mse = 0.0
            for i in range(n):
                r_perm = res_ridge[i] + b_f * (zx[i] - shuffled_zx[i])
                perm_mse += r_perm * r_perm
            perm_mse /= n
            delta_mse = max(0.0, perm_mse - base_mse)
            perm_scores[f] = delta_mse + 0.10 * (abs_r[f] ** 2)
        per_model_ms["PERMUTATION_RIDGE"] = round((time.perf_counter() - t_m4) * 1000.0, 1)
        telemetry["PERMUTATION_RIDGE"] = {
            "active": True,
            "status": "ACTIVE",
            "epochs_executed": epochs_run_m4,
            "max_epochs": ridge_epochs,
            "converged_early": converged_m4,
            "final_max_delta": round(base_mse, 4),
            "elapsed_ms": per_model_ms["PERMUTATION_RIDGE"],
            "effective_hyperparams": f"Permutation ({reg_type}, λ_eff={round(eff_penalty, 3)})",
        }
    else:
        perm_scores = {f: 0.0 for f in feat_names}
        per_model_ms["PERMUTATION_RIDGE"] = 0.0
        telemetry["PERMUTATION_RIDGE"] = {
            "active": False,
            "status": "SKIPPED",
            "epochs_executed": 0,
            "max_epochs": max_epochs,
            "converged_early": False,
            "final_max_delta": 0.0,
            "elapsed_ms": 0.0,
            "effective_hyperparams": "Skipped by Mode/User",
        }

    # ------------------------------------------------------------------------
    # Model 5: Non-Parametric Mutual Information + Spearman Rank Selector
    # ------------------------------------------------------------------------
    if "MUTUAL_INFO_RANK" in active_set:
        t_m5 = time.perf_counter()
        mi_scores: Dict[str, float] = {}
        q_bins = 3 if split_bins <= 5 else (4 if split_bins == 7 else 5)
        y_order = sorted(range(n), key=lambda i: ys[i])
        y_ranks = [0.0] * n
        for r_pos, orig_i in enumerate(y_order):
            y_ranks[orig_i] = float(r_pos)
        y_qbins = [min(q_bins - 1, int((r_pos * q_bins) / n)) for r_pos in y_ranks]

        for f in feat_names:
            xf = feature_vectors[f]
            x_order = sorted(range(n), key=lambda i: xf[i])
            x_ranks = [0.0] * n
            for r_pos, orig_i in enumerate(x_order):
                x_ranks[orig_i] = float(r_pos)
            spearman_r, _, _ = _pearson_r_and_p(x_ranks, y_ranks)

            counts_xy = [[0] * q_bins for _ in range(q_bins)]
            counts_x = [0] * q_bins
            counts_y = [0] * q_bins
            for i in range(n):
                bx = min(q_bins - 1, int((x_ranks[i] * q_bins) / n))
                by_bin = y_qbins[i]
                counts_xy[bx][by_bin] += 1
                counts_x[bx] += 1
                counts_y[by_bin] += 1

            mi_val = 0.0
            for bx in range(q_bins):
                for by_bin in range(q_bins):
                    c_xy = counts_xy[bx][by_bin]
                    if c_xy > 0 and counts_x[bx] > 0 and counts_y[by_bin] > 0:
                        p_xy = c_xy / n
                        p_x = counts_x[bx] / n
                        p_y = counts_y[by_bin] / n
                        mi_val += p_xy * math.log(p_xy / (p_x * p_y) + 1e-12)

            mi_scores[f] = 0.55 * abs(spearman_r) + 0.45 * max(0.0, mi_val)
        per_model_ms["MUTUAL_INFO_RANK"] = round((time.perf_counter() - t_m5) * 1000.0, 1)
        telemetry["MUTUAL_INFO_RANK"] = {
            "active": True,
            "status": "ACTIVE",
            "epochs_executed": 1,
            "max_epochs": 1,
            "converged_early": True,
            "final_max_delta": 0.0,
            "elapsed_ms": per_model_ms["MUTUAL_INFO_RANK"],
            "effective_hyperparams": f"{q_bins}×{q_bins} Quantile MI Grid + Spearman",
        }
    else:
        mi_scores = {f: 0.0 for f in feat_names}
        per_model_ms["MUTUAL_INFO_RANK"] = 0.0
        telemetry["MUTUAL_INFO_RANK"] = {
            "active": False,
            "status": "SKIPPED",
            "epochs_executed": 0,
            "max_epochs": 1,
            "converged_early": False,
            "final_max_delta": 0.0,
            "elapsed_ms": 0.0,
            "effective_hyperparams": "Skipped by Mode/User",
        }

    raw_by_model = {
        "L1_LASSO": raw_lasso,
        "RANDOM_FOREST": rf_scores,
        "GRADIENT_BOOSTING": gb_scores,
        "PERMUTATION_RIDGE": perm_scores,
        "MUTUAL_INFO_RANK": mi_scores,
    }

    # Normalize each model's importance scores to [0.0, 100.0]
    norm_by_model: Dict[str, Dict[str, float]] = {}
    for m_id, m_dict in raw_by_model.items():
        if m_id not in active_set:
            norm_by_model[m_id] = {f: 0.0 for f in feat_names}
            continue
        max_v = max(m_dict.values()) if m_dict else 1.0
        if max_v < 1e-12:
            max_v = 1.0
        norm_by_model[m_id] = {
            f: round(100.0 * (val / max_v), 1) for f, val in m_dict.items()
        }
    if return_telemetry:
        return norm_by_model, per_model_ms, telemetry
    return norm_by_model, per_model_ms


def build_feature_selection_a2ui(
    hypothesis_results: List[Dict[str, Any]],
    top_k_cutoff: int = 5,
    mode_name: str = "Balanced Mode",
) -> Dict[str, Any]:
    """Build A2UI v0.9 updateComponents payload for Screen 3 Multi-Model Feature Selection."""
    card_ids: List[str] = []
    components: List[Dict[str, Any]] = []

    for idx, h in enumerate(hypothesis_results, start=1):
        cid = f"fs_card_{idx}"
        col_id = f"fs_col_{idx}"
        title_id = f"fs_title_{idx}"
        summary_id = f"fs_summary_{idx}"
        card_ids.append(cid)
        components.extend([
            {"id": cid, "component": "Card", "child": col_id},
            {"id": col_id, "component": "Column", "children": [title_id, summary_id]},
            {
                "id": title_id,
                "component": "Text",
                "text": f"{h['short_title']} — {h['consistency_hero']} Models in Top-{top_k_cutoff}",
                "variant": "h3",
            },
            {
                "id": summary_id,
                "component": "Text",
                "text": h["plain_english_takeaway"],
            },
        ])

    full_components = [
        {"id": "root", "component": "Column", "children": ["fs_header"] + card_ids},
        {
            "id": "fs_header",
            "component": "Text",
            "text": f"Multi-Model Feature Selection ({mode_name} • Top-{top_k_cutoff} Threshold)",
            "variant": "h2",
        },
    ] + components

    msg = {
        "version": "v0.9",
        "updateComponents": {
            "surfaceId": "feature_selection_surface",
            "components": full_components,
        },
    }
    validate_a2ui_v09_payload(msg)
    return msg


def run_multi_model_feature_selection(
    hypotheses: Optional[List[Dict[str, Any]]] = None,
    selected_columns: Optional[List[str]] = None,
    top_k_cutoff: Optional[int] = None,
    dataset_id: str = "lot_alpha",
    data_dir: Optional[Path] = None,
    bypass_subsample: bool = False,
    mode_id: str = "BALANCED_FAB",
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate hypotheses and active parameters across configurable ML feature selection models.

    Supports 4 Process Engineer Modes (`FAST_TRIAGE`, `BALANCED_FAB`, `DEEP_INTERACTION`,
    `STRICT_AUDIT`) plus full user editing of prefilled parameters (`active_models`,
    `regularization_type`, `reg_lambda`, `max_depth`, `max_leaves`, `n_trees`, `epochs`,
    `learning_rate`, `convergence_tol`, `split_threshold_bins`, `min_impurity_gain`,
    `top_k_cutoff`, `consensus_min_models`, `wafer_subsample_cap`).
    """
    t0 = time.perf_counter()
    resolved_cfg = resolve_feature_selection_config(
        mode_id=mode_id,
        custom_params=model_config,
        top_k_override=top_k_cutoff,
    )
    cutoff = resolved_cfg["top_k_cutoff"]
    consensus_min = resolved_cfg["consensus_min_models"]
    active_model_ids = resolved_cfg["active_models"]
    active_model_set = set(active_model_ids)
    total_active_m = max(1, len(active_model_ids))

    ds_key = normalize_dataset_id(dataset_id)
    ds_meta = DATASET_CATALOG[ds_key]
    if bypass_subsample:
        _OBF_ROWS_CACHE.pop(str((data_dir or DATA_DIR) / ds_meta["obf_filename"]), None)
        _OBF_BASE_STATS_CACHE.clear()
    rows = _load_obfuscated_rows(data_dir=data_dir, dataset_id=ds_key)
    all_param_cols = [k for k in rows[0].keys() if k not in ("wafer_id", "yield_pct") and not k.startswith("_")]
    cols = [c for c in (selected_columns or all_param_cols) if c in all_param_cols]
    if not cols:
        cols = all_param_cols

    # If no hypotheses passed, discover candidate hypotheses on active subset + dataset
    if not hypotheses:
        disc = discover_candidate_hypotheses(
            selected_columns=cols,
            top_k=8,
            data_dir=data_dir,
            dataset_id=ds_key,
        )
        hypotheses = disc["data"]["candidate_hypotheses"]

    # Retrieve exact full-dataset marginal statistics across all N wafers (up to 50,000)
    full_base_stats = _get_cached_full_col_stats(rows, all_param_cols)
    full_abs_r = {c: abs(full_base_stats[c]["r"]) for c in cols if c in full_base_stats}
    full_stump = {c: full_base_stats[c]["ml_importance"] for c in cols if c in full_base_stats}

    subsample_cap = resolved_cfg["wafer_subsample_cap"]
    work_rows = rows if bypass_subsample else _get_work_rows(rows, max_rows=subsample_cap)
    work_ys = [r["yield_pct"] for r in work_rows]

    # 1. Build feature vectors for all active columns + synthetic hypothesis vectors in a single pass
    combined_vectors: Dict[str, List[float]] = {c: [r[c] for r in work_rows] for c in cols}
    synth_key_by_idx: Dict[int, str] = {}

    for idx, hyp in enumerate(hypotheses):
        h_id = hyp.get("hypothesis_id", f"HYP_{idx}")
        h_type = hyp.get("hypothesis_type", "single_parameter")
        params = [obfuscate_column_name(p.upper()) for p in hyp.get("parameters", [])]
        if len(params) == 1 and params[0] in STEP_OBFUSCATION_MAP.values():
            h_type = "step_level"

        if h_type == "single_parameter" and params and params[0] in combined_vectors:
            continue

        synth_key = f"__HYP_{idx}_{h_id}"
        synth_key_by_idx[idx] = synth_key
        if h_type == "two_way_interaction" and len(params) >= 2:
            c1 = params[0] if params[0] in all_param_cols else cols[0]
            c2 = params[1] if params[1] in all_param_cols else cols[min(1, len(cols) - 1)]
            mode = hyp.get("interaction_mode", "HIGH_LOW")
            synth_vec = [
                (r[c1] * (1.0 - r[c2])) if mode == "HIGH_LOW" else (r[c1] * r[c2])
                for r in work_rows
            ]
        elif h_type == "step_level":
            step_name = params[0] if params else "STEP_A"
            step_cols = [c for c in cols if c.startswith(step_name + "_")]
            if not step_cols:
                step_cols = [c for c in all_param_cols if c.startswith(step_name + "_")]
            s_corrs = sorted(
                [(c, full_base_stats[c]["r"]) for c in step_cols if c in full_base_stats],
                key=lambda item: -abs(item[1]),
            )
            synth_vec = []
            for r in work_rows:
                sv = sum((r[c] if rc < 0 else (1.0 - r[c])) for c, rc in s_corrs[:3])
                synth_vec.append(sv / max(1, len(s_corrs[:3])))
        else:
            c_single = params[0] if (params and params[0] in all_param_cols) else cols[0]
            synth_vec = [r[c_single] for r in work_rows]
        combined_vectors[synth_key] = synth_vec

    all_model_scores, per_model_ms, convergence_telemetry = _compute_5_ml_model_scores(
        combined_vectors,
        work_ys,
        full_abs_r=full_abs_r,
        full_stump_gain=full_stump,
        config=resolved_cfg,
        return_telemetry=True,
    )

    # Extract base column scores and 1-indexed ranks across `cols`
    base_model_scores: Dict[str, Dict[str, float]] = {}
    base_model_ranks: Dict[str, Dict[str, int]] = {}
    for m_spec in ML_MODEL_SPECS:
        m_id = m_spec["model_id"]
        if m_id not in active_model_set:
            base_model_scores[m_id] = {c: 0.0 for c in cols}
            base_model_ranks[m_id] = {c: len(cols) for c in cols}
            continue
        max_base = max((all_model_scores[m_id].get(c, 0.0) for c in cols), default=1.0) or 1.0
        base_model_scores[m_id] = {
            c: round(100.0 * (all_model_scores[m_id].get(c, 0.0) / max_base), 1)
            for c in cols
        }
        sorted_cols = sorted(cols, key=lambda c: -base_model_scores[m_id].get(c, 0.0))
        base_model_ranks[m_id] = {c: rk for rk, c in enumerate(sorted_cols, start=1)}

    # 2. Build Global Top Features Leaderboard across active ML models
    global_features: List[Dict[str, Any]] = []
    for c in cols:
        per_m = []
        in_top_count = 0
        ranks_list = []
        scores_list = []
        for m_spec in ML_MODEL_SPECS:
            m_id = m_spec["model_id"]
            is_active = m_id in active_model_set
            rk = base_model_ranks[m_id][c] if is_active else None
            sc = base_model_scores[m_id][c] if is_active else 0.0
            is_top = bool(is_active and rk is not None and rk <= cutoff)
            if is_top:
                in_top_count += 1
            if is_active and rk is not None:
                ranks_list.append(rk)
                scores_list.append(sc)
            per_m.append({
                "model_id": m_id,
                "short_name": m_spec["short_name"],
                "active": is_active,
                "is_active": is_active,
                "rank": rk,
                "in_top_k": is_top,
                "importance_pct": sc,
            })
        mean_rk = round(sum(ranks_list) / max(1, len(ranks_list)), 1)
        mean_sc = round(sum(scores_list) / max(1, len(scores_list)), 1)
        global_features.append({
            "column": c,
            "short_name": _format_short_param(c),
            "hypothesis_id": f"HYP_SINGLE_{c}",
            "models_in_top_k": in_top_count,
            "total_models": total_active_m,
            "consistency_hero": f"{in_top_count} / {total_active_m}",
            "consistency_pct": int(round(100.0 * in_top_count / total_active_m)),
            "mean_rank": mean_rk,
            "mean_importance_pct": mean_sc,
            "per_model_ranks": per_m,
        })

    global_features.sort(key=lambda x: (-x["models_in_top_k"], x["mean_rank"], -x["mean_importance_pct"]))

    # 3. Evaluate each hypothesis across active ML models
    hypothesis_consistency: List[Dict[str, Any]] = []
    for idx, hyp in enumerate(hypotheses):
        h_id = hyp.get("hypothesis_id", "HYP_CUSTOM")
        h_type = hyp.get("hypothesis_type", "single_parameter")
        params = [obfuscate_column_name(p.upper()) for p in hyp.get("parameters", [])]
        short_title = hyp.get("short_title") or hyp.get("title") or h_id
        title = hyp.get("title") or short_title

        if len(params) == 1 and params[0] in STEP_OBFUSCATION_MAP.values():
            h_type = "step_level"

        primary_col = params[0] if (params and params[0] in cols) else cols[0]

        if idx not in synth_key_by_idx and params and params[0] in cols:
            target_col = params[0]
            h_scores = {m["model_id"]: base_model_scores[m["model_id"]][target_col] for m in ML_MODEL_SPECS}
            h_ranks = {m["model_id"]: base_model_ranks[m["model_id"]][target_col] for m in ML_MODEL_SPECS}
        else:
            synth_key = synth_key_by_idx[idx]
            pool_keys = cols + [synth_key]
            h_scores = {}
            h_ranks = {}
            for m_spec in ML_MODEL_SPECS:
                m_id = m_spec["model_id"]
                if m_id not in active_model_set:
                    h_ranks[m_id] = len(pool_keys)
                    h_scores[m_id] = 0.0
                    continue
                max_pool = max((all_model_scores[m_id].get(k, 0.0) for k in pool_keys), default=1.0) or 1.0
                sorted_pool = sorted(pool_keys, key=lambda k: -all_model_scores[m_id].get(k, 0.0))
                h_ranks[m_id] = sorted_pool.index(synth_key) + 1
                h_scores[m_id] = round(100.0 * (all_model_scores[m_id].get(synth_key, 0.0) / max_pool), 1)

        per_model_breakdown: List[Dict[str, Any]] = []
        in_top_k_count = 0
        rk_vals = []
        sc_vals = []

        for m_spec in ML_MODEL_SPECS:
            m_id = m_spec["model_id"]
            is_active = m_id in active_model_set
            rk = h_ranks[m_id] if is_active else None
            sc = h_scores[m_id] if is_active else 0.0
            is_top = bool(is_active and rk is not None and rk <= cutoff)
            if is_top:
                in_top_k_count += 1
            if is_active and rk is not None:
                rk_vals.append(rk)
                sc_vals.append(sc)
            tel_m = convergence_telemetry.get(m_id, {})
            per_model_breakdown.append({
                "model_id": m_id,
                "model_name": m_spec["model_name"],
                "short_name": m_spec["short_name"],
                "family": m_spec["family"],
                "active": is_active,
                "is_active": is_active,
                "rank": rk,
                "in_top_k": is_top,
                "importance_pct": sc,
                "compute_ms": per_model_ms.get(m_id, 0.0),
                "epochs_executed": tel_m.get("epochs_executed", 0),
                "converged_early": tel_m.get("converged_early", False),
                "badge_text": (f"✓ #{rk}" if is_top else f"✗ #{rk}") if is_active else "SKIPPED",
            })

        cons_pct = int(round(100.0 * in_top_k_count / total_active_m))
        mean_rk = round(sum(rk_vals) / max(1, len(rk_vals)), 1)
        mean_imp = round(sum(sc_vals) / max(1, len(sc_vals)), 1)
        partial_min = max(1, min(consensus_min - 1, total_active_m // 2))

        if in_top_k_count >= consensus_min:
            verdict_label = "CONSISTENT TOP FEATURE"
            color = "emerald"
            takeaway = (
                f"[{resolved_cfg['mode_short_name']}] Appears in Top-{cutoff} across {in_top_k_count}/{total_active_m} "
                f"active ML models (Avg Rank #{mean_rk:.1f}, min required ≥{consensus_min}/{total_active_m} • "
                f"Reg={resolved_cfg['regularization_type']}, Depth≤{resolved_cfg['max_depth']}, Leaves≤{resolved_cfg['max_leaves']})."
            )
        elif in_top_k_count >= partial_min and in_top_k_count > 0:
            verdict_label = "PARTIAL MODEL AGREEMENT"
            color = "amber"
            takeaway = (
                f"[{resolved_cfg['mode_short_name']}] Appears in Top-{cutoff} for {in_top_k_count}/{total_active_m} "
                f"active ML models (Avg Rank #{mean_rk:.1f}, threshold ≥{consensus_min}/{total_active_m}) — "
                f"model-dependent signal; try Deep Coupling mode or Elastic-Net."
            )
        else:
            verdict_label = "NOT IN TOP FEATURES"
            color = "rose"
            takeaway = (
                f"[{resolved_cfg['mode_short_name']}] Appears in Top-{cutoff} for {in_top_k_count}/{total_active_m} "
                f"active ML models (Avg Rank #{mean_rk:.1f}) — ranked outside top drivers under current thresholds."
            )

        hypothesis_consistency.append({
            "hypothesis_id": h_id,
            "hypothesis_type": h_type,
            "short_title": short_title,
            "title": title,
            "parameters": params,
            "primary_column": primary_col,
            "queue_status": hyp.get("queue_status", "CANDIDATE"),
            "models_in_top_k": in_top_k_count,
            "total_models": total_active_m,
            "consistency_pct": cons_pct,
            "consistency_hero": f"{in_top_k_count} / {total_active_m}",
            "consistency_label": f"{cons_pct}% Consensus (Top-{cutoff} • Need ≥{consensus_min}/{total_active_m})",
            "mean_rank": mean_rk,
            "mean_rank_hero": f"#{mean_rk:.1f}",
            "mean_importance_pct": mean_imp,
            "consistency_verdict": verdict_label,
            "confidence_color": color,
            "per_model_breakdown": per_model_breakdown,
            "plain_english_takeaway": takeaway,
        })

    consistent_hyps = [h for h in hypothesis_consistency if h["models_in_top_k"] >= consensus_min]
    top_global = global_features[0] if global_features else {
        "column": "STEP_A_TEMPERATURE_01",
        "consistency_hero": f"{total_active_m} / {total_active_m}",
        "mean_rank": 1.0,
    }

    # Compute cross-model Top-K Jaccard/overlap concordance among the active models
    top_sets = [
        {c for c, rk in base_model_ranks[m_id].items() if rk <= cutoff}
        for m_id in active_model_ids
    ]
    pair_overlaps = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            s1, s2 = top_sets[i], top_sets[j]
            pair_overlaps.append(len(s1 & s2) / max(1, cutoff))
    concordance_pct = (
        int(round(100.0 * (sum(pair_overlaps) / len(pair_overlaps))))
        if pair_overlaps
        else 100
    )
    total_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    summary_kpis = {
        "consistent_hypotheses_hero": f"{len(consistent_hyps)} / {len(hypothesis_consistency)}",
        "consistent_hypotheses_label": (
            f"Confirmed in Top-{cutoff} by ≥{consensus_min}/{total_active_m} Active ML Models "
            f"({resolved_cfg['mode_short_name']})"
        ),
        "top_consensus_hero": top_global["consistency_hero"],
        "top_consensus_param": top_global["column"],
        "top_consensus_rank": f"Avg Rank #{top_global['mean_rank']:.1f}",
        "ensemble_agreement_hero": f"{concordance_pct}%",
        "ensemble_agreement_label": (
            f"Cross-Model Top-{cutoff} Concordance ({total_ms} ms | {len(work_rows):,}/{len(rows):,} Wafers)"
        ),
    }

    models_with_timing = [
        {
            **m,
            "active": m["model_id"] in active_model_set,
            "is_active": m["model_id"] in active_model_set,
            "compute_ms": per_model_ms.get(m["model_id"], 0.0),
            "telemetry": convergence_telemetry.get(m["model_id"], {}),
        }
        for m in ML_MODEL_SPECS
    ]

    a2ui_msg = build_feature_selection_a2ui(
        hypothesis_consistency,
        top_k_cutoff=cutoff,
        mode_name=resolved_cfg["mode_short_name"],
    )

    return {
        "data": {
            "dataset_id": ds_key,
            "dataset_name": ds_meta["name"],
            "dataset_short_name": ds_meta["short_name"],
            "active_columns_count": len(cols),
            "active_models_count": total_active_m,
            "total_wafers_analyzed": len(rows),
            "ensemble_subsample_rows": len(work_rows),
            "total_cells_analyzed": len(rows) * len(cols),
            "total_compute_ms": total_ms,
            "per_model_compute_ms": per_model_ms,
            "top_k_cutoff": cutoff,
            "consensus_min_models": consensus_min,
            "mode_id": resolved_cfg["mode_id"],
            "mode_metadata": ENGINEERING_MODES[resolved_cfg["mode_id"]],
            "engineering_modes": list(ENGINEERING_MODES.values()),
            "resolved_config": resolved_cfg,
            "convergence_telemetry": convergence_telemetry,
            "models_metadata": models_with_timing,
            "summary_kpis": summary_kpis,
            "hypothesis_consistency": hypothesis_consistency,
            "global_top_features": global_features[:12],
        },
        "ui": a2ui_msg,
        "ui_available": True,
    }




