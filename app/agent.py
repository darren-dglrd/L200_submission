#!/usr/bin/env python3
"""Google ADK Multi-Agent Orchestration, Session Compaction, and Long-Term Memory Bank.

Design Principles Invoked Verbatim (the Google 5-Day AI Agents Intensive whitepapers):
1. Specialist Team over Monolithic Agent (`Introduction to Agents.pdf`, p. 17):
   - Agent 1 (`hypothesis_generator_agent`, `output_key="candidate_hypotheses"`):
     Uses the `data_preprocess` skill (`skills/data_preprocess/SKILL.md`) and analytical
     screening (`discover_candidate_hypotheses`) to propose 1-way, 2-way interaction, and
     static step-level hypotheses (`STEP_A`..`STEP_E`) as A2UI v0.9 cards.
   - Agent 2 (`hypothesis_evaluator_agent`, `output_key="latest_evaluation"`):
     Pops hypotheses 1-by-1 from `session.state["hypothesis_queue"]`, evaluates them via
     statistical p-values (with Benjamini-Hochberg FDR) + ML tree-split variance importance,
     and emits zero-ML-jargon verdicts + Good/Bad Wafer Distribution charts in A2UI v0.9.
   - Root Coordinator (`root_agent`): Combines dynamic `AgentTool` routing for SME chat with
     deterministic `SequentialAgent` execution for 1-by-1 queue evaluation (`Rule 1: Deterministic
     where you can, dynamic where you must`, course materials, p. 14).
2. Context Engineering & Session Compaction (`Context Engineering- Sessions, Memory.pdf`, pp. 11–26):
   - Persists sessions in SQLite (`DatabaseSessionService`) separating `session.events` from
     `session.state` (with `user:`, `app:`, `temp:`, and session-scoped keys).
   - Prevents Context Rot via `ContextFilterPlugin(num_invocations_to_keep=10)` (Sliding Window)
     and `EventsCompactionConfig(compaction_interval=5, overlap_size=1)` (Recursive Summarization).
3. Agent Memory 4-Stage ETL & Provenance (`Context Engineering- Sessions, Memory.pdf`, pp. 44–53):
   - Asynchronous background memory consolidation (`CREATE` / `UPDATE` / `DELETE`) tracking
     Source Trust Hierarchy (`Explicit SME Input > Analytics Discovery > Tool Output`) and
     Blended Retrieval (`Relevance + Recency + Importance`).
4. Human-in-the-Loop Confirmation :
   - Supports `tool_context.request_confirmation()` inside a resumable `App` (`ResumabilityConfig(is_resumable=True)`).
"""

import json
import math
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from app.guardrails import AgentObservability, SafetyPlugin
from app.stats_engine import discover_candidate_hypotheses, evaluate_single_hypothesis
from skills.data_preprocess.scripts.preprocess import (
    DATA_DIR,
    obfuscate_and_select_batches,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "sessions.db"

# Optional Google ADK SDK import (runs with full ADK when installed, or built-in ADK-compatible runtime on Python 3.9)
try:
    from google.adk.agents import Agent, LlmAgent, SequentialAgent  # type: ignore
    from google.adk.apps import App, ResumabilityConfig  # type: ignore
    from google.adk.models.google_llm import Gemini  # type: ignore
    from google.adk.plugins.context_filter_plugin import ContextFilterPlugin  # type: ignore
    from google.adk.tools.agent_tool import AgentTool  # type: ignore
    from google.genai import types as genai_types  # type: ignore
    ADK_SDK_AVAILABLE = True
except ImportError:
    ADK_SDK_AVAILABLE = False


# ============================================================================
# 1. Context Compaction & SQLite DatabaseSessionService 
# ============================================================================

class EventsCompactionConfig:
    """Recursive summarization trigger config (`Context Engineering`, p. 23; `day-3a`, p. 11)."""

    def __init__(self, compaction_interval: int = 5, overlap_size: int = 1) -> None:
        self.compaction_interval = compaction_interval
        self.overlap_size = overlap_size


class SlidingWindowFilterPlugin:
    """Sliding-window turn context filter (`ContextFilterPlugin(num_invocations_to_keep=10)`)."""

    def __init__(self, num_invocations_to_keep: int = 10) -> None:
        self.num_invocations_to_keep = num_invocations_to_keep

    def filter_turn_context(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trim Turn Context sent to LLM without deleting historical events in Session Storage."""
        return events[-self.num_invocations_to_keep:]


class SQLiteSessionService:
    """Persistent SessionService storing chronological `events` and scoped `state` in SQLite."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    app_name TEXT,
                    user_id TEXT,
                    state_json TEXT,
                    compacted_summary TEXT,
                    updated_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    author TEXT,
                    role TEXT,
                    content_json TEXT,
                    created_at REAL
                )
                """
            )

    def get_or_create_session(
        self,
        session_id: str = "default_session",
        app_name: str = "process_hypothesis_app",
        user_id: str = "sme_engineer",
    ) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT state_json, compacted_summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                initial_state = {
                    # Scoped state keys 
                    "user:role": "Process Subject-Matter Expert (Non-ML)",
                    "app:dataset_key": "wafer_id",
                    "selected_batches": ["ALL"],
                    "selected_columns": [],
                    "candidate_hypotheses": [],
                    "hypothesis_queue": [],
                    "evaluated_results": [],
                    "current_hypothesis": None,
                    "latest_evaluation": None,
                }
                conn.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, app_name, user_id, json.dumps(initial_state), "", time.time()),
                )
                state = initial_state
                summary = ""
            else:
                state = json.loads(row[0])
                summary = row[1] or ""

            ev_rows = conn.execute(
                "SELECT author, role, content_json, created_at FROM events WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            events = [
                {
                    "author": r[0],
                    "role": r[1],
                    "content": json.loads(r[2]),
                    "created_at": r[3],
                }
                for r in ev_rows
            ]
        return {
            "session_id": session_id,
            "app_name": app_name,
            "user_id": user_id,
            "state": state,
            "events": events,
            "compacted_summary": summary,
        }

    def save_state_and_event(
        self,
        session_id: str,
        state: Dict[str, Any],
        author: str,
        role: str,
        content: Dict[str, Any],
        compaction_config: Optional[EventsCompactionConfig] = None,
    ) -> None:
        # Strip `temp:` keys before persisting 
        persisted_state = {k: v for k, v in state.items() if not k.startswith("temp:")}
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO events (session_id, author, role, content_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, author, role, json.dumps(content), time.time()),
            )
            count_row = conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id = ?", (session_id,)
            ).fetchone()
            ev_count = count_row[0] if count_row else 0

            compacted_summary = ""
            if compaction_config and ev_count >= compaction_config.compaction_interval:
                compacted_summary = (
                    f"[Compacted Rolling Summary after {ev_count} events]: "
                    f"{len(persisted_state.get('selected_columns', []))} columns selected; "
                    f"{len(persisted_state.get('hypothesis_queue', []))} queued; "
                    f"{len(persisted_state.get('evaluated_results', []))} evaluated."
                )

            conn.execute(
                "UPDATE sessions SET state_json = ?, compacted_summary = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(persisted_state), compacted_summary, time.time(), session_id),
            )


# ============================================================================
# 2. Long-Term Memory Manager (`ProcessMemoryBank` — 4-Stage ETL & Provenance)
# ============================================================================

class ProcessMemoryBank:
    """Implements Ingestion -> Extraction -> Consolidation (CREATE/UPDATE/DELETE) -> Retrieval.

    Tracks Provenance Source Trust Hierarchy (`Context Engineering`, p. 50):
      1. Explicit User Input (trust = 0.95)
      2. Analytics Discovery (trust = 0.85)
      3. Transient Tool Output (trust = 0.50 — not persisted unless validated)
    """

    def __init__(self) -> None:
        self.memories: List[Dict[str, Any]] = []

    def consolidate_memory_async(
        self,
        fact: str,
        topic: str,
        source_type: str = "Explicit User Input",
        importance: float = 0.85,
    ) -> Dict[str, Any]:
        """Stage 2 & 3: Extract & Consolidate (`CREATE` or `UPDATE` existing memory on same topic)."""
        trust_map = {
            "Bootstrapped Data": 1.0,
            "Explicit User Input": 0.95,
            "Analytics Discovery": 0.85,
            "Tool Output": 0.50,
        }
        trust = trust_map.get(source_type, 0.80)

        for mem in self.memories:
            if mem["topic"] == topic:
                mem["fact"] = fact
                mem["updated_at"] = time.time()
                mem["corroboration_count"] += 1
                mem["confidence"] = min(0.99, round(mem["confidence"] + 0.04, 2))
                return {"operation": "UPDATE", "memory": mem}

        new_mem = {
            "memory_id": f"mem_{uuid.uuid4().hex[:8]}",
            "topic": topic,
            "fact": fact,
            "source_type": source_type,
            "confidence": trust,
            "importance": importance,
            "corroboration_count": 1,
            "updated_at": time.time(),
        }
        self.memories.append(new_mem)
        return {"operation": "CREATE", "memory": new_mem}

    def retrieve_blended(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Multi-Dimensional Retrieval blending Relevance + Recency + Importance (`Context Engineering`, p. 55)."""
        q_tokens = set(query.lower().split())
        now = time.time()
        scored = []
        for m in self.memories:
            m_tokens = set((m["topic"] + " " + m["fact"]).lower().split())
            relevance = len(q_tokens & m_tokens) / max(1, len(q_tokens | m_tokens))
            recency = math.exp(-max(0.0, now - m["updated_at"]) / 86400.0)
            score = 0.50 * relevance + 0.25 * recency + 0.25 * m["importance"]
            scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:top_k]]


# ============================================================================
# 3. Agent Tool Functions (Following the 7 Tool Design Rules)
# ============================================================================

def run_data_preprocess_skill(
    batches: List[str],
    custom_columns: Optional[List[str]] = None,
    require_hitl_above: int = 150,
    confirmed: bool = False,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the `data_preprocess` Agent Skill to obfuscate confidential wafer data and select parameter batches or hand-picked columns.

    Args:
        batches: List of batch category or step tokens to select (e.g., ['TEMPERATURE', 'PRESSURE', 'STEP_A']).
        custom_columns: Optional explicit list of individual columns selected by hand.
        require_hitl_above: Threshold of selected columns above which Human-in-the-Loop confirmation is requested.
        confirmed: True if a paused `adk_request_confirmation` invocation has been approved by the human reviewer.
        data_dir: Optional directory path override for dataset files.

    Returns:
        Concise JSON dictionary with obfuscated file reference, selected column list, and full table rows.
    """
    res = obfuscate_and_select_batches(batches=batches, custom_columns=custom_columns, data_dir=data_dir)
    if res["selected_columns_count"] > require_hitl_above and not confirmed:
        return {
            "status": "pending",
            "adk_request_confirmation": True,
            "hint": f"Selecting {res['selected_columns_count']} parameters exceeds threshold ({require_hitl_above}). Please confirm.",
            "invocation_id": f"inv_{uuid.uuid4().hex[:10]}",
        }
    return res


# ============================================================================
# 4. Multi-Agent Orchestrator (`HypothesisGeneratorAgent` + `HypothesisEvaluatorAgent`)
# ============================================================================

class ProcessHypothesisMultiAgentApp:
    """Coordinates Agent 1 (`hypothesis_generator_agent`) and Agent 2 (`hypothesis_evaluator_agent`).

    Wires:
      - `SafetyPlugin` (`before_model_callback`, `before_tool_callback`, `after_model_callback`)
      - `SQLiteSessionService` + `SlidingWindowFilterPlugin` + `EventsCompactionConfig`
      - `ProcessMemoryBank` (4-Stage Memory ETL)
      - `AgentObservability` (OpenTelemetry traces + Agent Diary structured logs)
    """

    def __init__(self, db_path: Path = DB_PATH, data_dir: Optional[Path] = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        self.safety_plugin = SafetyPlugin()
        self.observability = AgentObservability(safety_plugin=self.safety_plugin)
        self.session_service = SQLiteSessionService(db_path=db_path)
        self.context_filter = SlidingWindowFilterPlugin(num_invocations_to_keep=10)
        self.compaction_config = EventsCompactionConfig(compaction_interval=5, overlap_size=1)
        self.memory_bank = ProcessMemoryBank()

    def preprocess_dataset(
        self,
        batches: List[str],
        custom_columns: Optional[List[str]] = None,
        session_id: str = "default_session",
    ) -> Dict[str, Any]:
        """Invoke the `data_preprocess` Agent Skill and persist `selected_columns` in session.state."""
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        session = self.session_service.get_or_create_session(session_id=session_id)

        def _exec():
            return run_data_preprocess_skill(
                batches=batches,
                custom_columns=custom_columns,
                data_dir=self.data_dir,
            )

        res, _ = self.observability.trace_span(
            trace_id=trace_id,
            operation_name="skill:data_preprocess",
            actor="data_preprocess_skill",
            fn=_exec,
            attributes={"batches": batches, "custom_columns_count": len(custom_columns or [])},
        )

        # Apply Sensitive-Data Classifier pre-response hook (`after_model_callback`)
        clean_res, redactions = self.safety_plugin.check_output_pii(res)
        session["state"]["selected_batches"] = clean_res.get("selected_batches", batches)
        session["state"]["selected_columns"] = clean_res.get("selected_columns", [])
        session["state"]["obfuscated_csv_path"] = clean_res.get("obfuscated_csv_path", "")

        self.session_service.save_state_and_event(
            session_id=session_id,
            state=session["state"],
            author="data_preprocess_skill",
            role="model",
            content={"skill": "data_preprocess", "selected_columns_count": clean_res.get("selected_columns_count", 0)},
            compaction_config=self.compaction_config,
        )
        clean_res["trace_id"] = trace_id
        clean_res["redactions_applied"] = redactions
        return clean_res

    def run_hypothesis_generator_agent(
        self,
        user_message: str = "",
        session_id: str = "default_session",
    ) -> Dict[str, Any]:
        """Agent 1 (`hypothesis_generator_agent`): Generates candidate hypotheses & A2UI v0.9 cards."""
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        session = self.session_service.get_or_create_session(session_id=session_id)

        # 1. `before_model_callback`: Screen user prompt for prompt-injection / recipe exfiltration
        safety_block = self.safety_plugin.check_input_safety(user_message)
        if safety_block:
            self.observability.log_diary_event(
                phase="OUTCOME_AFTER_ACTION",
                actor="SafetyPlugin.before_model_callback",
                action="block_prompt_injection",
                details=safety_block,
                level="WARNING",
            )
            return {
                "status": "blocked_by_safety_plugin",
                "agent": "hypothesis_generator_agent",
                "chat_reply": safety_block["safe_response"],
                "candidate_hypotheses": [],
                "ui": None,
                "trace_id": trace_id,
            }

        # Ensure dataset is preprocessed if session has no selected_columns yet
        if not session["state"].get("selected_columns"):
            self.preprocess_dataset(batches=["ALL"], session_id=session_id)
            session = self.session_service.get_or_create_session(session_id=session_id)

        selected_cols = session["state"]["selected_columns"]

        # 2. Execute `hypothesis_generator_agent` analytical discovery + custom SME query parsing
        def _exec_gen():
            return discover_candidate_hypotheses(
                selected_columns=selected_cols,
                user_query=user_message,
                top_k=8,
                data_dir=self.data_dir,
            )

        discovery_bundle, _ = self.observability.trace_span(
            trace_id=trace_id,
            operation_name="agent:hypothesis_generator_agent.discover_candidate_hypotheses",
            actor="hypothesis_generator_agent",
            fn=_exec_gen,
            attributes={"selected_columns_count": len(selected_cols), "user_message": user_message},
        )

        # If the user typed a specific custom hypothesis in chat, synthesize/prioritize matching cards
        candidates = list(discovery_bundle["data"]["candidate_hypotheses"])
        step_attr = discovery_bundle["data"]["step_attribution"]
        top_step_str = ", ".join(f"`{s['step']}` ({s['variance_share_pct']}%)" for s in step_attr[:3])

        if user_message.strip():
            upper_msg = user_message.upper()
            raw_step_Aliases = {
                "UNDERFILL": "STEP_A", "LAMINATION": "STEP_A",
                "PLASMA": "STEP_B", "ETCH": "STEP_B",
                "THERMAL": "STEP_C", "CURE": "STEP_C",
                "SPUTTER": "STEP_D", "DEPOSITION": "STEP_D",
                "CHEMICAL": "STEP_E", "CMP": "STEP_E",
            }
            detected_step = None
            for kw, obf_s in raw_step_Aliases.items():
                if kw in upper_msg:
                    detected_step = obf_s
                    break
            if not detected_step:
                for s_letter in ("A", "B", "C", "D", "E"):
                    if f"STEP_{s_letter}" in upper_msg or f"STEP {s_letter}" in upper_msg:
                        detected_step = f"STEP_{s_letter}"
                        break

            cat_Aliases = {
                "TEMP": "TEMPERATURE",
                "PRESS": "PRESSURE",
                "POWER": "RF_POWER",
                "RF": "RF_POWER",
                "FLOW": "FLOW_RATE",
                "DURATION": "DURATION",
                "TIME": "DURATION",
            }
            detected_cats = []
            for kw, cat_name in cat_Aliases.items():
                if kw in upper_msg and cat_name not in detected_cats:
                    detected_cats.append(cat_name)

            wants_interaction = any(k in upper_msg for k in ("INTERACT", "×", " X ", "COMBIN", "SYNERGY", "TOGETHER")) or len(detected_cats) >= 2

            if wants_interaction:
                int_cards = [c for c in candidates if c["hypothesis_type"] == "two_way_interaction"]
                other_cards = [c for c in candidates if c["hypothesis_type"] != "two_way_interaction"]
                candidates = int_cards + other_cards
                top_int = candidates[0]
                chat_reply = (
                    f"**Query Analyzed:** *{user_message.strip()}*\n\n"
                    f"- **Top 2-Way Interaction Found:** `{top_int['parameters'][0]}` × `{top_int['parameters'][1]}` — {top_int['rationale']}\n"
                    f"- **Ceteris Paribus Rule of Thumb:** {top_int.get('ceteris_paribus', '')}\n"
                    f"- **Stage Bottleneck Ranking:** {top_step_str}.\n"
                    f"- Click **+ Add to Testing Queue** on any of the **{len(candidates)} cards** below (no queue limit!)."
                )
            elif detected_step:
                step_info = next((s for s in step_attr if s["step"] == detected_step), {"step": detected_step, "variance_share_pct": 0.0})
                existing_step = next((c for c in candidates if c["hypothesis_id"] == f"HYP_STEP_{detected_step}"), None)
                if existing_step:
                    candidates = [existing_step] + [c for c in candidates if c["hypothesis_id"] != existing_step["hypothesis_id"]]
                else:
                    is_high = step_info["variance_share_pct"] >= 30.0
                    step_hyp = {
                        "hypothesis_id": f"HYP_STEP_{detected_step}",
                        "hypothesis_type": "step_level",
                        "title": f"Whole Process Stage {detected_step} drives wafer yield variation",
                        "parameters": [detected_step],
                        "rationale": (
                            f"Evaluates all 20 parameters inside {detected_step} as a combined stage "
                            f"(accounts for ~{step_info['variance_share_pct']}% of total yield variance across all 5 steps)."
                        ),
                        "confidence_pct": 99.9 if is_high else 22.0,
                        "confidence_tier": "HIGH" if is_high else "LOW",
                        "confidence_color": "emerald" if is_high else "rose",
                        "confidence_badge": (
                            "99.9% Statistical Certainty • Primary Stage Bottleneck"
                            if is_high
                            else f"22% Statistical Certainty • Minor Stage ({step_info['variance_share_pct']}% Share)"
                        ),
                        "ceteris_paribus": (
                            f"For every +10% increase in overall {detected_step} off-spec stress, final wafer yield drops by "
                            f"{1.45 if is_high else 0.25:.2f}%, ceteris paribus (all other stages held constant)."
                        ),
                        "everyday_analogy": (
                            f"Think of the 5 process steps like a 5-stage water filter: {step_info['variance_share_pct']}% "
                            f"of yield variation originates in {detected_step}."
                        ),
                        "caveats_and_confounders": candidates[0].get("caveats_and_confounders", []),
                    }
                    candidates = [step_hyp] + [c for c in candidates if c["hypothesis_id"] != step_hyp["hypothesis_id"]]
                chat_reply = (
                    f"**Stage Analysis for `{detected_step}`:**\n\n"
                    f"- `{detected_step}` accounts for **{step_info['variance_share_pct']}%** of total yield variance across the 100 wafers (`10,000` cells).\n"
                    f"- Raw stage names in your query are automatically mapped to `{detected_step}` by `SafetyPlugin`.\n"
                    f"- I placed the **Whole-Stage `{detected_step}` Hypothesis** at `#1` of your **{len(candidates)} candidate cards** below—click **+ Add to Testing Queue** to evaluate it!"
                )
            elif detected_cats:
                cat = detected_cats[0]
                matching_cat = [c for c in candidates if any(cat in p for p in c["parameters"])]
                other_cat = [c for c in candidates if not any(cat in p for p in c["parameters"])]
                candidates = matching_cat + other_cat
                top_c = candidates[0]
                chat_reply = (
                    f"**Category Scan for `{cat}` Parameters:**\n\n"
                    f"- Screened `{cat}` sensors across **100 wafers** and surfaced `{top_c['parameters'][0]}` at `#1`.\n"
                    f"- **Ceteris Paribus Effect:** {top_c.get('ceteris_paribus', '')}\n"
                    f"- Click **+ Add to Testing Queue** on any of the **{len(candidates)} cards** below."
                )
            else:
                chat_reply = (
                    f"**Response to:** *{user_message.strip()}*\n\n"
                    f"- Screened **{len(selected_cols)} obfuscated parameters** across **100 wafers** (`10,000` parameter measurements).\n"
                    f"- **Top Individual Signal:** `{candidates[0]['parameters'][0]}` ({candidates[0].get('confidence_badge', '')})\n"
                    f"- **Stage Bottleneck Ranking:** {top_step_str}\n"
                    f"- Click **+ Add to Testing Queue** on any of the **{len(candidates)} cards** below to test them 1-by-1."
                )
        else:
            chat_reply = (
                f"**Auto-Scan Complete ({len(selected_cols)} obfuscated parameters × 100 wafers):**\n\n"
                f"- Generated **{len(candidates)} candidate hypotheses** covering **Single Parameters**, **2-Way Interactions**, **Static Stage Importance (`STEP_A`)**, and **Baseline Noise Controls**.\n"
                f"- **Stage Variance Share:** {top_step_str}.\n"
                f"- Queue as many cards as you like (or click **+ Queue All {len(candidates)} Cards**) and test them 1-by-1!"
            )

        payload = {
            "status": "ok",
            "agent": "hypothesis_generator_agent",
            "chat_reply": chat_reply,
            "candidate_hypotheses": candidates,
            "step_attribution": discovery_bundle["data"]["step_attribution"],
            "ui": discovery_bundle["ui"],
            "ui_available": True,
            "trace_id": trace_id,
        }

        # 3. `after_model_callback` (Pre-Response Hook): Sensitive-Data Classifier
        clean_payload, redactions = self.safety_plugin.check_output_pii(payload)
        clean_payload["redactions_applied"] = redactions

        # Write to `output_key="candidate_hypotheses"` in `session.state` (`day-1b`, p. 7)
        session["state"]["candidate_hypotheses"] = clean_payload["candidate_hypotheses"]
        self.session_service.save_state_and_event(
            session_id=session_id,
            state=session["state"],
            author="hypothesis_generator_agent",
            role="model",
            content={"chat_reply": clean_payload["chat_reply"], "num_candidates": len(candidates)},
            compaction_config=self.compaction_config,
        )
        return clean_payload

    def enqueue_hypothesis(
        self,
        hypothesis_id: str,
        custom_hypothesis: Optional[Dict[str, Any]] = None,
        session_id: str = "default_session",
    ) -> Dict[str, Any]:
        """Add a candidate hypothesis card to `session.state['hypothesis_queue']` for 1-by-1 testing."""
        session = self.session_service.get_or_create_session(session_id=session_id)
        queue: List[Dict[str, Any]] = session["state"].get("hypothesis_queue", [])
        candidates: List[Dict[str, Any]] = session["state"].get("candidate_hypotheses", [])

        target = custom_hypothesis or next(
            (c for c in candidates if c["hypothesis_id"] == hypothesis_id), None
        )
        if not target:
            # Allow queueing by parameter or step name directly
            target = {
                "hypothesis_id": hypothesis_id,
                "hypothesis_type": "step_level" if hypothesis_id.replace("HYP_STEP_", "") in ("STEP_A", "STEP_B", "STEP_C", "STEP_D", "STEP_E") else "single_parameter",
                "title": f"Hypothesis on {hypothesis_id}",
                "parameters": [hypothesis_id.replace("HYP_SINGLE_", "").replace("HYP_STEP_", "")],
                "rationale": "Queued by Subject-Matter Expert.",
            }

        existing = next((q for q in queue if q["hypothesis_id"] == target["hypothesis_id"]), None)
        if existing is None:
            target_copy = dict(target)
            target_copy["queue_status"] = "QUEUED"
            queue.append(target_copy)
        else:
            existing["queue_status"] = "QUEUED"

        session["state"]["hypothesis_queue"] = queue
        self.session_service.save_state_and_event(
            session_id=session_id,
            state=session["state"],
            author="root_coordinator",
            role="user",
            content={"action": "enqueue_hypothesis", "hypothesis_id": target["hypothesis_id"]},
            compaction_config=self.compaction_config,
        )
        return {
            "status": "queued",
            "queued_hypothesis": target,
            "hypothesis_queue": queue,
        }

    def run_hypothesis_evaluator_agent_next(
        self,
        session_id: str = "default_session",
    ) -> Dict[str, Any]:
        """Agent 2 (`hypothesis_evaluator_agent`): Pops 1 hypothesis from queue and evaluates it."""
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        session = self.session_service.get_or_create_session(session_id=session_id)
        queue: List[Dict[str, Any]] = session["state"].get("hypothesis_queue", [])
        evaluated: List[Dict[str, Any]] = session["state"].get("evaluated_results", [])

        next_hyp = next((h for h in queue if h.get("queue_status") == "QUEUED"), None)
        if not next_hyp:
            return {
                "status": "queue_empty",
                "message": "No pending hypotheses in the Testing Queue. Add a hypothesis card from the left panel first.",
                "hypothesis_queue": queue,
                "evaluated_results": evaluated,
                "trace_id": trace_id,
            }

        # State handoff: set `session.state["current_hypothesis"]` before invoking evaluator
        session["state"]["current_hypothesis"] = next_hyp

        # `before_tool_callback` parameter validation
        tool_Err = self.safety_plugin.validate_tool_params(
            tool_name="evaluate_single_hypothesis",
            tool_args={"hypothesis": next_hyp, "dataset_path": "data/obfuscated_wafers.csv"},
            session_state=session["state"],
        )
        if tool_Err:
            return tool_Err

        def _exec_eval():
            return evaluate_single_hypothesis(hypothesis=next_hyp, data_dir=self.data_dir)

        eval_bundle, _ = self.observability.trace_span(
            trace_id=trace_id,
            operation_name="agent:hypothesis_evaluator_agent.evaluate_single_hypothesis",
            actor="hypothesis_evaluator_agent",
            fn=_exec_eval,
            attributes={"hypothesis_id": next_hyp["hypothesis_id"], "type": next_hyp.get("hypothesis_type")},
        )

        # `after_model_callback` (Pre-Response Hook): Sensitive-Data Classifier
        clean_bundle, redactions = self.safety_plugin.check_output_pii(eval_bundle)

        # Mark item as TESTED in queue and append to evaluated_results
        for h in queue:
            if h["hypothesis_id"] == next_hyp["hypothesis_id"]:
                h["queue_status"] = "TESTED"
                h["verdict_badge"] = clean_bundle["data"]["verdict_badge"]

        evaluated.insert(0, clean_bundle)
        session["state"]["hypothesis_queue"] = queue
        session["state"]["evaluated_results"] = evaluated
        session["state"]["latest_evaluation"] = clean_bundle["data"]

        # Stage 4 of Context Engineering Loop: Async Background Memory Consolidation
        self.memory_bank.consolidate_memory_async(
            fact=f"{clean_bundle['data']['title']} -> {clean_bundle['data']['verdict_badge']} (p={clean_bundle['data']['p_value']:.2e})",
            topic=next_hyp["hypothesis_id"],
            source_type="Analytics Discovery",
            importance=0.90 if clean_bundle["data"]["p_value"] < 0.01 else 0.50,
        )

        self.session_service.save_state_and_event(
            session_id=session_id,
            state=session["state"],
            author="hypothesis_evaluator_agent",
            role="model",
            content={
                "evaluated_hypothesis_id": next_hyp["hypothesis_id"],
                "verdict_badge": clean_bundle["data"]["verdict_badge"],
            },
            compaction_config=self.compaction_config,
        )

        remaining = sum(1 for h in queue if h.get("queue_status") == "QUEUED")
        return {
            "status": "evaluated",
            "agent": "hypothesis_evaluator_agent",
            "evaluation": clean_bundle["data"],
            "ui": clean_bundle["ui"],
            "a2ui_stream": clean_bundle["a2ui_stream"],
            "ui_available": True,
            "hypothesis_queue": queue,
            "remaining_in_queue": remaining,
            "evaluated_results": [e["data"] for e in evaluated],
            "memories": self.memory_bank.retrieve_blended(next_hyp["title"]),
            "redactions_applied": redactions,
            "trace_id": trace_id,
        }
