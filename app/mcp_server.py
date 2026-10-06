#!/usr/bin/env python3
"""Zero-Dependency Model Context Protocol (MCP) Stdio Server for Process Hypothesis Tools.

Design Principles Invoked Verbatim (the Google 5-Day AI Agents Intensive whitepapers):
1. MCP Client & Server Architecture (`Agent Tools & Interoperability with MCP.pdf`, pp. 9–24):
   - Exposes `run_data_preprocess_skill`, `discover_candidate_hypotheses`, and
     `evaluate_single_hypothesis` over standard JSON-RPC 2.0 stdio transport (`N + M` interop).
   - Returns structured `isError: true` descriptive tool guidance when invalid parameters are
     supplied (`Provide descriptive error messages`, p. 20).
"""

import json
import sys
from typing import Any, Dict

from app.stats_engine import discover_candidate_hypotheses, evaluate_single_hypothesis
from skills.data_preprocess.scripts.preprocess import obfuscate_and_select_batches

MCP_TOOLS = [
    {
        "name": "run_data_preprocess_skill",
        "description": "Obfuscates confidential wafer dataset (mapping raw steps like UNDERFILL_LAMINATION to STEP_A..STEP_E, stripping RECIPE_ID, normalizing to [0,1]) and selects parameter batches.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "batches": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Batch tokens such as TEMPERATURE, PRESSURE, STEP_A, or ALL.",
                }
            },
            "required": ["batches"],
        },
    },
    {
        "name": "discover_candidate_hypotheses",
        "description": "Scans selected obfuscated columns for 1-way main effects, 2-way interactions, and static step-level bottlenecks, returning A2UI v0.9 cards.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_query": {"type": "string", "description": "Optional SME domain query."}
            },
        },
    },
    {
        "name": "evaluate_single_hypothesis",
        "description": "Evaluates a single queued hypothesis with p-values, Benjamini-Hochberg FDR, ML tree-split importance, and Good/Bad Wafer Distribution charts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string"},
                "hypothesis_type": {"type": "string"},
                "parameters": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["hypothesis_id", "parameters"],
        },
    },
]


def handle_jsonrpc_request(req: Dict[str, Any]) -> Dict[str, Any]:
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "process-hypothesis-mcp", "version": "0.1.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS}}
    if method == "tools/call":
        t_name = params.get("name")
        t_args = params.get("arguments") or {}
        try:
            if t_name == "run_data_preprocess_skill":
                res = obfuscate_and_select_batches(batches=t_args.get("batches", ["ALL"]))
            elif t_name == "discover_candidate_hypotheses":
                res = discover_candidate_hypotheses(user_query=t_args.get("user_query", ""))
            elif t_name == "evaluate_single_hypothesis":
                res = evaluate_single_hypothesis(hypothesis=t_args)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "isError": True,
                        "content": [{
                            "type": "text",
                            "text": (
                                f"Unknown tool '{t_name}'. Available tools: "
                                f"{', '.join(t['name'] for t in MCP_TOOLS)}."
                            ),
                        }],
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "isError": False,
                    "content": [{"type": "text", "text": json.dumps(res)}],
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Tool execution error: {exc}"}],
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        resp = handle_jsonrpc_request(req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
