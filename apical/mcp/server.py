"""Production-Grade Local Apical Plexus MCP Server.

Exposes native background RPC tools:
- plexus_route(query, top_k): Sub-millisecond routing across registered tool/skill catalogs in constant memory.
- plexus_learn(activation, target_class): Zero-retention weight adaptation.
- plexus_reinforce(query, tool_id): Real-time reward reinforcement on accepted assistant turns.
- plexus_verify_receipt(receipt_json): Validates Ed25519 signatures & ΔBytes == 0 attestation.
- plexus_stats(): Returns aggregate runtime metrics and token-saving analytics.

Air-Gapped Invariants:
- Zero raw user prompt or feature vector persistence
- Constant O(1) memory guarantees (persisted_bytes_delta == 0)
- Ed25519 cryptographic receipts verified offline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from apical.catalog import (
    DynamicToolRegistry,
    get_default_registry,
    get_tool_by_id,
)
from apical.enclave_client import EnclaveDirectClient, EnclaveRPCError
from apical.preference import DeveloperPreferenceAdapter
from apical.telemetry import (
    DEFAULT_TELEMETRY_LOG,
    ZeroRetentionTelemetryLogger,
    get_telemetry_logger,
)
from apical.verifier import verify_zero_retention_receipt

# Direct logging exclusively to stderr so stdout remains pristine JSON-RPC stream
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PlexusMCP] %(message)s",
)
logger = logging.getLogger("PlexusMCPServer")


class PlexusMcpServer:
    """Core engine and RPC dispatcher for the Apical Plexus MCP Server."""

    def __init__(
        self,
        enclave_client: Optional[EnclaveDirectClient] = None,
        telemetry_logger: Optional[ZeroRetentionTelemetryLogger] = None,
        registry: Optional[DynamicToolRegistry] = None,
    ) -> None:
        self.client = enclave_client or EnclaveDirectClient()
        self.telemetry = telemetry_logger or get_telemetry_logger()
        self.registry = registry or get_default_registry()
        self.preference_adapter = DeveloperPreferenceAdapter(
            enclave_client=self.client,
            telemetry_logger=self.telemetry,
        )

    def plexus_register_tools(self, tools: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Register or update custom tools in the dynamic registry."""
        count = self.registry.register_tools(tools)
        return {
            "status": "success",
            "registered_count": count,
            "total_registered_tools": len(self.registry),
            "registered_tools": [t["name"] for t in self.registry.list_tools()],
        }

    # --- Tool 1: plexus_route ---
    def plexus_route(
        self,
        query: str,
        top_k: int = 3,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Route natural language query across registered tools/skills in constant memory.

        Executes real dense embedding encoding and fixed-byte synaptic routing on the enclave.
        Calculates prompt tokens saved vs. naive static schema injection.
        """
        t0 = time.perf_counter()
        query_str = str(query).strip()

        # If caller provides active tools, dynamically index them
        if tools:
            self.registry.register_tools(tools)
            candidates = list(tools)
        else:
            candidates = self.registry.list_tools()

        if not candidates:
            return {
                "status": "no_tools_registered",
                "error": "No tools currently registered. Provide a 'tools' array in plexus_route or register them with plexus_register_tools.",
                "routes": [],
                "tokens_saved": 0,
            }

        # Format tool signatures for the enclave
        tool_signatures: List[Dict[str, Any]] = []
        for c in candidates:
            fn = c.get("function", c)
            tool_signatures.append({
                "name": str(fn.get("name", "")),
                "description": str(fn.get("description", "")),
            })

        # Route through enclave using real neural embeddings and ONNX INT8 AVX2
        result = self.client.route_tools(query=query_str, tools=tool_signatures, top_k=top_k)
        latency_us = (time.perf_counter() - t0) * 1_000_000.0

        routed_indices = result.get("routed_indices", [])
        scores = result.get("scores", [])
        receipt = result.get("receipt", {})

        enriched_routes: List[Dict[str, Any]] = []
        for idx, score in zip(routed_indices, scores):
            if idx < len(candidates):
                c = candidates[idx]
                enriched_routes.append({
                    "name": tool_signatures[idx]["name"],
                    "description": tool_signatures[idx]["description"],
                    "affinity": score,
                    "schema": c.get("schema", c),
                    "category": c.get("category", "general"),
                })

        savings = self.registry.calculate_token_savings(enriched_routes, candidates)
        top_rule_name = enriched_routes[0]["name"] if enriched_routes else "none"
        top_conf = enriched_routes[0]["affinity"] if enriched_routes else None

        input_digest = ZeroRetentionTelemetryLogger.compute_sha256(query_str)

        self.telemetry.log_operation(
            operation="route",
            model_id="dynamic_tools",
            latency_us=latency_us,
            receipt=receipt,
            input_digest=input_digest,
            top_rule_id=None,
            confidence=top_conf,
            tokens_saved=savings["tokens_saved"],
        )

        return {
            "status": "success",
            "routes": enriched_routes,
            "top_tool": enriched_routes[0] if enriched_routes else None,
            "token_savings": savings,
            "stage_timings_us": result.get("stage_timings_us", {}),
            "receipt": receipt,
        }

    # --- Tool 2: plexus_learn ---
    def plexus_learn(
        self,
        activation: str,
        target_class: int,
        tenant_id: str = "sandbox",
        model_id: str = "sandbox_router",
    ) -> Dict[str, Any]:
        """Zero-retention online synaptic weight adaptation.

        Updates the fixed-byte container in-place (persisted_bytes_delta == 0),
        scrubs input buffers, and returns signed Ed25519 receipt.
        """
        t0 = time.perf_counter()
        act_str = str(activation).strip()
        input_digest = ZeroRetentionTelemetryLogger.compute_sha256(act_str)

        resp = self.client.learn(
            tenant_id=tenant_id,
            model_id=model_id,
            query=act_str,
            target_class=int(target_class),
        )
        latency_us = (time.perf_counter() - t0) * 1_000_000.0

        receipt = resp.get("receipt", {})
        delta = int(receipt.get("persisted_bytes_delta", 0))

        # Log zero-retention telemetry
        self.telemetry.log_operation(
            operation="learn",
            model_id=model_id,
            latency_us=latency_us,
            receipt=receipt,
            input_digest=input_digest,
            top_rule_id=int(target_class),
            confidence=resp.get("confidence"),
            tokens_saved=0,
        )

        return {
            "status": resp.get("status", "updated"),
            "target_class": int(target_class),
            "winner_class": resp.get("winner_class"),
            "confidence": resp.get("confidence"),
            "update_count": resp.get("update_count", 0),
            "persisted_bytes_delta": delta,
            "latency_ms": round(latency_us / 1000.0, 3),
            "receipt": receipt,
        }

    # --- Tool 3: plexus_reinforce ---
    def plexus_reinforce(
        self,
        query: str,
        tool_id: Optional[int] = None,
        rule_id: Optional[int] = None,
        tenant_id: str = "sandbox",
        model_id: str = "sandbox_router",
    ) -> Dict[str, Any]:
        """Reinforce a successful assistant turn with real-time reward online.

        Strengthens the synaptic pathway between query semantic embedding and target tool.
        """
        target = tool_id if tool_id is not None else rule_id
        if target is None:
            raise ValueError("Either tool_id or rule_id must be provided")

        t0 = time.perf_counter()
        query_str = str(query).strip()
        input_digest = ZeroRetentionTelemetryLogger.compute_sha256(query_str)

        resp = self.client.reinforce(
            tenant_id=tenant_id,
            model_id=model_id,
            query=query_str,
            target_rule=int(target),
        )
        latency_us = (time.perf_counter() - t0) * 1_000_000.0

        receipt = resp.get("receipt", {})
        tool_def = get_tool_by_id(int(tool_id))
        tool_name = tool_def.name if tool_def else f"rule_{tool_id}"

        self.telemetry.log_operation(
            operation="reinforce",
            model_id=model_id,
            latency_us=latency_us,
            receipt=receipt,
            input_digest=input_digest,
            top_rule_id=int(tool_id),
            confidence=resp.get("confidence"),
            tokens_saved=0,
        )

        return {
            "status": "reinforced",
            "target_rule": int(tool_id),
            "tool_name": tool_name,
            "confidence": resp.get("confidence"),
            "update_count": resp.get("update_count", 0),
            "persisted_bytes_delta": int(receipt.get("persisted_bytes_delta", 0)),
            "latency_ms": round(latency_us / 1000.0, 3),
            "receipt": receipt,
        }

    # --- Tool 4: plexus_verify_receipt ---
    def plexus_verify_receipt(self, receipt_json: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Cryptographically verify an Ed25519 execution receipt and assert zero retention."""
        if isinstance(receipt_json, str):
            try:
                receipt = json.loads(receipt_json)
            except json.JSONDecodeError as e:
                return {
                    "verified": False,
                    "error": f"Invalid JSON string: {e}",
                    "persisted_bytes_delta": None,
                }
        else:
            receipt = receipt_json

        t0 = time.perf_counter()
        try:
            is_valid = verify_zero_retention_receipt(receipt)
            latency_us = (time.perf_counter() - t0) * 1_000_000.0

            return {
                "verified": is_valid,
                "receipt_id": receipt.get("receipt_id"),
                "operation": receipt.get("operation"),
                "persisted_bytes_delta": receipt.get("persisted_bytes_delta"),
                "input_buffer_zeroed": receipt.get("input_buffer_zeroed"),
                "enclave_public_key_hex": receipt.get("enclave_public_key_hex"),
                "attestation_mode": receipt.get("attestation_mode"),
                "verification_latency_us": round(latency_us, 1),
            }
        except Exception as e:
            return {
                "verified": False,
                "error": str(e),
                "persisted_bytes_delta": receipt.get("persisted_bytes_delta"),
            }

    # --- Tool 5: plexus_stats ---
    def plexus_stats(self) -> Dict[str, Any]:
        """Return aggregate runtime metrics, enclave health, and token savings analytics."""
        t0 = time.perf_counter()
        try:
            ping_data = self.client.ping()
            enclave_status = "ONLINE"
        except Exception as e:
            ping_data = {"error": str(e)}
            enclave_status = "OFFLINE"

        telemetry_summary = self.telemetry.compute_summary()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "status": "ready",
            "enclave_status": enclave_status,
            "enclave_details": ping_data,
            "registered_tools_count": len(self.registry),
            "registered_tools": [t["name"] for t in self.registry.list_tools()[:20]],
            "telemetry": telemetry_summary,
            "query_latency_ms": round(latency_ms, 3),
        }

    # --- Tool 7: plexus_wipe_telemetry ---
    def plexus_wipe_telemetry(self, backup: bool = False) -> Dict[str, Any]:
        """Atomically wipe/purge local zero-retention telemetry log while preserving zero-retention invariants."""
        return self.telemetry.wipe(backup=backup)

    # --- Dispatcher for MCP Protocol / Direct Invocation ---
    def dispatch_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool name and arguments to internal implementations."""
        if tool_name == "plexus_route":
            query = arguments.get("query", "")
            top_k = int(arguments.get("top_k", 3))
            tools = arguments.get("tools")
            return self.plexus_route(query=query, top_k=top_k, tools=tools)

        elif tool_name == "plexus_register_tools":
            tools = arguments.get("tools", [])
            return self.plexus_register_tools(tools=tools)

        elif tool_name == "plexus_learn":
            activation = arguments.get("activation", "")
            target_class = int(arguments.get("target_class", 0))
            return self.plexus_learn(activation=activation, target_class=target_class)

        elif tool_name == "plexus_reinforce":
            query = arguments.get("query", "")
            tool_id = int(arguments.get("tool_id", 0))
            return self.plexus_reinforce(query=query, tool_id=tool_id)

        elif tool_name == "plexus_verify_receipt":
            receipt = arguments.get("receipt_json") or arguments.get("receipt", {})
            return self.plexus_verify_receipt(receipt_json=receipt)

        elif tool_name == "plexus_stats":
            return self.plexus_stats()

        elif tool_name == "plexus_wipe_telemetry":
            backup = bool(arguments.get("backup", False))
            return self.plexus_wipe_telemetry(backup=backup)

        elif tool_name == "plexus_preference_classify":
            text = arguments.get("text", "")
            top_k = int(arguments.get("top_k", 3))
            matches = self.preference_adapter.classify(text=text, top_k=top_k)
            return {"matches": [asdict_match(m) for m in matches]}

        elif tool_name == "plexus_preference_learn":
            correction = arguments.get("correction", "")
            dim_id = int(arguments.get("dimension_id", 0))
            return self.preference_adapter.learn_correction(correction_text=correction, preferred_dimension_id=dim_id)

        else:
            raise ValueError(f"Unknown tool: {tool_name}")


def asdict_match(m: Any) -> Dict[str, Any]:
    return {
        "dimension_id": m.dimension_id,
        "name": m.name,
        "affinity": m.affinity,
        "directive": m.directive,
    }


def get_mcp_tool_definitions() -> List[Dict[str, Any]]:
    """Return JSON Schema definitions for MCP tools/list."""
    return [
        {
            "name": "plexus_route",
            "description": "Sub-millisecond semantic routing across registered tools/skills in constant memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language request or task description."},
                    "top_k": {"type": "integer", "description": "Number of top rules/tools to activate (default 3).", "default": 3},
                    "tools": {
                        "type": "array",
                        "description": "Optional list of candidate tool schemas to route. If omitted, routes across automatically discovered and registered local tools.",
                        "items": {"type": "object"},
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "plexus_register_tools",
            "description": "Register custom or discovered tool schemas into the Plexus dynamic registry for semantic routing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tools": {
                        "type": "array",
                        "description": "Array of tool schema objects to register.",
                        "items": {"type": "object"},
                    }
                },
                "required": ["tools"],
            },
        },
        {
            "name": "plexus_learn",
            "description": "Zero-retention online continual adaptation on synaptic weights (persisted_bytes_delta == 0).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "activation": {"description": "Input dense feature vector or text query.", "type": ["array", "string"]},
                    "target_class": {"type": "integer", "description": "Target class or rule identifier to adapt towards."},
                },
                "required": ["activation", "target_class"],
            },
        },
        {
            "name": "plexus_reinforce",
            "description": "Real-time reward reinforcement on accepted assistant turns under zero retention.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"description": "Natural language query or feature vector of the accepted turn.", "type": ["string", "array"]},
                    "tool_id": {"type": "integer", "description": "Identifier of the tool that resolved the turn."},
                },
                "required": ["query", "tool_id"],
            },
        },
        {
            "name": "plexus_verify_receipt",
            "description": "Cryptographically verifies Ed25519 signature and zero-retention invariant (persisted_bytes_delta == 0).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "receipt_json": {"description": "JSON string or dictionary of the execution receipt.", "type": ["string", "object"]},
                },
                "required": ["receipt_json"],
            },
        },
        {
            "name": "plexus_stats",
            "description": "Returns aggregate runtime statistics, latency distribution, tokens saved, and attestation status.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "plexus_wipe_telemetry",
            "description": "Atomically wipes/purges the local zero-retention telemetry log file, resetting operations counter and lifetime tokens saved.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "backup": {
                        "type": "boolean",
                        "description": "Whether to copy existing log to .bak before wiping (default false).",
                        "default": False,
                    }
                },
            },
        },
    ]


def run_mcp_stdio_server() -> None:
    """Run MCP server over standard input/output with line-delimited JSON-RPC."""
    server = PlexusMcpServer()
    logger.info("Starting Apical Plexus MCP Stdio Server...")

    for line in sys.stdin:
        line_str = line.strip()
        if not line_str:
            continue

        try:
            req = json.loads(line_str)
        except json.JSONDecodeError as e:
            err_resp = {"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {e}"}, "id": None}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        # Handle MCP notifications (no id)
        if req_id is None:
            if method == "notifications/initialized":
                logger.info("Client acknowledged initialized notification.")
            continue

        # Handle MCP JSON-RPC requests
        try:
            if method == "initialize":
                res = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "apical-plexus", "version": "0.5.5"},
                }
                out = {"jsonrpc": "2.0", "id": req_id, "result": res}

            elif method == "ping":
                out = {"jsonrpc": "2.0", "id": req_id, "result": {}}

            elif method == "tools/list":
                out = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": get_mcp_tool_definitions()}}

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                tool_result = server.dispatch_tool(tool_name, arguments)
                out = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(tool_result, indent=2)}],
                        "isError": False,
                    },
                }

            # Direct fallback for plain JSON-RPC commands
            elif method in ("plexus_route", "plexus_learn", "plexus_reinforce", "plexus_verify_receipt", "plexus_stats"):
                tool_result = server.dispatch_tool(method, params)
                out = {"jsonrpc": "2.0", "id": req_id, "result": tool_result}

            else:
                out = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except Exception as e:
            logger.exception("Error executing MCP method %s: %s", method, e)
            out = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }

        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="apical-mcp",
        description="Apical Plexus MCP Server & Telemetry Dashboard",
    )
    parser.add_argument("-v", "--version", action="version", version="Apical Plexus MCP Daemon v0.5.5")
    parser.add_argument("--dashboard", action="store_true", help="Print ANSI/Markdown telemetry dashboard")
    parser.add_argument("--stats", action="store_true", help="Print JSON statistics summary")
    parser.add_argument("--wipe", action="store_true", help="Atomically wipe telemetry log")
    parser.add_argument("--backup", action="store_true", help="Preserve .bak copy before wiping")
    args = parser.parse_args()

    if args.wipe:
        server = PlexusMcpServer()
        res = server.plexus_wipe_telemetry(backup=args.backup)
        print(json.dumps(res, indent=2))
        sys.exit(0)

    if args.dashboard:
        logger_instance = get_telemetry_logger()
        print(logger_instance.render_markdown_dashboard())
        sys.exit(0)

    if args.stats:
        server = PlexusMcpServer()
        print(json.dumps(server.plexus_stats(), indent=2))
        sys.exit(0)

    if sys.stdin.isatty():
        sys.stderr.write(
            "Apical Plexus MCP Server (v0.5.5) listening for JSON-RPC on stdio.\n"
            "Waiting for client requests... (Press Ctrl+C to stop)\n\n"
            "Tip: Run with '--dashboard' to view telemetry, '--stats' for JSON statistics, or '--help' for options.\n"
        )
        sys.stderr.flush()

    run_mcp_stdio_server()


if __name__ == "__main__":
    main()
