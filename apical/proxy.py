"""Apical Plexus In-Line LLM Schema Pruning Reverse Proxy (apical-proxy).

Sits transparently between developer IDEs/agents (Cursor, Continue, Aider, LangChain)
and upstream model providers (OpenAI, Anthropic, local vLLM).

Intercepts prompt requests containing dozens of tool/function schemas, extracts the user's
current turn query, routes against the local zero-retention Apical Enclave in < 10 ms,
prunes non-relevant tool schemas, records cryptographic zero-retention telemetry,
and forwards the streamlined payload upstream.

Compatible with:
- OpenAI API: POST /v1/chat/completions (streaming SSE and non-streaming)
- Anthropic API: POST /v1/messages (streaming SSE and non-streaming)
- Transparent passthrough for all other endpoints (/v1/models, /v1/embeddings)
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from apical.catalog import get_default_registry
from apical.enclave_client import EnclaveDirectClient, EnclaveRPCError
from apical.telemetry import (
    DEFAULT_TELEMETRY_LOG,
    ZeroRetentionTelemetryLogger,
    get_telemetry_logger,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [apical-proxy] %(message)s",
)
logger = logging.getLogger("ApicalProxy")

HOP_BY_HOP_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
}


def extract_user_query(messages: List[Dict[str, Any]]) -> str:
    """Extract natural language text of the latest user turn from messages array."""
    if not messages:
        return ""

    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            elif isinstance(content, list):
                # Handle OpenAI / Anthropic multi-part content blocks
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                    elif isinstance(part, str):
                        parts.append(part)
                return " ".join(parts).strip()
    return ""


def extract_gemini_user_query(contents: List[Dict[str, Any]]) -> str:
    """Extract natural language text of the latest user turn from Gemini contents array."""
    if not contents:
        return ""

    for item in reversed(contents):
        role = item.get("role")
        if role in ("user", None, ""):
            parts = item.get("parts", [])
            text_parts = []
            for p in parts:
                if isinstance(p, dict) and "text" in p:
                    text_parts.append(str(p["text"]))
                elif isinstance(p, str):
                    text_parts.append(p)
            if text_parts:
                return " ".join(text_parts).strip()
    return ""


def calculate_tools_tokens(tools: List[Dict[str, Any]]) -> int:
    """Calculate approximate token footprint for a list of tool schemas."""
    total_chars = sum(len(json.dumps(t, separators=(",", ":"))) for t in tools)
    return max(1, total_chars // 4)


class PlexusProxyHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler that prunes tool schemas before forwarding upstream."""

    enclave_client: EnclaveDirectClient
    telemetry: ZeroRetentionTelemetryLogger
    openai_upstream: str
    anthropic_upstream: str
    gemini_upstream: str
    top_k: int
    min_tools_to_prune: int
    embedding_model: Optional[str]
    http_client: httpx.Client

    def _extract_model_name(self, body: Dict[str, Any]) -> str:
        """Extract model identifier from path or body."""
        if "models/" in self.path:
            # Extract from Gemini URL: e.g. /v1beta/models/gemini-1.5-pro:generateContent
            parts = self.path.split("models/")
            if len(parts) > 1:
                return parts[1].split(":")[0]
        return str(body.get("model", "unknown_model"))

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests (e.g. /health or passthrough to upstream)."""
        if self.path in ("/health", "/healthz", "/"):
            status_data = {
                "status": "online",
                "service": "apical-proxy",
                "top_k": self.top_k,
                "min_tools_to_prune": self.min_tools_to_prune,
            }
            try:
                ping = self.enclave_client.ping()
                status_data["enclave"] = "connected"
                status_data["enclave_pubkey"] = ping.get("enclave_public_key_hex", "")
            except Exception as e:
                status_data["enclave"] = f"unreachable: {e}"

            self._send_json(200, status_data)
            return

        self._forward_request("GET")

    def do_POST(self) -> None:
        """Handle POST requests with dynamic tool schema pruning."""
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len) if content_len > 0 else b""

        # Check if this is an endpoint with candidate tools to prune
        is_openai_chat = self.path.endswith("/chat/completions")
        is_anthropic_msg = self.path.endswith("/messages")
        is_gemini_gen = ":generateContent" in self.path or ":streamGenerateContent" in self.path

        if (is_openai_chat or is_anthropic_msg or is_gemini_gen) and raw_body:
            try:
                body = json.loads(raw_body.decode("utf-8"))
                pruned_body = self._maybe_prune_tools(
                    body,
                    is_anthropic=is_anthropic_msg,
                    is_gemini=is_gemini_gen,
                )
                forward_bytes = json.dumps(pruned_body).encode("utf-8")
                self._forward_request("POST", body_bytes=forward_bytes)
                return
            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON request body; forwarding unmodified")
            except Exception as e:
                logger.exception("Error during tool pruning; falling back to original request: %s", e)

        # Fallback passthrough
        self._forward_request("POST", body_bytes=raw_body)

    def _maybe_prune_tools(
        self,
        body: Dict[str, Any],
        is_anthropic: bool = False,
        is_gemini: bool = False,
    ) -> Dict[str, Any]:
        """Prune tools array using the live enclave router."""
        if is_gemini:
            tools_list = body.get("tools")
            if not isinstance(tools_list, list) or not tools_list:
                return body

            all_decls: List[Dict[str, Any]] = []
            for t_block in tools_list:
                if isinstance(t_block, dict):
                    decls = t_block.get("function_declarations", [])
                    if isinstance(decls, list):
                        all_decls.extend(decls)

            if len(all_decls) < self.min_tools_to_prune:
                return body

            contents = body.get("contents", [])
            user_query = extract_gemini_user_query(contents)
            if not user_query:
                return body

            tool_signatures = [
                {"name": str(d.get("name", "")), "description": str(d.get("description", ""))}
                for d in all_decls
            ]
            try:
                get_default_registry().register_tools(all_decls)
            except Exception:
                pass

            t0 = time.perf_counter()
            try:
                route_res = self.enclave_client.route_tools(
                    query=user_query,
                    tools=tool_signatures,
                    top_k=self.top_k,
                    embedding_model=self.embedding_model,
                )
                latency_us = (time.perf_counter() - t0) * 1_000_000.0
                routed_indices = route_res.get("routed_indices", [])
                receipt = route_res.get("receipt", {})
                scores = route_res.get("scores", [])

                if routed_indices:
                    pruned_decls = [all_decls[i] for i in routed_indices if i < len(all_decls)]
                    orig_tokens = calculate_tools_tokens(all_decls)
                    pruned_tokens = calculate_tools_tokens(pruned_decls)
                    tokens_saved = max(0, orig_tokens - pruned_tokens)
                    savings_pct = round((tokens_saved / max(1, orig_tokens)) * 100.0, 1)

                    logger.info(
                        "Pruned Gemini %d decls -> %d decls in %.1f ms | Saved %d tokens (%s%%) | Top: %s",
                        len(all_decls),
                        len(pruned_decls),
                        latency_us / 1000.0,
                        tokens_saved,
                        savings_pct,
                        tool_signatures[routed_indices[0]]["name"] if routed_indices else "none",
                    )

                    model_name = self._extract_model_name(body)
                    input_digest = ZeroRetentionTelemetryLogger.compute_sha256(user_query)

                    self.telemetry.log_operation(
                        operation="proxy_prune",
                        model_id=model_name,
                        latency_us=latency_us,
                        receipt=receipt,
                        input_digest=input_digest,
                        top_rule_id=routed_indices[0] if routed_indices else None,
                        confidence=scores[0] if scores else None,
                        tokens_saved=tokens_saved,
                    )

                    body["tools"] = [{"function_declarations": pruned_decls}]
            except Exception as e:
                logger.warning("Error during Gemini tool pruning: %s (forwarding all decls)", e)

            return body

        # OpenAI / Anthropic format
        tools = body.get("tools")
        if not isinstance(tools, list) or len(tools) < self.min_tools_to_prune:
            return body

        messages = body.get("messages", [])
        user_query = extract_user_query(messages)
        if not user_query:
            return body

        # Extract tool signatures for semantic routing
        tool_signatures: List[Dict[str, Any]] = []
        for t in tools:
            if is_anthropic:
                name = str(t.get("name", ""))
                desc = str(t.get("description", ""))
            else:
                fn = t.get("function", t)
                name = str(fn.get("name", ""))
                desc = str(fn.get("description", ""))
            tool_signatures.append({"name": name, "description": desc})
        try:
            get_default_registry().register_tools(tools)
        except Exception:
            pass

        t0 = time.perf_counter()
        try:
            route_res = self.enclave_client.route_tools(
                query=user_query,
                tools=tool_signatures,
                top_k=self.top_k,
                embedding_model=self.embedding_model,
            )
            latency_us = (time.perf_counter() - t0) * 1_000_000.0
            routed_indices = route_res.get("routed_indices", [])
            receipt = route_res.get("receipt", {})
            scores = route_res.get("scores", [])

            if routed_indices:
                pruned_tools = [tools[i] for i in routed_indices if i < len(tools)]
                orig_tokens = calculate_tools_tokens(tools)
                pruned_tokens = calculate_tools_tokens(pruned_tools)
                tokens_saved = max(0, orig_tokens - pruned_tokens)
                savings_pct = round((tokens_saved / max(1, orig_tokens)) * 100.0, 1)

                logger.info(
                    "Pruned %d tools -> %d tools in %.1f ms | Saved %d tokens (%s%%) | Top: %s",
                    len(tools),
                    len(pruned_tools),
                    latency_us / 1000.0,
                    tokens_saved,
                    savings_pct,
                    tool_signatures[routed_indices[0]]["name"] if routed_indices else "none",
                )

                # Log zero-retention telemetry
                model_name = self._extract_model_name(body)
                input_digest = ZeroRetentionTelemetryLogger.compute_sha256(user_query)

                self.telemetry.log_operation(
                    operation="proxy_prune",
                    model_id=model_name,
                    latency_us=latency_us,
                    receipt=receipt,
                    input_digest=input_digest,
                    top_rule_id=routed_indices[0] if routed_indices else None,
                    confidence=scores[0] if scores else None,
                    tokens_saved=tokens_saved,
                )

                body["tools"] = pruned_tools
        except EnclaveRPCError as e:
            logger.warning("Enclave tool routing RPC error: %s (forwarding all tools)", e)
        except Exception as e:
            logger.exception("Unexpected error in route_tools: %s (forwarding all tools)", e)

        return body

    def _forward_request(self, method: str, body_bytes: Optional[bytes] = None) -> None:
        """Forward HTTP request upstream and stream response back to client."""
        # Determine upstream URL
        if ":generateContent" in self.path or ":streamGenerateContent" in self.path:
            base_url = self.gemini_upstream.rstrip("/")
        elif "/messages" in self.path:
            base_url = self.anthropic_upstream.rstrip("/")
        else:
            base_url = self.openai_upstream.rstrip("/")

        target_url = f"{base_url}{self.path}"

        # Clean headers
        req_headers = {}
        for k, v in self.headers.items():
            if k.lower() not in HOP_BY_HOP_HEADERS:
                req_headers[k] = v

        try:
            req = self.http_client.build_request(
                method=method,
                url=target_url,
                headers=req_headers,
                content=body_bytes,
                timeout=120.0,
            )

            upstream_resp = self.http_client.send(req, stream=True)

            # Send response headers (exclude content-encoding since iter_bytes() decompresses on the fly)
            self.send_response(upstream_resp.status_code)
            for k, v in upstream_resp.headers.items():
                k_lower = k.lower()
                if k_lower not in HOP_BY_HOP_HEADERS and k_lower != "content-encoding":
                    self.send_header(k, v)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Stream body chunks to client
            for chunk in upstream_resp.iter_bytes(chunk_size=4096):
                self.wfile.write(chunk)
                self.wfile.flush()

            upstream_resp.close()

        except httpx.RequestError as e:
            logger.error("Upstream connection failure: %s", e)
            self._send_json(502, {"error": f"Upstream proxy gateway error: {e}"})

    def _send_json(self, status: int, data: Dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default BaseHTTPRequestHandler access log to keep console clean."""
        return


def create_proxy_server(
    port: int = 8080,
    host: str = "127.0.0.1",
    openai_upstream: str = "https://api.openai.com/v1",
    anthropic_upstream: str = "https://api.anthropic.com/v1",
    gemini_upstream: str = "https://generativelanguage.googleapis.com",
    top_k: int = 5,
    min_tools_to_prune: int = 6,
    embedding_model: Optional[str] = None,
    enclave_client: Optional[EnclaveDirectClient] = None,
    telemetry_logger: Optional[ZeroRetentionTelemetryLogger] = None,
) -> ThreadingHTTPServer:
    """Instantiate and configure the ThreadingHTTPServer proxy."""
    client = enclave_client or EnclaveDirectClient()
    telemetry = telemetry_logger or get_telemetry_logger()
    http_transport = httpx.HTTPTransport(retries=1)
    shared_http = httpx.Client(transport=http_transport, timeout=120.0)

    class CustomHandler(PlexusProxyHandler):
        pass

    CustomHandler.enclave_client = client
    CustomHandler.telemetry = telemetry
    CustomHandler.openai_upstream = openai_upstream
    CustomHandler.anthropic_upstream = anthropic_upstream
    CustomHandler.gemini_upstream = gemini_upstream
    CustomHandler.top_k = top_k
    CustomHandler.min_tools_to_prune = min_tools_to_prune
    CustomHandler.embedding_model = embedding_model
    CustomHandler.http_client = shared_http

    server = ThreadingHTTPServer((host, port), CustomHandler)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Apical Plexus In-Line LLM Schema Pruning Reverse Proxy")
    parser.add_argument("--port", type=int, default=8080, help="Local listen port (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Local listen host (default: 127.0.0.1)")
    parser.add_argument("--openai-upstream", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="Upstream OpenAI API base URL")
    parser.add_argument("--anthropic-upstream", type=str, default=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"), help="Upstream Anthropic API base URL")
    parser.add_argument("--gemini-upstream", type=str, default=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"), help="Upstream Google Gemini API base URL")
    parser.add_argument("--top-k", type=int, default=5, help="Number of shortlisted candidate tools to retain (default: 5)")
    parser.add_argument("--min-tools", type=int, default=6, help="Minimum tool count threshold to trigger pruning (default: 6)")
    parser.add_argument("--embedding-model", type=str, default=os.getenv("APICAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"), help="Dense embedding model for tool semantic ranking (default: all-MiniLM-L6-v2)")

    args = parser.parse_args()

    server = create_proxy_server(
        port=args.port,
        host=args.host,
        openai_upstream=args.openai_upstream,
        anthropic_upstream=args.anthropic_upstream,
        gemini_upstream=args.gemini_upstream,
        top_k=args.top_k,
        min_tools_to_prune=args.min_tools,
        embedding_model=args.embedding_model,
    )

    logger.info("==================================================================")
    logger.info("Apical Plexus Reverse Proxy running at http://%s:%d", args.host, args.port)
    logger.info("OpenAI Base URL:     http://%s:%d/v1", args.host, args.port)
    logger.info("Anthropic Base URL:  http://%s:%d", args.host, args.port)
    logger.info("Gemini Base URL:     http://%s:%d", args.host, args.port)
    logger.info("Forwarding OpenAI -> %s", args.openai_upstream)
    logger.info("Forwarding Anthropic -> %s", args.anthropic_upstream)
    logger.info("Forwarding Gemini -> %s", args.gemini_upstream)
    logger.info("Top-K Tools Shortlist: %d (triggered when tools >= %d)", args.top_k, args.min_tools)
    logger.info("Embedding Backbone:    %s", args.embedding_model)
    logger.info("Zero-Retention Ledger: %s", DEFAULT_TELEMETRY_LOG)
    logger.info("==================================================================")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down apical-proxy gracefully.")
        server.server_close()


if __name__ == "__main__":
    main()
