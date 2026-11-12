"""Integration and unit tests for Apical Plexus Reverse Proxy (apical-proxy)."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import unittest
import httpx

from apical.enclave_client import EnclaveDirectClient
from apical.proxy import (
    create_proxy_server,
    extract_user_query,
    calculate_tools_tokens,
)
from apical.telemetry import ZeroRetentionTelemetryLogger
from apical.verifier import verify_zero_retention_receipt


class TestApicalProxy(unittest.TestCase):
    """Test suite for apical-proxy in-line schema pruner."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_log_path = Path("/tmp/test_proxy_telemetry.jsonl")
        if cls.test_log_path.exists():
            cls.test_log_path.unlink()

        cls.test_telemetry = ZeroRetentionTelemetryLogger(log_path=cls.test_log_path)
        cls.enclave_client = EnclaveDirectClient()

        # Start proxy on an ephemeral port
        cls.port = 18080
        cls.server = create_proxy_server(
            port=cls.port,
            host="127.0.0.1",
            openai_upstream="https://httpbin.org/anything",
            anthropic_upstream="https://httpbin.org/anything",
            gemini_upstream="https://httpbin.org/anything",
            top_k=2,
            min_tools_to_prune=4,
            enclave_client=cls.enclave_client,
            telemetry_logger=cls.test_telemetry,
        )
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        if cls.test_log_path.exists():
            cls.test_log_path.unlink()

    def test_extract_user_query(self) -> None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I check git status?"},
            {"role": "assistant", "content": "Use git status."},
            {"role": "user", "content": [{"type": "text", "text": "What about uncommitted diffs?"}]},
        ]
        q = extract_user_query(messages)
        self.assertEqual(q, "What about uncommitted diffs?")

    def test_proxy_health_endpoint(self) -> None:
        resp = httpx.get(f"http://127.0.0.1:{self.port}/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["enclave"], "connected")
        self.assertTrue(len(data["enclave_pubkey"]) > 0)

    def test_cors_options(self) -> None:
        resp = httpx.options(f"http://127.0.0.1:{self.port}/v1/chat/completions")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")

    def test_tool_pruning_and_telemetry(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "git_commit", "description": "Record changes to the git repository"}},
            {"type": "function", "function": {"name": "view_file", "description": "View file contents from the workspace filesystem"}},
            {"type": "function", "function": {"name": "stripe_refund", "description": "Refund duplicate credit card charges on Stripe"}},
            {"type": "function", "function": {"name": "k8s_restart", "description": "Restart Kubernetes pod deployment in cluster"}},
            {"type": "function", "function": {"name": "zendesk_ticket", "description": "Escalate customer service ticket in Zendesk"}},
        ]

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Can you open and view the file package.json?"}
            ],
            "tools": tools,
        }

        # Send request to proxy
        resp = httpx.post(f"http://127.0.0.1:{self.port}/v1/chat/completions", json=payload, timeout=10.0)
        self.assertEqual(resp.status_code, 200)

        # httpbin echoes back what was forwarded in its 'json' field
        echoed = resp.json().get("json", {})
        forwarded_tools = echoed.get("tools", [])

        # Proxy top_k was set to 2
        self.assertEqual(len(forwarded_tools), 2)
        top_tool_name = forwarded_tools[0]["function"]["name"]
        self.assertEqual(top_tool_name, "view_file")

        # Verify telemetry record
        records = self.test_telemetry.read_records()
        self.assertTrue(len(records) >= 1)
        last_record = records[-1]
        self.assertEqual(last_record.operation, "proxy_prune")
        self.assertEqual(last_record.persisted_bytes_delta, 0)
        self.assertTrue(last_record.tokens_saved > 0)
        self.assertTrue(last_record.enclave_verified)

    def test_gemini_format_pruning(self) -> None:
        function_decls = [
            {"name": "git_commit", "description": "Record changes to the git repository"},
            {"name": "view_file", "description": "View file contents from the workspace filesystem"},
            {"name": "stripe_refund", "description": "Refund duplicate credit card charges on Stripe"},
            {"name": "k8s_restart", "description": "Restart Kubernetes pod deployment in cluster"},
            {"name": "zendesk_ticket", "description": "Escalate customer service ticket in Zendesk"},
        ]

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Inspect the source code in main.py"}],
                }
            ],
            "tools": [{"function_declarations": function_decls}],
        }

        resp = httpx.post(
            f"http://127.0.0.1:{self.port}/v1beta/models/gemini-1.5-pro:generateContent",
            json=payload,
            timeout=10.0,
        )
        self.assertEqual(resp.status_code, 200)

        echoed = resp.json().get("json", {})
        forwarded_tools = echoed.get("tools", [])
        self.assertEqual(len(forwarded_tools), 1)
        pruned_decls = forwarded_tools[0].get("function_declarations", [])
        self.assertEqual(len(pruned_decls), 2)
        self.assertEqual(pruned_decls[0]["name"], "view_file")


if __name__ == "__main__":
    unittest.main()
