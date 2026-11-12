"""Unit tests for Apical Plexus SDK surfaces (PlexusNetwork and EdgeMicrocircuit)."""

from __future__ import annotations

import json
import time
import unittest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import httpx
import numpy as np

from apical import ApicalClient, verify_zero_retention_receipt
from apical.plexus import EdgeMicrocircuit, PlexusNetwork


class TestPlexusSDK(unittest.TestCase):
    """Test suite for client.plexus and client.plexus.edge interfaces."""

    def setUp(self) -> None:
        self.privkey = Ed25519PrivateKey.generate()
        self.pubkey_hex = self.privkey.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()

    def _make_signed_receipt(self, op: str) -> dict:
        raw = {
            "receipt_version": "1.0",
            "receipt_id": f"rcpt_{op}_001",
            "tenant_id": "tnt_test",
            "model_id": "model_test",
            "operation": op,
            "timestamp_ns": time.time_ns(),
            "input_sha256_digest": "0" * 64,
            "input_buffer_zeroed": True,
            "scrubbed_at_unix_ns": time.time_ns(),
            "persisted_bytes_delta": 0,
            "state_sha256_before": "a" * 64,
            "state_sha256_after": "b" * 64,
            "enclave_pcr0": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "enclave_public_key_hex": self.pubkey_hex,
        }
        canonical_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sig = self.privkey.sign(canonical_bytes)
        raw["ed25519_signature"] = sig.hex()
        return raw

    def _make_mock_client(self) -> ApicalClient:
        def handler(request: httpx.Request) -> httpx.Response:
            url_path = request.url.path
            body = json.loads(request.content.decode("utf-8")) if request.content else {}

            if url_path == "/.well-known/apical-enclave-pubkey.json":
                return httpx.Response(200, json={
                    "public_key_hex": self.pubkey_hex,
                    "algorithm": "Ed25519",
                })

            elif url_path == "/api/v2/models":
                receipt = self._make_signed_receipt("plexus_create")
                return httpx.Response(201, json={"ok": True, "data": {"status": "created", "receipt": receipt}})

            elif url_path == "/api/v2/predict":
                receipt = self._make_signed_receipt("plexus_predict")
                pred_cls = 3 if body.get("model_id") == "default_edge" else 42
                return httpx.Response(200, json={
                    "ok": True,
                    "data": {
                        "prediction": pred_cls,
                        "confidence": 0.94,
                        "probabilities": [0.0] * pred_cls + [0.94],
                        "reconstructed_vector": body.get("vector", []),
                        "recovery_gain_pct": 8.5,
                        "receipt": receipt,
                        "latency_ms": 0.15,
                    },
                })

            elif url_path == "/api/v2/learn":
                receipt = self._make_signed_receipt("plexus_learn")
                return httpx.Response(200, json={
                    "ok": True,
                    "data": {
                        "status": "adapted",
                        "target_class": body.get("target_class", 42),
                        "winner_class": body.get("target_class", 42),
                        "update_count": 1,
                        "persisted_bytes_delta": 0,
                        "receipt": receipt,
                        "latency_ms": 0.18,
                    },
                })

            elif url_path == "/api/v2/route":
                receipt = self._make_signed_receipt("plexus_route")
                return httpx.Response(200, json={
                    "ok": True,
                    "data": {
                        "routed_class": 7,
                        "top_k_rules": [7, 12, 19],
                        "scores": [0.92, 0.81, 0.70],
                        "receipt": receipt,
                        "latency_ms": 0.16,
                    },
                })

            elif url_path == "/api/v2/unlearn":
                receipt = self._make_signed_receipt("plexus_erase_class")
                return httpx.Response(200, json={
                    "ok": True,
                    "data": {
                        "status": "erased",
                        "erased_class_id": body.get("class_id", 42),
                        "persisted_bytes_delta": 0,
                        "receipt": receipt,
                        "latency_ms": 0.09,
                    },
                })

            return httpx.Response(404, json={"ok": False, "error": "Not found"})

        transport = httpx.MockTransport(handler)
        return ApicalClient(api_key="apk_test_key", transport=transport)

    def test_plexus_network_lifecycle(self) -> None:
        client = self._make_mock_client()

        # 1. Instantiate via client.plexus
        plexus = client.plexus(
            model_id="qwen-intent-router",
            in_features=896,
            num_classes=150,
            tier="enterprise",
        )
        self.assertIsInstance(plexus, PlexusNetwork)
        self.assertEqual(plexus.tier, 1)

        # 2. Predict
        vec = np.random.randn(896).astype(np.float32)
        pred_res = plexus.predict(vec)
        self.assertEqual(pred_res.prediction, 42)
        self.assertAlmostEqual(pred_res.confidence, 0.94)

        # 3. Learn & verify receipt
        learn_res = plexus.learn(vec, target_class=42)
        self.assertEqual(learn_res.status, "adapted")
        self.assertEqual(learn_res.target_class, 42)
        self.assertEqual(learn_res.persisted_bytes_delta, 0)
        self.assertTrue(verify_zero_retention_receipt(learn_res.receipt, client.enclave_pubkey))

        # 4. Route
        route_res = plexus.route(vec, top_k=3)
        self.assertEqual(route_res.routed_class, 7)
        self.assertEqual(len(route_res.top_k_rules), 3)

        # 5. Unlearn
        unlearn_res = plexus.unlearn(class_id=42)
        self.assertEqual(unlearn_res.status, "erased")
        self.assertEqual(unlearn_res.persisted_bytes_delta, 0)

    def test_edge_microcircuit_lifecycle(self) -> None:
        client = self._make_mock_client()

        # 1. Instantiate via client.plexus.edge
        edge = client.plexus.edge(
            input_shape=(28, 28),
            num_classes=10,
            sram_budget_kb=101,
        )
        self.assertIsInstance(edge, EdgeMicrocircuit)

        # 2. Infer
        patch = np.ones((28, 28), dtype=np.float32)
        infer_res = edge.infer(patch)
        self.assertIn(infer_res.prediction, range(10))

        # 3. Learn class 3
        learn_res = edge.learn(patch, target_class=3)
        self.assertEqual(learn_res.target_class, 3)
        self.assertEqual(learn_res.persisted_bytes_delta, 0)

        # 4. Infer after learning
        infer_res2 = edge.infer(patch)
        self.assertEqual(infer_res2.prediction, 3)

        # 5. Associative recall under 50% occlusion
        occluded = patch.copy()
        occluded[:14, :] = 0.0
        recall_res = edge.associative_recall(occluded)
        self.assertEqual(len(recall_res.reconstructed_vector), 784)
        self.assertGreater(recall_res.recovery_gain_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
