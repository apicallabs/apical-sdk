"""Test suite for ApicalClient, Classifier, and MemoryRouter surfaces."""

from __future__ import annotations

import json
import time
import unittest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import httpx

from apical import ApicalClient
from apical.exceptions import AuthenticationError, RateLimitError, SparseProbeError


class TestClientAndProducts(unittest.TestCase):
    """Verify HTTP client behavior and product surface wrappers."""

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

    def test_classifier_full_lifecycle(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url_path = request.url.path
            body = json.loads(request.content.decode("utf-8")) if request.content else {}

            if url_path == "/api/v1/models":
                receipt = self._make_signed_receipt("create")
                return httpx.Response(201, json={"ok": True, "data": {"status": "created", "receipt": receipt}})

            elif url_path == "/api/v1/predict":
                receipt = self._make_signed_receipt("predict")
                return httpx.Response(200, json={
                    "ok": True,
                    "data": {
                        "prediction": 2,
                        "confidence": 0.88,
                        "crystallization_pct": None,
                        "stability_regime": None,
                        "receipt": receipt,
                    },
                })

            elif url_path == "/api/v1/learn":
                receipt = self._make_signed_receipt("learn")
                return httpx.Response(200, json={
                    "ok": True,
                    "data": {
                        "status": "updated",
                        "target_class": body.get("target_class", 1),
                        "winner_class": 1,
                        "confidence": 0.95,
                        "update_count": 5,
                        "persisted_bytes_delta": 0,
                        "receipt": receipt,
                    },
                })

            elif url_path == "/api/v1/unlearn":
                receipt = self._make_signed_receipt("erase_class")
                return httpx.Response(200, json={
                    "ok": True,
                    "data": {
                        "status": "erased",
                        "erased_class_id": body.get("class_id", 0),
                        "persisted_bytes_delta": 0,
                        "receipt": receipt,
                    },
                })

            return httpx.Response(404, json={"ok": False, "error": "Not found"})

        transport = httpx.MockTransport(handler)
        client = ApicalClient(
            api_key="apk_test_123",
            privacy_secret="test_secret",
            base_url="https://mock.apical-labs.com",
            transport=transport,
        )

        classifier = client.arbor.classifier("intent_model", in_features=32, num_classes=5)

        # 1. Create
        create_res = classifier.create(in_features=32, num_classes=5)
        self.assertEqual(create_res["status"], "created")

        # 2. Predict
        pred_res = classifier.predict([0.1] * 32)
        self.assertEqual(pred_res.prediction, 2)
        self.assertAlmostEqual(pred_res.confidence, 0.88)

        # 3. Learn
        learn_res = classifier.learn([0.1] * 32, target_class=1)
        self.assertEqual(learn_res.status, "updated")
        self.assertEqual(learn_res.persisted_bytes_delta, 0)

        # 4. Unlearn
        unlearn_res = classifier.unlearn(class_id=1)
        self.assertEqual(unlearn_res.status, "erased")
        self.assertEqual(unlearn_res.erased_class_id, 1)

    def test_memory_router_lifecycle(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url_path = request.url.path
            body = json.loads(request.content.decode("utf-8")) if request.content else {}

            if url_path == "/api/v1/route":
                receipt = self._make_signed_receipt("route")
                return httpx.Response(200, json={
                    "ok": True,
                    "data": {
                        "routes": [{"rule_id": 4, "affinity": 0.89}, {"rule_id": 1, "affinity": 0.35}],
                        "top_rule_id": 4,
                        "receipt": receipt,
                    },
                })

            elif url_path == "/api/v1/reinforce":
                receipt = self._make_signed_receipt("reinforce")
                return httpx.Response(200, json={
                    "ok": True,
                    "data": {
                        "status": "updated",
                        "target_class": body.get("target_rule", 4),
                        "confidence": 0.91,
                        "update_count": 3,
                        "persisted_bytes_delta": 0,
                        "receipt": receipt,
                    },
                })

            return httpx.Response(404, json={"ok": False, "error": "Not found"})

        transport = httpx.MockTransport(handler)
        client = ApicalClient(api_key="apk_test_123", transport=transport)
        router = client.arbor.memory_router("agent_router", embedding_dim=16, num_rules=8)

        # Route
        route_res = router.route([0.2] * 16, top_k=2)
        self.assertEqual(len(route_res.routes), 2)
        self.assertEqual(route_res.top_rule_id, 4)
        self.assertEqual(route_res.routes[0].rule_id, 4)

        # Reinforce
        reinforce_res = router.reinforce([0.2] * 16, target_rule=4)
        self.assertEqual(reinforce_res.target_rule, 4)
        self.assertEqual(reinforce_res.persisted_bytes_delta, 0)

    def test_error_handling(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "auth_fail" in request.headers.get("authorization", ""):
                return httpx.Response(401, json={"ok": False, "error": "Invalid API key", "code": "ERR_AUTH_FAILED"})
            elif "rate_limit" in request.headers.get("authorization", ""):
                return httpx.Response(429, json={"ok": False, "error": "Rate limit exceeded", "code": "ERR_RATE_LIMITED"})
            elif "sparse" in request.headers.get("authorization", ""):
                return httpx.Response(400, json={"ok": False, "error": "Sparse probe rejected", "code": "ERR_ORACLE_SPARSE_PROBE"})
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)

        # 401
        c_auth = ApicalClient(api_key="auth_fail", transport=transport)
        with self.assertRaises(AuthenticationError):
            c_auth.request("GET", "/test")

        # 429
        c_rate = ApicalClient(api_key="rate_limit", transport=transport)
        with self.assertRaises(RateLimitError):
            c_rate.request("GET", "/test")

        # Sparse probe
        c_sparse = ApicalClient(api_key="sparse", transport=transport)
        with self.assertRaises(SparseProbeError):
            c_sparse.request("GET", "/test")

    def test_arbor_namespace_and_bridge(self) -> None:
        receipt = self._make_signed_receipt("bridge")

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/api/v1/bridge":
                return httpx.Response(200, json={
                    "ok": True,
                    "data": {
                        "status": "bridged",
                        "model_id": "vision",
                        "alignment_score": 0.965,
                        "persisted_bytes_delta": 0,
                        "update_count": 5,
                        "latency_ms": 12.4,
                        "receipt": receipt,
                    },
                })
            elif path == "/api/v1/unlearn":
                unlearn_receipt = self._make_signed_receipt("erase_class")
                return httpx.Response(200, json={
                    "ok": True,
                    "data": {
                        "status": "erased",
                        "erased_class_id": 42,
                        "persisted_bytes_delta": 0,
                        "receipt": unlearn_receipt,
                    },
                })
            return httpx.Response(404, json={"ok": False, "error": "Not found"})

        transport = httpx.MockTransport(handler)
        client = ApicalClient(api_key="apk_test_123", transport=transport)

        # 1. Test client.arbor namespace
        clf = client.arbor.classifier("vision", in_features=16, num_classes=10)
        cert = clf.erase_class(target_class=42, verify_receipt=False)
        self.assertEqual(cert.status, "erased")
        self.assertEqual(cert.persisted_bytes_delta, 0)
        self.assertEqual(cert.erased_class_id, 42)

        router = client.arbor.memory_router("router_1", embedding_dim=16, num_rules=4)
        self.assertEqual(router.router_id, "router_1")

        # 2. Test client.arbor.bridge with public_anchors dict
        z_pub = {
            "source": [[0.1] * 16, [0.2] * 16],
            "target": [[0.15] * 16, [0.25] * 16],
        }
        res_bridge = client.arbor.bridge("vision", public_anchors=z_pub, verify_receipt=False)
        self.assertEqual(res_bridge.status, "bridged")
        self.assertEqual(res_bridge.persisted_bytes_delta, 0)
        self.assertGreater(res_bridge.alignment_score, 0.9)
        self.assertEqual(res_bridge.update_count, 5)

        # 3. Test client.arbor.bridge with explicit source_anchors & target_anchors
        res_direct = client.arbor.bridge(
            "vision",
            source_anchors=[[0.1] * 16, [0.2] * 16],
            target_anchors=[[0.15] * 16, [0.25] * 16],
            verify_receipt=False,
        )
        self.assertEqual(res_direct.status, "bridged")
        self.assertEqual(res_direct.persisted_bytes_delta, 0)


if __name__ == "__main__":
    unittest.main()
