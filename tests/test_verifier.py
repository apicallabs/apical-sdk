"""Test suite for Offline Zero-Retention Receipt Verifier."""

from __future__ import annotations

import json
import time
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from apical.exceptions import FixedByteInvariantError, VerificationError
from apical.verifier import verify_zero_retention_receipt


class TestReceiptVerifier(unittest.TestCase):
    """Verify cryptographic and zero-retention invariant checks."""

    def setUp(self) -> None:
        self.privkey = Ed25519PrivateKey.generate()
        self.pubkey = self.privkey.public_key()
        self.pubkey_hex = self.pubkey.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()

    def _sign_receipt(self, receipt_dict: dict) -> dict:
        signing_dict = {k: v for k, v in receipt_dict.items() if k != "ed25519_signature"}
        canonical_bytes = json.dumps(signing_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sig = self.privkey.sign(canonical_bytes)
        signed = dict(receipt_dict)
        signed["ed25519_signature"] = sig.hex()
        return signed

    def test_verify_valid_receipt(self) -> None:
        raw_receipt = {
            "receipt_version": "1.0",
            "receipt_id": "rcpt_valid_001",
            "tenant_id": "tnt_acme",
            "model_id": "model_1",
            "operation": "learn",
            "timestamp_ns": time.time_ns(),
            "input_sha256_digest": "a" * 64,
            "input_buffer_zeroed": True,
            "scrubbed_at_unix_ns": time.time_ns(),
            "persisted_bytes_delta": 0,
            "state_sha256_before": "1" * 64,
            "state_sha256_after": "2" * 64,
            "enclave_pcr0": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "enclave_public_key_hex": self.pubkey_hex,
        }
        signed = self._sign_receipt(raw_receipt)
        self.assertTrue(verify_zero_retention_receipt(signed))
        self.assertTrue(verify_zero_retention_receipt(signed, public_key_hex=self.pubkey_hex))

    def test_verify_rejects_nonzero_persisted_bytes_delta(self) -> None:
        raw_receipt = {
            "receipt_id": "rcpt_invalid_delta",
            "input_buffer_zeroed": True,
            "persisted_bytes_delta": 64,  # Violation!
            "enclave_public_key_hex": self.pubkey_hex,
        }
        signed = self._sign_receipt(raw_receipt)
        with self.assertRaises(FixedByteInvariantError):
            verify_zero_retention_receipt(signed)

    def test_verify_rejects_unzeroed_ram_buffer(self) -> None:
        raw_receipt = {
            "receipt_id": "rcpt_unzeroed",
            "input_buffer_zeroed": False,  # Violation!
            "persisted_bytes_delta": 0,
            "enclave_public_key_hex": self.pubkey_hex,
        }
        signed = self._sign_receipt(raw_receipt)
        with self.assertRaises(VerificationError):
            verify_zero_retention_receipt(signed)

    def test_verify_rejects_tampered_payload(self) -> None:
        raw_receipt = {
            "receipt_id": "rcpt_tampered",
            "operation": "predict",
            "input_buffer_zeroed": True,
            "persisted_bytes_delta": 0,
            "enclave_public_key_hex": self.pubkey_hex,
        }
        signed = self._sign_receipt(raw_receipt)

        # Alter field after signing
        signed["operation"] = "learn"
        with self.assertRaises(VerificationError):
            verify_zero_retention_receipt(signed)

    def test_verify_with_keyring_list(self) -> None:
        raw_receipt = {
            "receipt_id": "rcpt_keyring_001",
            "operation": "route",
            "input_buffer_zeroed": True,
            "persisted_bytes_delta": 0,
            "enclave_public_key_hex": self.pubkey_hex,
        }
        signed = self._sign_receipt(raw_receipt)
        other_key = "00" * 32
        keyring = [other_key, self.pubkey_hex]
        self.assertTrue(verify_zero_retention_receipt(signed, public_key_hex=keyring))

    def test_verify_rejects_key_not_in_keyring(self) -> None:
        raw_receipt = {
            "receipt_id": "rcpt_keyring_untrusted",
            "operation": "route",
            "input_buffer_zeroed": True,
            "persisted_bytes_delta": 0,
            "enclave_public_key_hex": self.pubkey_hex,
        }
        signed = self._sign_receipt(raw_receipt)
        untrusted_keyring = ["11" * 32, "22" * 32]
        with self.assertRaises(VerificationError) as ctx:
            verify_zero_retention_receipt(signed, public_key_hex=untrusted_keyring)
        self.assertIn("not present in the authorized keyring", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
