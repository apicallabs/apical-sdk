"""Offline Cryptographic Zero-Retention Receipt Verifier.

Allows developers, compliance officers, and automated auditors to verify
the Ed25519 signature and zero-retention storage invariants offline without
making any network calls.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from apical.exceptions import FixedByteInvariantError, VerificationError


def verify_zero_retention_receipt(
    receipt: Dict[str, Any],
    public_key_hex: Optional[Union[str, Sequence[str]]] = None,
) -> bool:
    """Cryptographically verify an Apical Labs Zero-Retention Receipt.

    Checks:
    1. persisted_bytes_delta == 0 (Pigeonhole proof of zero replay buffer)
    2. input_buffer_zeroed == True (RAM scrubbing proof)
    3. Ed25519 signature over canonical JSON payload

    Args:
        receipt: The receipt dictionary returned by predict/learn/route/reinforce/unlearn.
        public_key_hex: Optional 64-char hex string of the enclave's public key.
                        If omitted, falls back to receipt["enclave_public_key_hex"].

    Returns:
        True if the signature and all zero-retention invariants are verified.

    Raises:
        FixedByteInvariantError: If persisted_bytes_delta != 0.
        VerificationError: If the signature is invalid or receipt is malformed.
    """
    if not isinstance(receipt, dict):
        raise VerificationError(f"Receipt must be a dictionary, got {type(receipt).__name__}")

    # 1. Assert Storage-Level Zero-Retention Invariant
    delta = receipt.get("persisted_bytes_delta")
    if delta is None:
        raise VerificationError("Missing 'persisted_bytes_delta' in receipt")
    if delta != 0:
        raise FixedByteInvariantError(
            f"Zero-Retention Invariant Violated! persisted_bytes_delta is {delta} != 0."
        )

    # 2. Assert RAM Input Buffer Scrubbing Invariant
    if not receipt.get("input_buffer_zeroed", False):
        raise VerificationError("Receipt indicates input buffer was NOT zeroed in RAM")

    # 3. Extract signature
    sig_hex = receipt.get("ed25519_signature")
    if not sig_hex:
        raise VerificationError("Missing 'ed25519_signature' in receipt")

    try:
        sig_bytes = bytes.fromhex(sig_hex)
    except ValueError as e:
        raise VerificationError(f"Malformed signature hex: {e}") from e

    # 4. Resolve public key
    receipt_key = receipt.get("enclave_public_key_hex")
    if isinstance(public_key_hex, (list, tuple, set)):
        if receipt_key and receipt_key not in public_key_hex:
            raise VerificationError(
                f"Receipt public key '{receipt_key}' is not present in the authorized keyring"
            )
        key_hex = receipt_key or (next(iter(public_key_hex)) if public_key_hex else None)
    else:
        key_hex = public_key_hex or receipt_key

    if not key_hex:
        raise VerificationError("No public key provided or present in receipt")

    try:
        pubkey_bytes = bytes.fromhex(key_hex)
        verifier = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
    except Exception as e:
        raise VerificationError(f"Invalid public key: {e}") from e

    # 5. Canonical JSON reconstruction (keys sorted, no extra whitespace)
    signing_dict = {k: v for k, v in receipt.items() if k != "ed25519_signature"}
    canonical_bytes = json.dumps(signing_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")

    # 6. Verify cryptographic signature
    try:
        verifier.verify(sig_bytes, canonical_bytes)
        return True
    except InvalidSignature as e:
        raise VerificationError("Cryptographic signature verification failed") from e
