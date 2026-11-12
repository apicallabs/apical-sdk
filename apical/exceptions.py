"""Exceptions for the Apical Labs Client SDK."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ApicalError(Exception):
    """Base exception for all Apical Labs SDK errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        parts = [self.message]
        if self.error_code:
            parts.append(f"[code: {self.error_code}]")
        if self.status_code:
            parts.append(f"[status: {self.status_code}]")
        return " ".join(parts)


class AuthenticationError(ApicalError):
    """Raised when the API key is missing, invalid, or revoked (HTTP 401)."""


class ProductNotEntitledError(ApicalError):
    """Raised when an API key attempts to access an un-entitled product (HTTP 403 ERR_PRODUCT_NOT_ENTITLED)."""


class RateLimitError(ApicalError):
    """Raised when the tenant's request rate or quota is exceeded (HTTP 429)."""


class ModelNotFoundError(ApicalError):
    """Raised when the specified model or router ID does not exist (HTTP 404)."""


class SparseProbeError(ApicalError):
    """Raised when an input vector is rejected by the anti-oracle input density guard."""


class SweepDetectedError(ApicalError):
    """Raised when sequential inputs trigger the anti-oracle sequential sweep guard."""


class VerificationError(ApicalError):
    """Raised when an Ed25519 Zero-Retention receipt fails cryptographic or invariant verification."""


class FixedByteInvariantError(VerificationError):
    """Raised when a receipt or server indicates persisted_bytes_delta != 0."""


class APIError(ApicalError):
    """Raised when the gateway or enclave returns an unhandled error response (HTTP 5xx)."""
