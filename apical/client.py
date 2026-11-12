"""Base HTTP Client for the Apical Labs Platform."""

from __future__ import annotations

import time
from typing import Any, Dict, NamedTuple, Optional, Sequence, Union
import httpx

from apical.exceptions import (
    APIError,
    ApicalError,
    AuthenticationError,
    ModelNotFoundError,
    ProductNotEntitledError,
    RateLimitError,
    SparseProbeError,
    SweepDetectedError,
)


DEFAULT_BASE_URL = "https://apical-labs.com"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3
SDK_VERSION = "0.5.5"


class BridgeResult(NamedTuple):
    """Result of an arbor.Bridge cross-backbone migration."""
    status: str
    model_id: str
    alignment_score: float
    persisted_bytes_delta: int
    update_count: int
    receipt: Dict[str, Any]
    latency_ms: float = 0.0

    @property
    def signature(self) -> str:
        """Return the Ed25519 signature from the cryptographic attestation receipt."""
        return str(self.receipt.get("ed25519_signature", ""))

    @property
    def ed25519_signature(self) -> str:
        """Alias for signature matching receipt schema."""
        return self.signature


class ArborNamespace:
    """Namespace for Apical Arbor continual learning and routing components."""

    def __init__(self, client: "ApicalClient") -> None:
        self._client = client

    def create_model(
        self,
        model_id: str,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        input_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        privacy_secret: Optional[str] = None,
    ) -> Any:
        """Create and initialize a fixed-byte model container on the Enclave, returning a Classifier."""
        in_dim = in_features or input_dim
        out_dim = out_features or num_classes
        if in_dim is None or out_dim is None:
            raise ValueError("Both input dimensions and output classes must be specified to create a model container.")

        payload = {
            "model_id": model_id,
            "in_features": in_dim,
            "out_features": out_dim,
        }
        self._client.request("POST", "/api/v1/models", payload)

        return self.classifier(
            model_id=model_id,
            in_features=in_dim,
            num_classes=out_dim,
            privacy_secret=privacy_secret,
        )

    def classifier(
        self,
        model_id: str,
        in_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        privacy_secret: Optional[str] = None,
    ) -> "Classifier":
        """Return a Classifier wrapper for classification and continuous learning."""
        from apical.classifier import Classifier

        secret = privacy_secret or self._client.privacy_secret
        return Classifier(
            client=self._client,
            model_id=model_id,
            in_features=in_features,
            num_classes=num_classes,
            privacy_secret=secret,
        )

    def memory_router(
        self,
        router_id: str,
        embedding_dim: Optional[int] = None,
        num_rules: Optional[int] = None,
        in_features: Optional[int] = None,
        num_tools: Optional[int] = None,
        privacy_secret: Optional[str] = None,
    ) -> "MemoryRouter":
        """Return a MemoryRouter wrapper for embedding routing and reinforcement."""
        from apical.router import MemoryRouter

        secret = privacy_secret or self._client.privacy_secret
        dim = embedding_dim if embedding_dim is not None else in_features
        rules = num_rules if num_rules is not None else num_tools
        return MemoryRouter(
            client=self._client,
            router_id=router_id,
            embedding_dim=dim,
            num_rules=rules,
            privacy_secret=secret,
        )

    def bridge(
        self,
        model_id: str,
        source_anchors: Optional[Any] = None,
        target_anchors: Optional[Any] = None,
        public_anchors: Optional[Any] = None,
        tier: int = 0,
        verify_receipt: bool = True,
    ) -> BridgeResult:
        """Execute zero-downtime cross-backbone model migration using public anchor pairs."""
        from apical.verifier import verify_zero_retention_receipt

        # Unpack public_anchors if provided as dict or tuple
        if public_anchors is not None:
            if isinstance(public_anchors, dict):
                source_anchors = public_anchors.get("source") or public_anchors.get("source_anchors")
                target_anchors = public_anchors.get("target") or public_anchors.get("target_anchors")
            elif isinstance(public_anchors, (list, tuple)) and len(public_anchors) == 2:
                source_anchors, target_anchors = public_anchors[0], public_anchors[1]

        if source_anchors is None or target_anchors is None:
            raise ValueError("Both source_anchors and target_anchors must be provided (or public_anchors)")

        def _to_list_matrix(m: Any) -> list[list[float]]:
            if hasattr(m, "tolist"):
                return m.tolist()
            if isinstance(m, (list, tuple)):
                return [list(row) if hasattr(row, "__iter__") else [float(row)] for row in m]
            return list(m)

        src_list = _to_list_matrix(source_anchors)
        tgt_list = _to_list_matrix(target_anchors)

        payload = {
            "model_id": model_id,
            "source_anchors": src_list,
            "target_anchors": tgt_list,
            "tier": tier,
        }

        resp = self._client.request("POST", "/api/v1/bridge", payload)
        data = resp.get("data", resp)

        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        return BridgeResult(
            status=str(data.get("status", "bridged")),
            model_id=str(data.get("model_id", model_id)),
            alignment_score=float(data.get("alignment_score", 1.0)),
            persisted_bytes_delta=int(data.get("persisted_bytes_delta", 0)),
            update_count=int(data.get("update_count", 0)),
            receipt=receipt,
            latency_ms=float(data.get("latency_ms", 0.0)),
        )


class ApicalClient:
    """Primary client for interacting with the Apical Labs API Gateway.

    Manages HTTP/2 connection pooling, API key authentication, and retry policies.
    """

    def __init__(
        self,
        api_key: str,
        privacy_secret: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key cannot be empty")

        self.api_key = api_key
        self.privacy_secret = privacy_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Client-Version": SDK_VERSION,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            transport=transport,
        )

    @property
    def arbor(self) -> ArborNamespace:
        """Access the Apical Arbor product namespace (classifier, memory_router, bridge)."""
        return ArborNamespace(self)

    @property
    def plexus(self) -> Any:
        """Access the Apical Plexus product namespace (multi-layer adapter and edge microcircuit)."""
        from apical.plexus import PlexusNamespace
        return PlexusNamespace(self)

    @property
    def enclave_pubkey(self) -> str:
        """Retrieve the enclave Ed25519 public key in hex format."""
        if not hasattr(self, "_cached_enclave_pubkey") or self._cached_enclave_pubkey is None:
            try:
                resp = self._client.get("/.well-known/apical-enclave-pubkey.json")
                if resp.status_code == 200:
                    self._cached_enclave_pubkey = resp.json().get("public_key_hex", "")
                else:
                    self._cached_enclave_pubkey = ""
            except Exception:
                self._cached_enclave_pubkey = ""
        return self._cached_enclave_pubkey

    def classifier(
        self,
        model_id: str,
        in_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        privacy_secret: Optional[str] = None,
    ) -> Any:
        """Return a Classifier wrapper for classification and continuous learning."""
        return self.arbor.classifier(
            model_id=model_id,
            in_features=in_features,
            num_classes=num_classes,
            privacy_secret=privacy_secret,
        )

    def memory_router(
        self,
        router_id: str,
        embedding_dim: Optional[int] = None,
        num_rules: Optional[int] = None,
        in_features: Optional[int] = None,
        num_tools: Optional[int] = None,
        privacy_secret: Optional[str] = None,
    ) -> Any:
        """Return a MemoryRouter wrapper for embedding routing and reinforcement."""
        return self.arbor.memory_router(
            router_id=router_id,
            embedding_dim=embedding_dim,
            num_rules=num_rules,
            in_features=in_features,
            num_tools=num_tools,
            privacy_secret=privacy_secret,
        )

    def bridge(
        self,
        model_id: str,
        source_anchors: Optional[Any] = None,
        target_anchors: Optional[Any] = None,
        public_anchors: Optional[Any] = None,
        tier: int = 0,
        verify_receipt: bool = True,
    ) -> BridgeResult:
        """Execute zero-downtime cross-backbone model migration using public anchor pairs."""
        return self.arbor.bridge(
            model_id=model_id,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            public_anchors=public_anchors,
            tier=tier,
            verify_receipt=verify_receipt,
        )

    def request(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute an HTTP request with automatic retry on transient network failures."""
        clean_path = "/" + path.lstrip("/")
        retries = 0

        while True:
            try:
                response = self._client.request(
                    method=method,
                    url=clean_path,
                    json=json_body,
                )

                if response.status_code == 200 or response.status_code == 201:
                    return response.json()

                self._handle_error_response(response)

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                retries += 1
                if retries >= self.max_retries:
                    raise APIError(
                        f"Network connection failed after {retries} retries: {e}",
                        status_code=None,
                        error_code="ERR_NETWORK_FAILURE",
                    ) from e
                time.sleep(0.1 * (2 ** retries))

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Parse structured API gateway errors and raise appropriate typed exceptions."""
        try:
            body = response.json()
        except Exception:
            body = {"error": response.text}

        code = body.get("code", "")
        message = body.get("error", response.text)
        status = response.status_code

        if status == 401:
            raise AuthenticationError(message, status_code=status, error_code=code, details=body)
        elif code == "ERR_PRODUCT_NOT_ENTITLED":
            raise ProductNotEntitledError(message, status_code=status, error_code=code, details=body)
        elif status == 404:
            raise ModelNotFoundError(message, status_code=status, error_code=code, details=body)
        elif status == 429:
            raise RateLimitError(message, status_code=status, error_code=code, details=body)
        elif code == "ERR_ORACLE_SPARSE_PROBE":
            raise SparseProbeError(message, status_code=status, error_code=code, details=body)
        elif code == "ERR_ORACLE_SWEEP_DETECTED":
            raise SweepDetectedError(message, status_code=status, error_code=code, details=body)
        else:
            raise APIError(message, status_code=status, error_code=code, details=body)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "ApicalClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
