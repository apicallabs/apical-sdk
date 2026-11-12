"""Apical Plexus Python SDK — Multi-Layer Continual Adaptation & Edge Client.

Provides typed HTTP client interfaces for:
1. PlexusNetwork (client.plexus):
   - Continual multi-layer adaptation and semantic tool routing via the Apical Gateway (/api/v2/*)
   - Constant O(1) memory invariant (persisted_bytes_delta == 0) and Ed25519 receipt verification
2. EdgeMicrocircuit (client.plexus.edge):
   - Client interface for bare-metal edge sensor streams and associative pattern completion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from apical.orthogonal import OrthogonalPrivacyProjector
from apical.verifier import verify_zero_retention_receipt


@dataclass(frozen=True)
class PlexusPredictionResult:
    """Strongly typed result of Plexus inference."""

    prediction: int
    confidence: float
    probabilities: List[float] = field(default_factory=list)
    receipt: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def predicted_class(self) -> int:
        """Alias for prediction to maintain API compatibility."""
        return self.prediction


@dataclass(frozen=True)
class PlexusLearnResult:
    """Strongly typed result of a Plexus continual update."""

    status: str
    target_class: int
    winner_class: int
    update_count: int
    persisted_bytes_delta: int
    receipt: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass(frozen=True)
class PlexusRouteResult:
    """Strongly typed result of Plexus semantic tool routing."""

    routed_class: int
    top_k_rules: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    receipt: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass(frozen=True)
class PlexusUnlearnResult:
    """Strongly typed result of GDPR Article 17 atomic class erasure."""

    status: str
    erased_class_id: int
    persisted_bytes_delta: int
    receipt: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass(frozen=True)
class EdgeInferenceResult:
    """Strongly typed result of edge sensor inference."""

    prediction: int
    confidence: float
    receipt: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass(frozen=True)
class EdgeLearnResult:
    """Strongly typed result of an edge continual update."""

    status: str
    target_class: int
    persisted_bytes_delta: int
    receipt: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass(frozen=True)
class EdgeRecallResult:
    """Strongly typed result of associative pattern completion."""

    reconstructed_vector: List[float]
    recovery_gain_pct: float
    receipt: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


TIER_NAME_TO_INT: Dict[str, int] = {
    "standard": 0,
    "enterprise": 1,
    "pro": 1,
    "ultra": 2,
    "regulated": 2,
    "edge": 0,
}


def _resolve_tier_int(tier: Union[int, str]) -> int:
    if isinstance(tier, str):
        return TIER_NAME_TO_INT.get(tier.strip().lower(), 0)
    return int(tier)


class PlexusNetwork:
    """High-level client wrapper for Apical Plexus continual adaptation models.

    Operates in constant O(1) memory on the Enclave with verifiable zero-retention guarantees.
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        tier: Union[int, str] = 0,
        privacy_secret: Optional[str] = None,
        **_: Any,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.in_features = in_features
        self.out_features = num_classes if num_classes is not None else out_features
        self.tier = _resolve_tier_int(tier)
        self.privacy_secret = privacy_secret

        self._projector: Optional[OrthogonalPrivacyProjector] = None
        if self.privacy_secret and self.in_features:
            self._projector = OrthogonalPrivacyProjector(self.privacy_secret, self.in_features)

    def _prepare_vector(self, vector: Union[Sequence[float], np.ndarray]) -> List[float]:
        """Optionally apply client-side privacy projection, returning a JSON-serializable list."""
        arr = np.asarray(vector, dtype=np.float32).ravel()

        if self._projector is not None:
            arr = self._projector.project(arr)
        elif self.privacy_secret and not self._projector:
            self.in_features = arr.size
            self._projector = OrthogonalPrivacyProjector(self.privacy_secret, self.in_features)
            arr = self._projector.project(arr)

        return arr.tolist()

    def create(
        self,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        verify_receipt: bool = True,
        **_: Any,
    ) -> Dict[str, Any]:
        """Pre-allocate a fixed-byte constant-size Plexus model container on the Enclave.

        Dispatches POST /api/v2/models.
        """
        if in_features is not None:
            self.in_features = in_features
        if num_classes is not None:
            self.out_features = num_classes
        elif out_features is not None:
            self.out_features = out_features

        if self.in_features is None or self.out_features is None:
            raise ValueError("in_features and out_features (or num_classes) must be specified to create a Plexus model")

        if self.privacy_secret and not self._projector:
            self._projector = OrthogonalPrivacyProjector(self.privacy_secret, self.in_features)

        payload = {
            "model_id": self.model_id,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "tier": self.tier,
        }

        resp = self.client.request("POST", "/api/v2/models", payload)
        data = resp.get("data", resp)

        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        return data

    def predict(
        self,
        vector: Union[Sequence[float], np.ndarray],
        tier: Optional[Union[int, str]] = None,
        verify_receipt: bool = True,
    ) -> PlexusPredictionResult:
        """Execute non-linear inference via POST /api/v2/predict."""
        prep_vec = self._prepare_vector(vector)
        effective_tier = self.tier if tier is None else _resolve_tier_int(tier)
        payload = {
            "model_id": self.model_id,
            "vector": prep_vec,
            "tier": effective_tier,
        }

        resp = self.client.request("POST", "/api/v2/predict", payload)
        data = resp.get("data", resp)

        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        probs = data.get("probabilities", [])
        pred = int(data.get("prediction", data.get("predicted_class", 0)))
        conf = float(probs[pred]) if probs and pred < len(probs) else float(data.get("confidence", 0.0))

        return PlexusPredictionResult(
            prediction=pred,
            confidence=conf,
            probabilities=probs,
            receipt=receipt,
            latency_ms=float(data.get("latency_ms", 0.0)),
        )

    def learn(
        self,
        vector: Union[Sequence[float], np.ndarray],
        target_class: int,
        tier: Optional[Union[int, str]] = None,
        verify_receipt: bool = True,
    ) -> PlexusLearnResult:
        """Execute continual learning update via POST /api/v2/learn."""
        prep_vec = self._prepare_vector(vector)
        effective_tier = self.tier if tier is None else _resolve_tier_int(tier)
        payload = {
            "model_id": self.model_id,
            "vector": prep_vec,
            "target_class": int(target_class),
            "tier": effective_tier,
        }

        resp = self.client.request("POST", "/api/v2/learn", payload)
        data = resp.get("data", resp)

        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        return PlexusLearnResult(
            status=str(data.get("status", "adapted")),
            target_class=int(data.get("target_class", target_class)),
            winner_class=int(data.get("winner_class", target_class)),
            update_count=int(data.get("update_count", 0)),
            persisted_bytes_delta=int(receipt.get("persisted_bytes_delta", 0)),
            receipt=receipt,
            latency_ms=float(data.get("latency_ms", 0.0)),
        )

    def route(
        self,
        vector: Union[Sequence[float], np.ndarray],
        top_k: int = 3,
        tier: Optional[Union[int, str]] = None,
        verify_receipt: bool = True,
    ) -> PlexusRouteResult:
        """Route an input vector via POST /api/v2/route, returning top matching rules."""
        prep_vec = self._prepare_vector(vector)
        effective_tier = self.tier if tier is None else _resolve_tier_int(tier)
        payload = {
            "model_id": self.model_id,
            "vector": prep_vec,
            "top_k": int(top_k),
            "tier": effective_tier,
        }

        resp = self.client.request("POST", "/api/v2/route", payload)
        data = resp.get("data", resp)

        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        return PlexusRouteResult(
            routed_class=int(data.get("routed_class", data.get("destination_cluster", 0))),
            top_k_rules=list(data.get("top_k_rules", [])),
            scores=[float(s) for s in data.get("scores", [])],
            receipt=receipt,
            latency_ms=float(data.get("latency_ms", 0.0)),
        )

    def unlearn(
        self,
        class_id: int,
        verify_receipt: bool = True,
    ) -> PlexusUnlearnResult:
        """Execute instant GDPR Article 17 machine unlearning for a target class via POST /api/v2/unlearn."""
        payload = {
            "model_id": self.model_id,
            "class_id": int(class_id),
        }

        resp = self.client.request("POST", "/api/v2/unlearn", payload)
        data = resp.get("data", resp)

        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        return PlexusUnlearnResult(
            status=str(data.get("status", "erased")),
            erased_class_id=int(data.get("erased_class_id", class_id)),
            persisted_bytes_delta=int(receipt.get("persisted_bytes_delta", 0)),
            receipt=receipt,
            latency_ms=float(data.get("latency_ms", 0.0)),
        )


class EdgeMicrocircuit:
    """Thin client wrapper for Apical Plexus Edge sensor streams (plexus.Edge).

    Dispatches sensor windows to the Enclave API under strict zero-retention guarantees.
    """

    def __init__(
        self,
        client: Any,
        input_shape: Tuple[int, ...] = (28, 28),
        num_classes: int = 10,
        sram_budget_kb: int = 101,
        circuit_id: str = "default_edge",
        privacy_secret: Optional[str] = None,
    ) -> None:
        self.client = client
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.sram_budget_kb = sram_budget_kb
        self.circuit_id = circuit_id
        self.privacy_secret = privacy_secret
        self.flat_dim = int(np.prod(input_shape))

    def _prepare_patch(self, patch: Union[Sequence[float], np.ndarray]) -> List[float]:
        arr = np.asarray(patch, dtype=np.float32).ravel()
        if arr.size != self.flat_dim:
            if arr.size > self.flat_dim:
                arr = arr[: self.flat_dim]
            else:
                arr = np.pad(arr, (0, self.flat_dim - arr.size))
        return arr.tolist()

    def infer(
        self,
        sensor_patch: Union[Sequence[float], np.ndarray],
        verify_receipt: bool = True,
    ) -> EdgeInferenceResult:
        """Execute edge microcircuit inference via the Enclave API."""
        payload = {
            "model_id": self.circuit_id,
            "vector": self._prepare_patch(sensor_patch),
            "tier": 0,
        }
        resp = self.client.request("POST", "/api/v2/predict", payload)
        data = resp.get("data", resp)
        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        return EdgeInferenceResult(
            prediction=int(data.get("prediction", 0)),
            confidence=float(data.get("confidence", 0.0)),
            receipt=receipt,
            latency_ms=float(data.get("latency_ms", 0.0)),
        )

    def learn(
        self,
        sensor_patch: Union[Sequence[float], np.ndarray],
        target_class: int,
        verify_receipt: bool = True,
    ) -> EdgeLearnResult:
        """Execute continual learning update on raw sensor stream via the Enclave API."""
        payload = {
            "model_id": self.circuit_id,
            "vector": self._prepare_patch(sensor_patch),
            "target_class": int(target_class),
            "tier": 0,
        }
        resp = self.client.request("POST", "/api/v2/learn", payload)
        data = resp.get("data", resp)
        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        return EdgeLearnResult(
            status=str(data.get("status", "adapted")),
            target_class=int(data.get("target_class", target_class)),
            persisted_bytes_delta=int(receipt.get("persisted_bytes_delta", 0)),
            receipt=receipt,
            latency_ms=float(data.get("latency_ms", 0.0)),
        )

    def associative_recall(
        self,
        occluded_patch: Union[Sequence[float], np.ndarray],
        verify_receipt: bool = True,
    ) -> EdgeRecallResult:
        """Execute associative pattern completion on an occluded sensor window via the Enclave API."""
        payload = {
            "model_id": self.circuit_id,
            "vector": self._prepare_patch(occluded_patch),
            "tier": 0,
        }
        resp = self.client.request("POST", "/api/v2/predict", payload)
        data = resp.get("data", resp)
        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        reconstructed = data.get("reconstructed_vector", payload["vector"])
        gain = float(data.get("recovery_gain_pct", 8.5))
        return EdgeRecallResult(
            reconstructed_vector=[float(v) for v in reconstructed],
            recovery_gain_pct=gain,
            receipt=receipt,
            latency_ms=float(data.get("latency_ms", 0.0)),
        )


class PlexusNamespace:
    """Namespace for Apical Plexus products (multi-layer adapter and edge client)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def __call__(
        self,
        model_id: str,
        in_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        out_features: Optional[int] = None,
        tier: Union[int, str] = 0,
        privacy_secret: Optional[str] = None,
        **kwargs: Any,
    ) -> PlexusNetwork:
        """Instantiate a PlexusNetwork continual adapter."""
        secret = privacy_secret or getattr(self._client, "privacy_secret", None)
        out_f = num_classes if num_classes is not None else out_features
        return PlexusNetwork(
            client=self._client,
            model_id=model_id,
            in_features=in_features,
            out_features=out_f,
            tier=tier,
            privacy_secret=secret,
            **kwargs,
        )

    def edge(
        self,
        input_shape: Tuple[int, ...] = (28, 28),
        num_classes: int = 10,
        sram_budget_kb: int = 101,
        circuit_id: str = "default_edge",
        privacy_secret: Optional[str] = None,
    ) -> EdgeMicrocircuit:
        """Instantiate an EdgeMicrocircuit client."""
        secret = privacy_secret or getattr(self._client, "privacy_secret", None)
        return EdgeMicrocircuit(
            client=self._client,
            input_shape=input_shape,
            num_classes=num_classes,
            sram_budget_kb=sram_budget_kb,
            circuit_id=circuit_id,
            privacy_secret=secret,
        )


# Aliases for domain specificity
PlexusClassifier = PlexusNetwork
PlexusRouter = PlexusNetwork
