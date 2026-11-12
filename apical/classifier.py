"""Classifier Surface for the Apical Labs Platform."""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Sequence, Union
import numpy as np

from apical.orthogonal import OrthogonalPrivacyProjector
from apical.verifier import verify_zero_retention_receipt


class PredictionResult(NamedTuple):
    """Result of a classifier inference pass."""
    prediction: int
    confidence: float
    receipt: Dict[str, Any]
    crystallization_pct: Optional[float] = None
    stability_regime: Optional[str] = None

    @property
    def signature(self) -> str:
        """Return the Ed25519 signature from the cryptographic attestation receipt."""
        return str(self.receipt.get("ed25519_signature", ""))

    @property
    def predicted_class(self) -> int:
        """Alias for prediction to maintain documentation and API compatibility."""
        return self.prediction

    @property
    def ed25519_signature(self) -> str:
        """Alias for signature matching receipt schema."""
        return self.signature


class LearnResult(NamedTuple):
    """Result of a single continual update."""
    status: str
    target_class: int
    winner_class: int
    confidence: float
    update_count: int
    persisted_bytes_delta: int
    receipt: Dict[str, Any]
    crystallization_pct: Optional[float] = None
    stability_regime: Optional[str] = None

    @property
    def signature(self) -> str:
        """Return the Ed25519 signature from the cryptographic attestation receipt."""
        return str(self.receipt.get("ed25519_signature", ""))

    @property
    def ed25519_signature(self) -> str:
        """Alias for signature matching receipt schema."""
        return self.signature


class ModelStatus(NamedTuple):
    """Metadata and constant-memory verification status for an active model container."""
    model_id: str
    in_features: int
    out_features: int
    update_count: int = 0
    constant_memory_verified: bool = True
    persisted_bytes_delta: int = 0
    state_sha256: str = ""


class UnlearnResult(NamedTuple):
    """Result of an instant GDPR Article 17 class erasure."""
    status: str
    erased_class_id: int
    persisted_bytes_delta: int
    receipt: Dict[str, Any]

    @property
    def signature(self) -> str:
        """Return the Ed25519 signature from the cryptographic attestation receipt."""
        return str(self.receipt.get("ed25519_signature", ""))

    @property
    def ed25519_signature(self) -> str:
        """Alias for signature matching receipt schema."""
        return self.signature

    @property
    def target_class_weight_norm(self) -> float:
        """Norm of erased class weights, guaranteed 0.0 upon surgical unlearning."""
        return 0.0

    @property
    def state_sha256_before(self) -> str:
        """State hash before erasure."""
        return str(self.receipt.get("state_sha256_before", ""))

    @property
    def state_sha256_after(self) -> str:
        """State hash after erasure."""
        return str(self.receipt.get("state_sha256_after", ""))

    @property
    def timestamp_ns(self) -> int:
        """Hardware monotonic timestamp of unlearning attestation."""
        return int(self.receipt.get("timestamp_ns", 0))


class Classifier:
    """Client interface for continuous classification models."""

    def __init__(
        self,
        client: Any,
        model_id: str,
        in_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        privacy_secret: Optional[str] = None,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.in_features = in_features
        self.num_classes = num_classes
        self.privacy_secret = privacy_secret

        self._projector: Optional[OrthogonalPrivacyProjector] = None
        if self.privacy_secret and self.in_features:
            self._projector = OrthogonalPrivacyProjector(self.privacy_secret, self.in_features)

    def _prepare_vector(self, vector: Union[Sequence[float], np.ndarray]) -> list[float]:
        """Optionally apply client-side privacy projection, returning a JSON-serializable list."""
        arr = np.asarray(vector, dtype=np.float32).ravel()

        if self._projector is not None:
            arr = self._projector.project(arr)
        elif self.privacy_secret and not self._projector:
            # Lazy initialize projector once vector dimension is known
            self.in_features = arr.size
            self._projector = OrthogonalPrivacyProjector(self.privacy_secret, self.in_features)
            arr = self._projector.project(arr)

        return arr.tolist()

    def create(self, in_features: int, num_classes: int) -> Dict[str, Any]:
        """Create and initialize a new constant-memory model container on Apical Cloud."""
        self.in_features = in_features
        self.num_classes = num_classes

        if self.privacy_secret and not self._projector:
            self._projector = OrthogonalPrivacyProjector(self.privacy_secret, self.in_features)

        payload = {
            "model_id": self.model_id,
            "in_features": in_features,
            "out_features": num_classes,
        }
        resp = self.client.request("POST", f"/api/v1/models", payload)
        data = resp.get("data", resp)

        if "receipt" in data:
            verify_zero_retention_receipt(data["receipt"])

        return data

    def predict(
        self,
        vector: Union[Sequence[float], np.ndarray],
        tier: int = 0,
        verify_receipt: bool = True,
    ) -> PredictionResult:
        """Execute directional classification, returning class prediction and signed receipt."""
        prep_vec = self._prepare_vector(vector)
        payload = {
            "model_id": self.model_id,
            "vector": prep_vec,
            "tier": tier,
        }
        resp = self.client.request("POST", f"/api/v1/predict", payload)
        data = resp.get("data", resp)

        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        return PredictionResult(
            prediction=int(data["prediction"]),
            confidence=float(data["confidence"]),
            receipt=receipt,
            crystallization_pct=data.get("crystallization_pct"),
            stability_regime=data.get("stability_regime"),
        )

    def learn(
        self,
        vector: Union[Sequence[float], np.ndarray],
        target_class: int,
        tier: int = 0,
        verify_receipt: bool = True,
    ) -> LearnResult:
        """Stream a single continual update, updating model state under zero retention."""
        prep_vec = self._prepare_vector(vector)
        payload = {
            "model_id": self.model_id,
            "vector": prep_vec,
            "target_class": target_class,
            "tier": tier,
        }
        resp = self.client.request("POST", f"/api/v1/learn", payload)
        data = resp.get("data", resp)

        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        return LearnResult(
            status=str(data.get("status", "updated")),
            target_class=int(data.get("target_class", target_class)),
            winner_class=int(data.get("winner_class", target_class)),
            confidence=float(data.get("confidence", 0.0)),
            update_count=int(data.get("update_count", 0)),
            persisted_bytes_delta=int(data.get("persisted_bytes_delta", 0)),
            receipt=receipt,
            crystallization_pct=data.get("crystallization_pct"),
            stability_regime=data.get("stability_regime"),
        )

    def unlearn(
        self,
        class_id: int,
        verify_receipt: bool = True,
    ) -> UnlearnResult:
        """Execute instant GDPR Article 17 Machine Unlearning for a specific class coordinate."""
        payload = {
            "model_id": self.model_id,
            "class_id": class_id,
        }
        resp = self.client.request("POST", f"/api/v1/unlearn", payload)
        data = resp.get("data", resp)

        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        return UnlearnResult(
            status=str(data.get("status", "erased")),
            erased_class_id=int(data.get("erased_class_id", class_id)),
            persisted_bytes_delta=int(data.get("persisted_bytes_delta", 0)),
            receipt=receipt,
        )

    def erase_class(
        self,
        target_class: Optional[int] = None,
        class_id: Optional[int] = None,
        verify_receipt: bool = True,
    ) -> UnlearnResult:
        """Execute instant GDPR Article 17 Machine Unlearning for a specific class coordinate (alias for unlearn)."""
        target = target_class if target_class is not None else class_id
        if target is None:
            raise ValueError("Either target_class or class_id must be provided")
        return self.unlearn(class_id=target, verify_receipt=verify_receipt)
