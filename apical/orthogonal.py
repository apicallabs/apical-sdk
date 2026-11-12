"""Client-Side Privacy Projector for Zero-Retention Blindness.

Applies a deterministic, customer-held linear projection before feature
vectors leave the customer's infrastructure.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import numpy as np


SALT_ORTHOGONAL_V1 = b"apical-orthogonal-v1"


class OrthogonalPrivacyProjector:
    """Deterministic client-side privacy projector.

    Derives a fixed-dimension projection operator from a client passphrase.
    The secret key never leaves the client premises.
    """

    def __init__(
        self,
        privacy_secret: str,
        in_features: Optional[int] = None,
        dimension: Optional[int] = None,
    ) -> None:
        dim = in_features if in_features is not None else dimension
        if not privacy_secret:
            raise ValueError("privacy_secret cannot be empty")
        if dim is None or dim < 2:
            raise ValueError(f"in_features (or dimension) must be specified and >= 2, got {dim}")

        self.in_features = dim

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=SALT_ORTHOGONAL_V1,
            info=f"apical-orthogonal-v1-{dim}".encode("utf-8"),
        )
        seed_bytes = hkdf.derive(privacy_secret.encode("utf-8"))
        seed_int = int.from_bytes(seed_bytes, "big")

        rng = np.random.Generator(np.random.PCG64(seed_int))
        _m = rng.standard_normal((dim, dim), dtype=np.float64)
        _u, _t = np.linalg.qr(_m)
        _s = np.sign(np.diag(_t))
        _s[_s == 0] = 1.0
        self._q = (_u * _s).astype(np.float32)

    @property
    def matrix(self) -> np.ndarray:
        """The derived projection operator."""
        return self._q

    def project(self, vector: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Project a single feature vector into the client-blinded subspace."""
        arr = np.asarray(vector, dtype=np.float32).ravel()
        if arr.size != self.in_features:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.in_features}, got {arr.size}"
            )

        return self._q @ arr

    def project_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Project a batch of feature vectors of shape (N, d)."""
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.in_features:
            raise ValueError(
                f"Batch shape mismatch: expected (N, {self.in_features}), got {arr.shape}"
            )

        return arr @ self._q.T
