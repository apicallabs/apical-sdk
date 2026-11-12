"""MemoryRouter Surface for the Apical Labs Platform."""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Union
import numpy as np

from apical.orthogonal import OrthogonalPrivacyProjector
from apical.verifier import verify_zero_retention_receipt


class RouteItem(NamedTuple):
    """An individual routing rule candidate with affinity score."""
    rule_id: int
    affinity: float

    @property
    def tool_id(self) -> int:
        """Alias for rule_id when routing across tool catalogs."""
        return self.rule_id


class RouteResult(NamedTuple):
    """Result of an embedding routing pass."""
    routes: List[RouteItem]
    top_rule_id: Optional[int]
    receipt: Dict[str, Any]

    @property
    def candidates(self) -> List[RouteItem]:
        """Alias for routes returning shortlisted candidate items."""
        return self.routes

    def __iter__(self):
        """Allow direct iteration over shortlisted candidate items."""
        return iter(self.routes)

    def __getitem__(self, index):
        """Allow indexing into shortlisted candidate items."""
        return self.routes[index]

    def __len__(self) -> int:
        return len(self.routes)


class ReinforceResult(NamedTuple):
    """Result of a memory rule reinforcement update."""
    status: str
    target_rule: int
    confidence: float
    update_count: int
    persisted_bytes_delta: int
    receipt: Dict[str, Any]


class MemoryRouter:
    """Client interface for continuous embedding routers and memory partitioning."""

    def __init__(
        self,
        client: Any,
        router_id: str,
        embedding_dim: Optional[int] = None,
        num_rules: Optional[int] = None,
        in_features: Optional[int] = None,
        num_tools: Optional[int] = None,
        privacy_secret: Optional[str] = None,
    ) -> None:
        self.client = client
        self.router_id = router_id
        self.embedding_dim = embedding_dim if embedding_dim is not None else in_features
        self.num_rules = num_rules if num_rules is not None else num_tools
        self.privacy_secret = privacy_secret

        self._projector: Optional[OrthogonalPrivacyProjector] = None
        if self.privacy_secret and self.embedding_dim:
            self._projector = OrthogonalPrivacyProjector(self.privacy_secret, self.embedding_dim)

    def _prepare_vector(self, embedding: Union[Sequence[float], np.ndarray]) -> list[float]:
        """Optionally apply orthogonal privacy projection, returning a JSON-serializable list."""
        arr = np.asarray(embedding, dtype=np.float32).ravel()

        if self._projector is not None:
            arr = self._projector.project(arr)
        elif self.privacy_secret and not self._projector:
            self.embedding_dim = arr.size
            self._projector = OrthogonalPrivacyProjector(self.privacy_secret, self.embedding_dim)
            arr = self._projector.project(arr)

        return arr.tolist()

    def create(self, embedding_dim: int, num_rules: int) -> Dict[str, Any]:
        """Create and initialize a new memory router on Apical Cloud."""
        self.embedding_dim = embedding_dim
        self.num_rules = num_rules

        if self.privacy_secret and not self._projector:
            self._projector = OrthogonalPrivacyProjector(self.privacy_secret, self.embedding_dim)

        payload = {
            "model_id": self.router_id,
            "in_features": embedding_dim,
            "out_features": num_rules,
        }
        resp = self.client.request("POST", f"/api/v1/models", payload)
        data = resp.get("data", resp)

        if "receipt" in data:
            verify_zero_retention_receipt(data["receipt"])

        return data

    def route(
        self,
        embedding: Union[Sequence[float], np.ndarray],
        top_k: int = 3,
        tier: int = 0,
        verify_receipt: bool = True,
    ) -> RouteResult:
        """Route input embedding across partitions, returning top K activated rules."""
        prep_vec = self._prepare_vector(embedding)
        payload = {
            "model_id": self.router_id,
            "vector": prep_vec,
            "top_k": top_k,
            "tier": tier,
        }
        resp = self.client.request("POST", f"/api/v1/route", payload)
        data = resp.get("data", resp)

        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        raw_routes = data.get("routes", [])
        routes = [RouteItem(rule_id=int(r["rule_id"]), affinity=float(r["affinity"])) for r in raw_routes]

        return RouteResult(
            routes=routes,
            top_rule_id=data.get("top_rule_id"),
            receipt=receipt,
        )

    def reinforce(
        self,
        embedding: Union[Sequence[float], np.ndarray],
        target_rule: Optional[int] = None,
        selected_tool_id: Optional[int] = None,
        tier: int = 0,
        verify_receipt: bool = True,
    ) -> ReinforceResult:
        """Reinforce a target routing pathway under zero retention."""
        rule = target_rule if target_rule is not None else selected_tool_id
        if rule is None:
            raise ValueError("Either target_rule or selected_tool_id must be provided")

        prep_vec = self._prepare_vector(embedding)
        payload = {
            "model_id": self.router_id,
            "vector": prep_vec,
            "target_rule": rule,
            "tier": tier,
        }
        resp = self.client.request("POST", f"/api/v1/reinforce", payload)
        data = resp.get("data", resp)

        receipt = data.get("receipt", {})
        if verify_receipt and receipt:
            verify_zero_retention_receipt(receipt)

        return ReinforceResult(
            status=str(data.get("status", "updated")),
            target_rule=int(data.get("target_class", target_rule)),
            confidence=float(data["confidence"]),
            update_count=int(data.get("update_count", 0)),
            persisted_bytes_delta=int(data.get("persisted_bytes_delta", 0)),
            receipt=receipt,
        )
