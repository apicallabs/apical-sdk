"""Apical Labs — Zero-Retention Continual Memory & Embedding Router.

Client SDK providing:
- ApicalClient: Main client connection pool
- Classifier: Continual classification with instant GDPR Art. 17 unlearning
- MemoryRouter: Real-time dynamic embedding routing and reinforcement
- OrthogonalPrivacyProjector: Client-side mathematical input de-identification
- verify_zero_retention_receipt: Offline Ed25519 receipt verification
"""

from __future__ import annotations

from apical.classifier import (
    Classifier,
    LearnResult,
    ModelStatus,
    PredictionResult,
    UnlearnResult,
)
from apical.client import ApicalClient, ArborNamespace, BridgeResult

# Backward-compatible convenience alias
Client = ApicalClient
from apical.exceptions import (
    APIError,
    ApicalError,
    AuthenticationError,
    FixedByteInvariantError,
    ModelNotFoundError,
    ProductNotEntitledError,
    RateLimitError,
    SparseProbeError,
    SweepDetectedError,
    VerificationError,
)
from apical.orthogonal import OrthogonalPrivacyProjector
from apical.plexus import (
    EdgeInferenceResult,
    EdgeLearnResult,
    EdgeMicrocircuit,
    EdgeRecallResult,
    PlexusClassifier,
    PlexusLearnResult,
    PlexusNamespace,
    PlexusNetwork,
    PlexusPredictionResult,
    PlexusRouteResult,
    PlexusRouter,
    PlexusUnlearnResult,
)
from apical.router import (
    MemoryRouter,
    ReinforceResult,
    RouteItem,
    RouteResult,
)
from apical.catalog import (
    CANONICAL_TOOL_CATALOG,
    TOOL_CATALOG_BY_ID,
    DynamicToolRegistry,
    ToolDefinition,
    calculate_token_savings,
    get_default_registry,
    get_tool_by_id,
)
from apical.enclave_client import (
    EnclaveConnectionError,
    EnclaveDirectClient,
    EnclaveRPCError,
)
from apical.preference import (
    DeveloperPreferenceAdapter,
    PreferenceMatch,
)
from apical.telemetry import (
    DEFAULT_TELEMETRY_LOG,
    TelemetryRecord,
    ZeroRetentionTelemetryLogger,
    get_telemetry_logger,
)
from apical.proxy import (
    PlexusProxyHandler,
    create_proxy_server,
)
from apical.verifier import verify_zero_retention_receipt


__version__ = "0.5.5"

__all__ = [
    "APIError",
    "ApicalClient",
    "ApicalError",
    "ArborNamespace",
    "AuthenticationError",
    "BridgeResult",
    "Classifier",
    "Client",
    "EdgeInferenceResult",
    "EdgeLearnResult",
    "EdgeMicrocircuit",
    "EdgeRecallResult",
    "FixedByteInvariantError",
    "LearnResult",
    "MemoryRouter",
    "ModelNotFoundError",
    "ModelStatus",
    "OrthogonalPrivacyProjector",
    "PlexusClassifier",
    "PlexusLearnResult",
    "PlexusNamespace",
    "PlexusNetwork",
    "PlexusPredictionResult",
    "PlexusRouteResult",
    "PlexusRouter",
    "PlexusUnlearnResult",
    "PredictionResult",
    "ProductNotEntitledError",
    "RateLimitError",
    "ReinforceResult",
    "RouteItem",
    "RouteResult",
    "SparseProbeError",
    "SweepDetectedError",
    "CANONICAL_TOOL_CATALOG",
    "calculate_token_savings",
    "create_proxy_server",
    "DeveloperPreferenceAdapter",
    "DynamicToolRegistry",
    "EnclaveConnectionError",
    "EnclaveDirectClient",
    "EnclaveRPCError",
    "get_default_registry",
    "get_telemetry_logger",
    "PlexusProxyHandler",
    "PreferenceMatch",
    "TelemetryRecord",
    "ToolDefinition",
    "UnlearnResult",
    "VerificationError",
    "verify_zero_retention_receipt",
    "ZeroRetentionTelemetryLogger",
]

