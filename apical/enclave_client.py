"""High-Performance Line-Delimited JSON-RPC Client for Apical Enclave.

Provides direct local transport to the air-gapped Apical Enclave worker via:
1. Primary UNIX Domain Socket (/run/apical-enclave/enclave.sock)
2. Fallback UNIX Domain Socket (/tmp/apical-enclave.sock)
3. TCP Loopback (127.0.0.1:8765) to bypass systemd PrivateTmp isolation

Air-Gapped Invariants:
- Zero external HTTP/network egress
- Constant O(1) memory verification (persisted_bytes_delta == 0)
- Ed25519 cryptographic receipt attestation
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import socket
import time
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger("ApicalEnclaveClient")

DEFAULT_PRIMARY_SOCKET = "/run/apical-enclave/enclave.sock"
DEFAULT_FALLBACK_SOCKET = "/tmp/apical-enclave.sock"
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 8765
DEFAULT_TIMEOUT_SECONDS = 30.0


class EnclaveConnectionError(Exception):
    """Raised when connection to the air-gapped enclave fails across all transports."""


class EnclaveRPCError(Exception):
    """Raised when the enclave returns an error response."""

    def __init__(self, message: str, code: str = "ERR_ENCLAVE_RPC", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.details = details or {}


class EnclaveDirectClient:
    """Direct, synchronous socket RPC client connecting to the running Apical Enclave."""

    def __init__(
        self,
        socket_path: Optional[str] = None,
        tcp_port: int = DEFAULT_TCP_PORT,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.socket_path = socket_path or os.environ.get("APICAL_ENCLAVE_SOCK", DEFAULT_PRIMARY_SOCKET)
        self.tcp_port = tcp_port
        self.timeout = timeout

    def _open_connection(self) -> socket.socket:
        """Establish connection trying primary UNIX socket, fallback UNIX socket, and loopback TCP."""
        # 1. Try configured socket path if exists
        sock_candidates = []
        if self.socket_path and self.socket_path != DEFAULT_FALLBACK_SOCKET:
            sock_candidates.append(self.socket_path)
        sock_candidates.append(DEFAULT_FALLBACK_SOCKET)

        for sock_file in sock_candidates:
            if os.path.exists(sock_file):
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.settimeout(self.timeout)
                    s.connect(sock_file)
                    return s
                except (OSError, ConnectionRefusedError):
                    s.close()

        # 2. Try TCP loopback fallback (bypasses systemd PrivateTmp isolation)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect((DEFAULT_TCP_HOST, self.tcp_port))
            return s
        except (OSError, ConnectionRefusedError) as e:
            s.close()
            raise EnclaveConnectionError(
                f"Cannot connect to Apical Enclave via UNIX sockets {sock_candidates} "
                f"or TCP loopback ({DEFAULT_TCP_HOST}:{self.tcp_port}). "
                "Ensure web.enclave.server is running."
            ) from e

    def call(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a single JSON-RPC payload to the enclave and parse response."""
        s = self._open_connection()
        try:
            payload_bytes = json.dumps(request_payload).encode("utf-8") + b"\n"
            s.sendall(payload_bytes)

            raw_buffer = bytearray()
            while True:
                chunk = s.recv(16384)
                if not chunk:
                    break
                raw_buffer.extend(chunk)
                if b"\n" in chunk:
                    break

            if not raw_buffer:
                raise EnclaveRPCError("Enclave returned empty or prematurely closed socket response")

            line = raw_buffer.decode("utf-8").strip()
            response = json.loads(line)

            if not response.get("ok", False):
                err_msg = response.get("error", "Unknown enclave error")
                err_code = response.get("code", "ERR_ENCLAVE_DISPATCH")
                raise EnclaveRPCError(err_msg, code=err_code, details=response)

            return response.get("data", response)
        finally:
            s.close()

    def ping(self) -> Dict[str, Any]:
        """Ping the enclave to verify readiness, PCR0, and Ed25519 public key."""
        return self.call({"operation": "ping"})

    def sandbox_route_query(self, query: str, top_k: int = 3, tier: int = 0) -> Dict[str, Any]:
        """Route natural language query using real all-MiniLM-L6-v2 embeddings on the enclave."""
        return self.call({
            "operation": "sandbox_route_query",
            "query": str(query),
            "top_k": int(top_k),
            "tier": int(tier),
        })

    def route_tools(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        top_k: int = 3,
        tier: int = 0,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route natural language query across arbitrary tool definitions on the enclave."""
        payload: Dict[str, Any] = {
            "operation": "route_tools",
            "query": str(query),
            "tools": tools,
            "top_k": int(top_k),
            "tier": int(tier),
        }
        if embedding_model:
            payload["embedding_model"] = str(embedding_model)
        return self.call(payload)

    def route(
        self,
        tenant_id: str,
        model_id: str,
        query: Optional[str] = None,
        vector: Optional[Sequence[float]] = None,
        top_k: int = 3,
        tier: int = 0,
    ) -> Dict[str, Any]:
        """Route natural language query or feature vector through fixed-byte synaptic model."""
        payload: Dict[str, Any] = {
            "operation": "route",
            "tenant_id": tenant_id,
            "model_id": model_id,
            "top_k": int(top_k),
            "tier": int(tier),
        }
        if query is not None:
            payload["query"] = str(query)
        elif vector is not None:
            payload["vector"] = list(vector)
        else:
            raise ValueError("Either 'query' or 'vector' must be provided to route()")
        return self.call(payload)

    def reinforce(
        self,
        tenant_id: str,
        model_id: str,
        query: Optional[str] = None,
        vector: Optional[Sequence[float]] = None,
        target_rule: int = 0,
        tier: int = 0,
    ) -> Dict[str, Any]:
        """Reinforce a target routing pathway under zero retention (persisted_bytes_delta == 0)."""
        payload: Dict[str, Any] = {
            "operation": "reinforce",
            "tenant_id": tenant_id,
            "model_id": model_id,
            "target_rule": int(target_rule),
            "tier": int(tier),
        }
        if query is not None:
            payload["query"] = str(query)
        elif vector is not None:
            payload["vector"] = list(vector)
        else:
            raise ValueError("Either 'query' or 'vector' must be provided to reinforce()")
        return self.call(payload)

    def learn(
        self,
        tenant_id: str,
        model_id: str,
        query: Optional[str] = None,
        vector: Optional[Sequence[float]] = None,
        target_class: int = 0,
        tier: int = 0,
    ) -> Dict[str, Any]:
        """Execute continual learning update on synaptic weights with Ed25519 attestation."""
        payload: Dict[str, Any] = {
            "operation": "learn",
            "tenant_id": tenant_id,
            "model_id": model_id,
            "target_class": int(target_class),
            "tier": int(tier),
        }
        if query is not None:
            payload["query"] = str(query)
        elif vector is not None:
            payload["vector"] = list(vector)
        else:
            raise ValueError("Either 'query' or 'vector' must be provided to learn()")
        return self.call(payload)

    def create_model(
        self,
        tenant_id: str,
        model_id: str,
        in_features: int,
        out_features: int,
    ) -> Dict[str, Any]:
        """Pre-allocate fixed-byte Apical Arbor model container."""
        return self.call({
            "operation": "create",
            "tenant_id": tenant_id,
            "model_id": model_id,
            "in_features": int(in_features),
            "out_features": int(out_features),
        })

    def plexus_create(
        self,
        tenant_id: str,
        model_id: str,
        in_features: int,
        out_features: int = 10,
        tier: int = 0,
        **_: Any,
    ) -> Dict[str, Any]:
        """Pre-allocate fixed-byte Apical Plexus model container."""
        return self.call({
            "operation": "plexus_create",
            "tenant_id": tenant_id,
            "model_id": model_id,
            "in_features": int(in_features),
            "out_features": int(out_features),
            "tier": int(tier),
        })

    def plexus_route(
        self,
        tenant_id: str,
        model_id: str,
        vector: Sequence[float],
        top_k: int = 3,
        tier: int = 0,
    ) -> Dict[str, Any]:
        """Route feature vector through Apical Plexus router."""
        return self.call({
            "operation": "plexus_route",
            "tenant_id": tenant_id,
            "model_id": model_id,
            "vector": list(vector),
            "top_k": int(top_k),
            "tier": int(tier),
        })

    def plexus_learn(
        self,
        tenant_id: str,
        model_id: str,
        vector: Sequence[float],
        target_class: int,
        tier: int = 0,
    ) -> Dict[str, Any]:
        """Execute Apical Plexus continual adaptation."""
        return self.call({
            "operation": "plexus_learn",
            "tenant_id": tenant_id,
            "model_id": model_id,
            "vector": list(vector),
            "target_class": int(target_class),
            "tier": int(tier),
        })

    def shred_model(
        self,
        tenant_id: str,
        model_id: str,
    ) -> Dict[str, Any]:
        """Cryptographically shred an allocated model container on the enclave."""
        return self.call({
            "operation": "shred_model",
            "tenant_id": tenant_id,
            "model_id": model_id,
        })

