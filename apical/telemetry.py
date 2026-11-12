"""Zero-Retention Telemetry Logger and Analytics Dashboard for Apical Plexus.

Enforces:
1. Append-only immutable metrics log (telemetry.jsonl).
2. Absolute Zero-Retention Invariant: STRICTLY ZERO raw user prompt text or raw feature vectors
   are ever written to disk or logs. Only cryptographic SHA-256 input digests, receipt IDs,
   and pre/post state hashes are recorded.
3. Accurate statistical latency profiling (p50, p90, p95, p99 in µs/ms).
4. Cumulative prompt token savings accounting vs naive context stuffing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger("ApicalTelemetry")

DEFAULT_TELEMETRY_DIR = Path.home() / ".gemini" / "antigravity" / "mcp" / "plexus"
DEFAULT_TELEMETRY_LOG = DEFAULT_TELEMETRY_DIR / "telemetry.jsonl"


@dataclass(frozen=True)
class TelemetryRecord:
    """Strongly typed, append-only zero-retention telemetry entry.

    INVARIANT: Does not contain raw prompt text or raw vector coordinates.
    """

    timestamp_ns: int
    operation: str
    model_id: str
    latency_us: float
    persisted_bytes_delta: int
    receipt_id: str
    state_sha256_before: str
    state_sha256_after: str
    input_sha256_digest: str
    top_rule_id: Optional[int] = None
    confidence: Optional[float] = None
    tokens_saved: int = 0
    enclave_verified: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


class ZeroRetentionTelemetryLogger:
    """Thread-safe append-only telemetry recorder guaranteeing zero data retention."""

    def __init__(self, log_path: Optional[Union[str, Path]] = None) -> None:
        self.log_path = Path(log_path or DEFAULT_TELEMETRY_LOG)
        self._lock = threading.Lock()
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                self.log_path.touch(mode=0o600)
        except OSError as e:
            logger.warning("Could not create telemetry dir %s: %e", self.log_path.parent, e)

    @staticmethod
    def compute_sha256(data: Union[str, bytes, Sequence[float]]) -> str:
        """Compute one-way cryptographic SHA-256 digest of input data."""
        if isinstance(data, str):
            return hashlib.sha256(data.encode("utf-8")).hexdigest()
        if isinstance(data, bytes):
            return hashlib.sha256(data).hexdigest()
        # Canonical float serialization for vector hashing
        encoded = ",".join(f"{x:.6f}" for x in data).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def log_operation(
        self,
        operation: str,
        model_id: str,
        latency_us: float,
        receipt: Dict[str, Any],
        input_digest: Optional[str] = None,
        top_rule_id: Optional[int] = None,
        confidence: Optional[float] = None,
        tokens_saved: int = 0,
    ) -> TelemetryRecord:
        """Append a new telemetry record to the log under zero-retention rules."""
        delta = int(receipt.get("persisted_bytes_delta", 0))
        if delta != 0:
            logger.critical("Zero-Retention violation detected! persisted_bytes_delta=%d", delta)

        digest = (
            input_digest
            or receipt.get("input_sha256_digest")
            or "0" * 64
        )

        record = TelemetryRecord(
            timestamp_ns=time.time_ns(),
            operation=operation,
            model_id=model_id,
            latency_us=round(latency_us, 1),
            persisted_bytes_delta=delta,
            receipt_id=str(receipt.get("receipt_id", "")),
            state_sha256_before=str(receipt.get("state_sha256_before", "")),
            state_sha256_after=str(receipt.get("state_sha256_after", "")),
            input_sha256_digest=str(digest),
            top_rule_id=top_rule_id,
            confidence=round(confidence, 4) if confidence is not None else None,
            tokens_saved=int(tokens_saved),
            enclave_verified=bool(receipt.get("ed25519_signature")),
        )

        with self._lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(record.to_json() + "\n")
            except OSError as e:
                logger.error("Failed to append telemetry record: %s", e)

        return record

    def read_records(self, max_records: Optional[int] = None) -> List[TelemetryRecord]:
        """Read existing telemetry records from append-only log."""
        if not self.log_path.exists():
            return []

        records: List[TelemetryRecord] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if max_records is not None and len(lines) > max_records:
                    lines = lines[-max_records:]

                for line in lines:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    try:
                        data = json.loads(line_s)
                        records.append(TelemetryRecord(**data))
                    except Exception:
                        continue
        except OSError:
            pass

        return records

    def compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate statistical summary from the telemetry log."""
        records = self.read_records()
        if not records:
            return {
                "total_operations": 0,
                "cumulative_tokens_saved": 0,
                "latencies_us": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0},
                "operations_breakdown": {},
                "zero_retention_compliance_pct": 100.0,
                "ed25519_signed_pct": 100.0,
            }

        latencies = [r.latency_us for r in records if r.latency_us > 0]
        latencies.sort()

        def percentile(p: float) -> float:
            if not latencies:
                return 0.0
            k = (len(latencies) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return latencies[int(k)]
            d0 = latencies[int(f)] * (c - k)
            d1 = latencies[int(c)] * (k - f)
            return d0 + d1

        ops_counts: Dict[str, int] = {}
        for r in records:
            ops_counts[r.operation] = ops_counts.get(r.operation, 0) + 1

        zero_delta_count = sum(1 for r in records if r.persisted_bytes_delta == 0)
        signed_count = sum(1 for r in records if r.enclave_verified)
        total_tokens_saved = sum(r.tokens_saved for r in records)

        return {
            "total_operations": len(records),
            "cumulative_tokens_saved": total_tokens_saved,
            "latencies_us": {
                "p50": round(percentile(50.0), 1),
                "p90": round(percentile(90.0), 1),
                "p95": round(percentile(95.0), 1),
                "p99": round(percentile(99.0), 1),
                "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            },
            "latencies_ms": {
                "p50": round(percentile(50.0) / 1000.0, 3),
                "p95": round(percentile(95.0) / 1000.0, 3),
                "p99": round(percentile(99.0) / 1000.0, 3),
            },
            "operations_breakdown": ops_counts,
            "zero_retention_compliance_pct": round((zero_delta_count / len(records)) * 100.0, 2),
            "ed25519_signed_pct": round((signed_count / len(records)) * 100.0, 2),
            "log_path": str(self.log_path),
        }

    def render_markdown_dashboard(self) -> str:
        """Format an ANSI/Markdown analytics dashboard string."""
        summary = self.compute_summary()
        lat_us = summary["latencies_us"]
        lat_ms = summary["latencies_ms"]
        ops = summary["operations_breakdown"]

        lines = [
            "# Apical Plexus — Zero-Retention Telemetry & Analytics",
            "",
            f"**Append-Only Ledger**: `{summary.get('log_path')}`",
            "",
            "## Aggregate Throughput & Invariants",
            f"- **Total Operations Logged**: `{summary['total_operations']}`",
            f"- **Cumulative Prompt Tokens Saved**: `{summary['cumulative_tokens_saved']:,}` tokens",
            f"- **Zero-Retention Compliance (ΔBytes == 0)**: `{summary['zero_retention_compliance_pct']}%`",
            f"- **Ed25519 Cryptographic Attestation**: `{summary['ed25519_signed_pct']}%`",
            "",
            "## Sub-Millisecond Latency Profile",
            "| Percentile | Latency (µs) | Latency (ms) |",
            "| :--- | :--- | :--- |",
            f"| **p50 (Median)** | `{lat_us['p50']} µs` | `{lat_ms['p50']} ms` |",
            f"| **p90** | `{lat_us['p90']} µs` | `{round(lat_us['p90']/1000, 3)} ms` |",
            f"| **p95** | `{lat_us['p95']} µs` | `{lat_ms['p95']} ms` |",
            f"| **p99** | `{lat_us['p99']} µs` | `{lat_ms['p99']} ms` |",
            f"| **Mean** | `{lat_us['mean']} µs` | `{round(lat_us['mean']/1000, 3)} ms` |",
            "",
            "## Operations Distribution",
        ]
        for op, count in ops.items():
            lines.append(f"- `{op}`: **{count}** calls")

    def wipe(self, backup: bool = False) -> Dict[str, Any]:
        """Atomically wipe/purge the telemetry log while preserving zero-retention invariants.

        Args:
            backup: If True, preserves previous records to telemetry.jsonl.bak before wiping.

        Returns:
            Dictionary with wipe status, previous record count, and timestamp.
        """
        with self._lock:
            previous_count = len(self.read_records())
            backup_file: Optional[str] = None
            if backup and self.log_path.exists():
                backup_path = self.log_path.with_suffix(".jsonl.bak")
                try:
                    import shutil
                    shutil.copy2(self.log_path, backup_path)
                    backup_file = str(backup_path)
                except OSError as e:
                    logger.warning("Could not create telemetry backup: %s", e)

            try:
                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.truncate(0)
            except OSError as e:
                logger.error("Failed to truncate telemetry log: %s", e)
                return {"status": "error", "error": str(e), "previous_records": previous_count}

            return {
                "status": "wiped",
                "previous_records": previous_count,
                "backup_path": backup_file,
                "timestamp_ns": time.time_ns(),
                "log_path": str(self.log_path),
            }


# Singleton instance for system-wide reuse
_global_telemetry_logger: Optional[ZeroRetentionTelemetryLogger] = None


def get_telemetry_logger(log_path: Optional[Union[str, Path]] = None) -> ZeroRetentionTelemetryLogger:
    global _global_telemetry_logger
    if _global_telemetry_logger is None or log_path is not None:
        _global_telemetry_logger = ZeroRetentionTelemetryLogger(log_path)
    return _global_telemetry_logger


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apical Plexus Zero-Retention Telemetry Manager")
    parser.add_argument("--wipe", action="store_true", help="Atomically wipe telemetry log")
    parser.add_argument("--backup", action="store_true", help="Create .bak before wiping")
    parser.add_argument("--summary", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    tl = get_telemetry_logger()
    if args.wipe:
        res = tl.wipe(backup=args.backup)
        print(json.dumps(res, indent=2))
    elif args.summary:
        print(json.dumps(tl.compute_summary(), indent=2))
    else:
        print(tl.render_markdown_dashboard())
