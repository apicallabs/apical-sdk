"""Apical Plexus MCP Package."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("PlexusMcpServer", "run_mcp_stdio_server"):
        from apical.mcp import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PlexusMcpServer", "run_mcp_stdio_server"]
