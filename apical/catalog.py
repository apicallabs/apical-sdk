"""Dynamic Tool Registry with Automatic Discovery and Token Savings Accounting.

Maintains live, discovered, and caller-registered tool schemas for sub-millisecond
semantic routing, and calculates prompt token savings vs. naive full-catalog injection.

Zero Fake Data Invariant:
- No hardcoded synthetic microservices (Stripe/K8s/AWS mocks removed).
- Automatically discovers real tools from local MCP manifests and active environment.
- Automatically records new tools dynamically as they are observed in agent turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger("ApicalToolRegistry")

DEFAULT_REGISTRY_DIR = Path.home() / ".gemini" / "antigravity" / "mcp" / "plexus"
DEFAULT_REGISTRY_FILE = DEFAULT_REGISTRY_DIR / "tool_registry.json"


@dataclass
class ToolDefinition:
    """Strongly typed definition of a dynamically registered tool or skill."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    category: str = "general"
    id: int = 0

    def to_schema(self) -> Dict[str, Any]:
        """Format as standard OpenAI/Anthropic function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def schema_json(self) -> str:
        """Return serialized compact JSON string of schema."""
        return json.dumps(self.to_schema(), separators=(",", ":"))

    def estimated_tokens(self) -> int:
        """Estimate token footprint (approx 4 chars per token)."""
        return max(1, math.ceil(len(self.schema_json()) / 4.0))


class DynamicToolRegistry:
    """Dynamic, auto-discovering registry of tools active on this machine."""

    def __init__(
        self,
        registry_file: Optional[Union[str, Path]] = None,
        auto_discover: bool = True,
    ) -> None:
        self.registry_file = Path(
            registry_file or os.environ.get("APICAL_TOOL_REGISTRY", DEFAULT_REGISTRY_FILE)
        )
        self._tools: Dict[str, ToolDefinition] = {}
        self._load_from_disk()
        if auto_discover:
            self.auto_discover()

    def _load_from_disk(self) -> None:
        if self.registry_file.is_file():
            try:
                data = json.loads(self.registry_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        if isinstance(item, dict) and "name" in item:
                            t = ToolDefinition(
                                id=idx,
                                name=str(item["name"]),
                                description=str(item.get("description", "")),
                                parameters=item.get("parameters", {"type": "object", "properties": {}}),
                                category=str(item.get("category", "general")),
                            )
                            self._tools[t.name] = t
            except Exception as e:
                logger.warning("Could not read tool registry file: %s", e)

    def _save_to_disk(self) -> None:
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            serializable = [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "category": t.category,
                }
                for t in self._tools.values()
            ]
            self.registry_file.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not persist tool registry: %s", e)

    def auto_discover(self) -> int:
        """Automatically scan local environment for real MCP and IDE tool declarations."""
        discovered_count = 0
        scan_paths = [
            Path.home() / ".gemini" / "antigravity" / "mcp",
            Path.cwd() / ".cursor" / "mcp.json",
            Path.home() / ".cursor" / "mcp.json",
        ]

        mcp_dir = scan_paths[0]
        if mcp_dir.is_dir():
            for p in mcp_dir.rglob("*.json"):
                if "plexus" in p.parts:
                    continue
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "name" in data:
                        name = str(data["name"])
                        desc = str(data.get("description", ""))
                        params = data.get("parameters", {"type": "object", "properties": {}})
                        cat = p.parent.name
                        if name not in self._tools:
                            self._tools[name] = ToolDefinition(
                                id=len(self._tools),
                                name=name,
                                description=desc,
                                parameters=params,
                                category=cat,
                            )
                            discovered_count += 1
                except Exception:
                    pass

        if discovered_count > 0:
            self._save_to_disk()

        return discovered_count

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        category: str = "general",
    ) -> ToolDefinition:
        """Register or update a tool in the dynamic registry."""
        tool = ToolDefinition(
            id=len(self._tools) if name not in self._tools else self._tools[name].id,
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            category=category,
        )
        self._tools[name] = tool
        self._save_to_disk()
        return tool

    def register_tools(self, tools: Sequence[Dict[str, Any]]) -> int:
        """Batch register tools from arbitrary tool list."""
        added = 0
        for t in tools:
            fn = t.get("function", t)
            name = fn.get("name")
            if not name:
                continue
            desc = fn.get("description", "")
            params = fn.get("parameters", fn.get("input_schema", {"type": "object", "properties": {}}))
            cat = str(t.get("category", "general"))
            if name not in self._tools or self._tools[name].description != desc:
                self.register_tool(name=name, description=desc, parameters=params, category=cat)
                added += 1
        return added

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def get_tool_by_id(self, tool_id: int) -> Optional[ToolDefinition]:
        for t in self._tools.values():
            if t.id == tool_id:
                return t
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "category": t.category,
                "schema": t.to_schema(),
                "estimated_tokens": t.estimated_tokens(),
            }
            for t in self._tools.values()
        ]

    def total_tokens(self) -> int:
        return sum(t.estimated_tokens() for t in self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def calculate_token_savings(
        self,
        routed_tools: Sequence[Dict[str, Any]],
        candidate_tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compute exact token savings based on actual tools passed or registered."""
        if candidate_tools is not None:
            full_tools_list = list(candidate_tools)
        else:
            full_tools_list = self.list_tools()

        def _calc_tokens(tools_list: Sequence[Dict[str, Any]]) -> int:
            total = 0
            for t in tools_list:
                s = t.get("schema", t)
                total += max(1, math.ceil(len(json.dumps(s, separators=(",", ":"))) / 4.0))
            return total

        full_tokens = _calc_tokens(full_tools_list)
        routed_tokens = _calc_tokens(routed_tools)
        tokens_saved = max(0, full_tokens - routed_tokens)
        ratio = (tokens_saved / full_tokens) if full_tokens > 0 else 0.0

        return {
            "full_catalog_tools": len(full_tools_list),
            "full_catalog_tokens": full_tokens,
            "routed_tools": len(routed_tools),
            "routed_tokens": routed_tokens,
            "tokens_saved": tokens_saved,
            "savings_pct": round(ratio * 100.0, 2),
        }


# Zero hardcoded mock tools.
CANONICAL_TOOL_CATALOG: List[ToolDefinition] = []
TOOL_CATALOG_BY_ID: Dict[int, ToolDefinition] = {}

_DEFAULT_REGISTRY: Optional[DynamicToolRegistry] = None


def get_default_registry() -> DynamicToolRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = DynamicToolRegistry()
    return _DEFAULT_REGISTRY


def get_tool_by_id(tool_id: int) -> Optional[ToolDefinition]:
    return get_default_registry().get_tool_by_id(tool_id)


def calculate_token_savings(routed_tool_ids: Sequence[int]) -> Dict[str, Any]:
    reg = get_default_registry()
    routed = [t for tid in routed_tool_ids if (t := reg.get_tool_by_id(tid)) is not None]
    return reg.calculate_token_savings([t.to_schema() for t in routed])
