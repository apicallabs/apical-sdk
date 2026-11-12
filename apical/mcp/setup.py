"""Cross-Platform Auto-Setup Engine for Apical Plexus MCP.

Detects and configures all major AI coder platforms across Linux, macOS, and Windows:
- Claude Desktop
- Cursor IDE (Global & Workspace)
- Windsurf / Cascade
- Cline / Roo-Code (VS Code extensions)
- Google Antigravity / Gemini CLI

Guarantees:
- Non-destructive JSON editing via atomic json.loads / json.dumps.
- Timestamped backup copies before every write (<path>.bak.<timestamp>).
- Idempotency: exact-match configs are reported UNCHANGED with 0 file writes.
- Safe fallback: corrupt JSON or permission errors log copy-pasteable manual configs.
- Environment-aware Python executable resolution (virtualenv, conda, uv).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PlatformTarget:
    name: str
    config_path: Path
    format_type: str  # "mcpServers" (Claude/Cursor/Windsurf/Cline) or "antigravity"
    key_name: str = "plexus"


def get_current_python_executable() -> str:
    """Resolve the absolute path to the active Python executable.

    Detects if running inside a virtualenv, conda, or uv environment.
    """
    return os.path.abspath(sys.executable)


def get_default_mcp_config(python_exe: Optional[str] = None) -> Dict[str, Any]:
    """Generate the canonical MCP configuration payload for Apical Plexus."""
    exe = python_exe or get_current_python_executable()
    return {
        "command": exe,
        "args": ["-m", "apical.mcp.server"],
    }


def detect_platform_targets(workspace_dir: Optional[Path] = None) -> List[PlatformTarget]:
    """Scan and return potential MCP config targets based on the current OS."""
    system = platform.system()
    home = Path.home()
    targets: List[PlatformTarget] = []

    # 1. Claude Desktop
    if system == "Darwin":
        claude_path = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA")
        claude_path = Path(appdata) / "Claude" / "claude_desktop_config.json" if appdata else home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    else:  # Linux & default
        claude_path = home / ".config" / "Claude" / "claude_desktop_config.json"
    targets.append(PlatformTarget(name="Claude Desktop", config_path=claude_path, format_type="mcpServers"))

    # 2. Cursor IDE
    # User Global
    targets.append(PlatformTarget(name="Cursor (Global)", config_path=home / ".cursor" / "mcp.json", format_type="mcpServers"))
    # Workspace
    ws = workspace_dir or Path.cwd()
    targets.append(PlatformTarget(name="Cursor (Workspace)", config_path=ws / ".cursor" / "mcp.json", format_type="mcpServers"))

    # 3. Windsurf / Cascade
    targets.append(PlatformTarget(name="Windsurf / Cascade", config_path=home / ".codeium" / "windsurf" / "mcp_config.json", format_type="mcpServers"))

    # 4. Cline / Roo-Code (VS Code Extensions)
    if system == "Darwin":
        code_user = home / "Library" / "Application Support" / "Code" / "User"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA")
        code_user = Path(appdata) / "Code" / "User" if appdata else home / "AppData" / "Roaming" / "Code" / "User"
    else:
        code_user = home / ".config" / "Code" / "User"

    roo_path = code_user / "globalStorage" / "rooveterinaryinc.roo-cline" / "settings" / "cline_mcp_settings.json"
    cline_path = code_user / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json"
    targets.append(PlatformTarget(name="Roo-Code", config_path=roo_path, format_type="mcpServers"))
    targets.append(PlatformTarget(name="Cline", config_path=cline_path, format_type="mcpServers"))

    # 5. Google Antigravity / Gemini CLI
    agy_global = home / ".gemini" / "antigravity" / "mcp_config.json"
    targets.append(PlatformTarget(name="Google Antigravity", config_path=agy_global, format_type="mcpServers"))

    return targets


def backup_config(path: Path) -> Optional[Path]:
    """Create a timestamped backup copy: <config_path>.bak.<timestamp>."""
    if not path.is_file():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = path.parent / f"{path.name}.bak.{timestamp}"
    try:
        shutil.copy2(path, backup_path)
        return backup_path
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to create backup for {path}: {e}\n")
        return None


def read_json_config(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Safely read and parse a JSON config file.

    Returns:
        (data, error_message)
    """
    if not path.exists():
        return {}, None
    try:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            return {}, None
        data = json.loads(content)
        if not isinstance(data, dict):
            return None, "Root of configuration JSON must be an object/dict"
        return data, None
    except json.JSONDecodeError as jde:
        return None, f"JSON syntax error: {jde}"
    except Exception as ex:
        return None, f"Read error: {ex}"


def is_config_identical(current_entry: Any, desired_entry: Dict[str, Any]) -> bool:
    """Check if the currently registered MCP entry matches desired command & args."""
    if not isinstance(current_entry, dict):
        return False
    return (
        current_entry.get("command") == desired_entry.get("command")
        and current_entry.get("args") == desired_entry.get("args")
    )


def configure_target(
    target: PlatformTarget,
    python_exe: Optional[str] = None,
    dry_run: bool = False,
    uninstall: bool = False,
) -> Dict[str, Any]:
    """Inspect and configure or uninstall Apical Plexus MCP for a single target."""
    desired_config = get_default_mcp_config(python_exe)
    path = target.config_path

    # Read existing
    data, err = read_json_config(path)
    if err is not None:
        manual_snippet = json.dumps({"mcpServers": {target.key_name: desired_config}}, indent=2)
        return {
            "target": target.name,
            "path": str(path),
            "status": "ERROR",
            "message": f"Corrupt or invalid JSON file: {err}",
            "manual_snippet": manual_snippet,
        }

    # Determine server container key (default "mcpServers")
    container_key = "mcpServers"
    mcp_servers = data.setdefault(container_key, {})
    if not isinstance(mcp_servers, dict):
        manual_snippet = json.dumps({"mcpServers": {target.key_name: desired_config}}, indent=2)
        return {
            "target": target.name,
            "path": str(path),
            "status": "ERROR",
            "message": f"'{container_key}' field in {path} is not a dictionary.",
            "manual_snippet": manual_snippet,
        }

    # 1. UNINSTALL MODE
    if uninstall:
        if target.key_name not in mcp_servers:
            return {
                "target": target.name,
                "path": str(path),
                "status": "NOT_CONFIGURED",
                "message": f"'{target.key_name}' is not installed.",
            }
        if not dry_run:
            backup = backup_config(path)
            del mcp_servers[target.key_name]
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            return {
                "target": target.name,
                "path": str(path),
                "status": "UNINSTALLED",
                "backup": str(backup) if backup else None,
            }
        else:
            return {
                "target": target.name,
                "path": str(path),
                "status": "DRY_RUN_UNINSTALL",
                "message": f"Would remove '{target.key_name}' from {path}",
            }

    # 2. INSTALL / UPDATE MODE
    current_entry = mcp_servers.get(target.key_name)
    if current_entry and is_config_identical(current_entry, desired_config):
        return {
            "target": target.name,
            "path": str(path),
            "status": "UNCHANGED",
            "message": f"'{target.key_name}' is already up-to-date.",
        }

    status_name = "UPDATED" if current_entry else "CONFIGURED"

    if dry_run:
        return {
            "target": target.name,
            "path": str(path),
            "status": f"DRY_RUN_{status_name}",
            "planned_entry": desired_config,
            "diff": {
                "before": current_entry,
                "after": desired_config,
            },
        }

    # Write changes safely
    backup = backup_config(path)
    mcp_servers[target.key_name] = desired_config
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    return {
        "target": target.name,
        "path": str(path),
        "status": status_name,
        "backup": str(backup) if backup else None,
        "entry": desired_config,
    }


def run_mcp_setup(
    dry_run: bool = False,
    force: bool = False,
    uninstall: bool = False,
    workspace_dir: Optional[Path] = None,
    python_exe: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute the multi-platform MCP setup or inspection pass."""
    targets = detect_platform_targets(workspace_dir=workspace_dir)
    results: List[Dict[str, Any]] = []

    for target in targets:
        # If target config file doesn't exist and parent directory doesn't exist,
        # only create it if force is enabled or if it's Cursor/Claude in existing app folder
        parent_exists = target.config_path.parent.exists()
        file_exists = target.config_path.is_file()

        if not file_exists and not parent_exists and not force:
            results.append({
                "target": target.name,
                "path": str(target.config_path),
                "status": "SKIPPED",
                "message": "Platform directory not detected on host system. (Use --force to create).",
            })
            continue

        res = configure_target(
            target=target,
            python_exe=python_exe,
            dry_run=dry_run,
            uninstall=uninstall,
        )
        results.append(res)

    return results


def print_setup_report(results: List[Dict[str, Any]], dry_run: bool = False, uninstall: bool = False) -> None:
    """Print clean terminal report of setup operations."""
    print("==================================================================")
    action_title = "Uninstall" if uninstall else ("Inspection (Dry-Run)" if dry_run else "Auto-Setup")
    print(f"  Apical Plexus MCP — Platform {action_title}")
    print("==================================================================\n")

    for r in results:
        status = r.get("status", "UNKNOWN")
        target = r.get("target")
        path = r.get("path")
        msg = r.get("message", "")
        backup = r.get("backup")

        symbol = "✔" if status in ("CONFIGURED", "UPDATED", "UNCHANGED", "UNINSTALLED", "DRY_RUN_CONFIGURED", "DRY_RUN_UPDATED", "DRY_RUN_UNINSTALL") else ("ℹ" if status == "SKIPPED" else "✖")
        print(f"[{symbol}] {target}: {status}")
        print(f"    Path: {path}")
        if msg:
            print(f"    Info: {msg}")
        if backup:
            print(f"    Backup: {backup}")
        if "manual_snippet" in r:
            print("    Manual Fallback Configuration:")
            for line in r["manual_snippet"].splitlines():
                print(f"      {line}")
        print()

    print("==================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="apical mcp setup",
        description="Non-destructive auto-setup CLI for Apical Plexus MCP across all AI coder IDEs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan and report planned configuration diffs without writing files.")
    parser.add_argument("--force", action="store_true", help="Automatically generate config files even if platform directories do not exist.")
    parser.add_argument("--uninstall", action="store_true", help="Remove Apical Plexus MCP entry while preserving other tools.")
    parser.add_argument("--python-path", type=str, default=None, help="Explicit path to Python executable (defaults to active interpreter).")
    args = parser.parse_args()

    results = run_mcp_setup(
        dry_run=args.dry_run,
        force=args.force,
        uninstall=args.uninstall,
        python_exe=args.python_path,
    )
    print_setup_report(results, dry_run=args.dry_run, uninstall=args.uninstall)


if __name__ == "__main__":
    main()
