"""Top-level CLI router for `python -m apical` and `apical` command."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        from apical.mcp.__main__ import main as mcp_main

        sys.argv.pop(1)
        mcp_main()
    else:
        print("Apical Labs SDK CLI")
        print("Commands:")
        print("  apical mcp setup [--dry-run] [--force] [--uninstall]")
        print("  apical mcp [--dashboard] [--stats] [--wipe]")
        sys.exit(0)


if __name__ == "__main__":
    main()
