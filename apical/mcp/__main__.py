"""CLI entry point for python -m apical.mcp."""

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        from apical.mcp.setup import main as setup_main

        # Strip 'setup' so setup argparse parses remaining flags
        sys.argv.pop(1)
        setup_main()
    else:
        from apical.mcp.server import main as server_main

        server_main()


if __name__ == "__main__":
    main()

