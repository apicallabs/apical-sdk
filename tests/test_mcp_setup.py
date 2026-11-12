"""Comprehensive Test Suite for Apical Plexus MCP Auto-Setup Engine."""

import json
from pathlib import Path
import tempfile
import unittest

from apical.mcp.setup import (
    PlatformTarget,
    configure_target,
    detect_platform_targets,
    get_default_mcp_config,
    read_json_config,
    run_mcp_setup,
)


class TestMcpSetup(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_detect_platform_targets_returns_major_platforms(self):
        targets = detect_platform_targets(workspace_dir=self.tmp_path)
        names = [t.name for t in targets]
        self.assertIn("Claude Desktop", names)
        self.assertIn("Cursor (Global)", names)
        self.assertIn("Cursor (Workspace)", names)
        self.assertIn("Windsurf / Cascade", names)
        self.assertIn("Google Antigravity", names)

    def test_configure_target_fresh_install(self):
        config_file = self.tmp_path / "claude_desktop_config.json"
        target = PlatformTarget(name="Test Platform", config_path=config_file, format_type="mcpServers")

        python_exe = "/usr/bin/python3"
        res = configure_target(target, python_exe=python_exe, dry_run=False)

        self.assertEqual(res["status"], "CONFIGURED")
        self.assertTrue(config_file.is_file())

        data, err = read_json_config(config_file)
        self.assertIsNone(err)
        self.assertIn("mcpServers", data)
        self.assertIn("plexus", data["mcpServers"])
        self.assertEqual(data["mcpServers"]["plexus"]["command"], python_exe)
        self.assertEqual(data["mcpServers"]["plexus"]["args"], ["-m", "apical.mcp.server"])

    def test_configure_target_idempotency_unchanged(self):
        config_file = self.tmp_path / "cursor_mcp.json"
        target = PlatformTarget(name="Cursor", config_path=config_file, format_type="mcpServers")
        python_exe = "/usr/bin/python3"

        # Pass 1: Fresh install
        res1 = configure_target(target, python_exe=python_exe, dry_run=False)
        self.assertEqual(res1["status"], "CONFIGURED")

        # Pass 2: Exactly identical
        res2 = configure_target(target, python_exe=python_exe, dry_run=False)
        self.assertEqual(res2["status"], "UNCHANGED")

    def test_configure_target_updates_existing_differing_entry(self):
        config_file = self.tmp_path / "mcp.json"
        initial_data = {
            "mcpServers": {
                "other_tool": {"command": "node", "args": ["tool.js"]},
                "plexus": {"command": "old_python", "args": ["-m", "old"]},
            }
        }
        config_file.write_text(json.dumps(initial_data), encoding="utf-8")
        target = PlatformTarget(name="Custom Target", config_path=config_file, format_type="mcpServers")

        res = configure_target(target, python_exe="/new/python", dry_run=False)
        self.assertEqual(res["status"], "UPDATED")
        self.assertIsNotNone(res["backup"])
        self.assertTrue(Path(res["backup"]).is_file())

        # Verify other_tool preserved
        data, _ = read_json_config(config_file)
        self.assertIn("other_tool", data["mcpServers"])
        self.assertEqual(data["mcpServers"]["plexus"]["command"], "/new/python")

    def test_configure_target_dry_run_does_not_modify_disk(self):
        config_file = self.tmp_path / "dry_run_test.json"
        target = PlatformTarget(name="Dry Run Target", config_path=config_file, format_type="mcpServers")

        res = configure_target(target, python_exe="/usr/bin/python3", dry_run=True)
        self.assertEqual(res["status"], "DRY_RUN_CONFIGURED")
        self.assertFalse(config_file.exists(), "Dry-run must not create or modify files on disk!")

    def test_configure_target_corrupted_json_handled_safely(self):
        config_file = self.tmp_path / "corrupted.json"
        config_file.write_text("{ unclosed json string ...", encoding="utf-8")
        target = PlatformTarget(name="Corrupted Target", config_path=config_file, format_type="mcpServers")

        res = configure_target(target, python_exe="/usr/bin/python3", dry_run=False)
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("Corrupt or invalid JSON", res["message"])
        self.assertIn("manual_snippet", res)
        self.assertIn("mcpServers", res["manual_snippet"])

    def test_configure_target_uninstall(self):
        config_file = self.tmp_path / "uninstall_test.json"
        initial_data = {
            "mcpServers": {
                "keeper": {"command": "keeper"},
                "plexus": {"command": "plexus"},
            }
        }
        config_file.write_text(json.dumps(initial_data), encoding="utf-8")
        target = PlatformTarget(name="Target", config_path=config_file, format_type="mcpServers")

        res = configure_target(target, uninstall=True, dry_run=False)
        self.assertEqual(res["status"], "UNINSTALLED")
        self.assertIsNotNone(res["backup"])

        data, _ = read_json_config(config_file)
        self.assertNotIn("plexus", data["mcpServers"])
        self.assertIn("keeper", data["mcpServers"])


if __name__ == "__main__":
    unittest.main()
